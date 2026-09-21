"""12 - ALLINEA LA CACHE all'universo corrente.

Offline, ZERO chiamate Bloomberg.

PERCHE' SERVE. enrich_reference e' incrementale e non toglie mai nulla: fa
    out = old if old is not None else ...
quindi le righe dei titoli RIMOSSI dal file universo restano in ref_*.parquet per
sempre. E pipeline.build_market legge i nominali DALLA CACHE, non dall'universo:
    ref_n = bbg.load("ref_nominal");  ref_n = ref_n[ref_n["mkt"] == nom_mkt]
Risultato: togliere un callable/puttable/FRN dall'Excel non ha alcun effetto finche'
la cache non viene ripulita -- match_nominals continua a pescarlo e il fit NSS a
includerlo. Questo script chiude il buco.

COSA TOCCA
  ref_linker / ref_nominal   righe (index = ISIN)
  px_mid_* px_bid_* px_ask_* colonne (prezzi linker)
  ytm_* px_nom_*             colonne (nominali)
NON tocca i pannelli delle basi: 04 li riscrive interi, non li fonde.

Lancia prima 01_universe.py, che rigenera universe.parquet.
DRY_RUN=True stampa e basta. Rileggi l'elenco, poi metti False.
"""
import pandas as pd
import bbg
from config import CACHE

# Colonne che tradiscono un frame LONG (tidy) fuso per sbaglio dentro un pannello wide:
# _update_wide fa out.combine_first(df) senza verificare la forma, quindi se _bdh torna
# long (flakiness narwhals, la stessa che _bdp gia' gestisce con il retry per singolo)
# queste finiscono nel pannello come se fossero ISIN. Non sono dati mancanti: sono
# rumore AGGIUNTO. Vanno tolte, ma vanno distinte dagli ISIN rimossi davvero.
LONG_COLS = {"date", "field", "ticker", "value", "security", "securities"}

# Lo stesso merge sporca anche l'INDICE: il frame long ha un RangeIndex (0,1,2,...) che
# combine_first converte in datetime dall'epoch, quindi il pannello si porta dietro
# decine di migliaia di righe a 1970-01-01 + k nanosecondi. Verificato su ytm_DE: 37.275
# righe spurie su 43.428, con ZERO dati nelle colonne ISIN. Sono innocue -- build_market
# usa px.index.intersection(ytm.index) e px_mid e' pulito -- ma sono 7x memoria e I/O.
# PULL_FLOOR e' 2003-01-01: in un pannello di prezzi/rendimenti obbligazionari qualunque
# data prima del 1990 e' spazzatura per costruzione.
ROW_FLOOR = pd.Timestamp("1990-01-01")

# ----------------------------------------------------------------- impostazioni
DRY_RUN = False

# ----------------------------------------------------------------- esecuzione
up = CACHE / "universe.parquet"
if not up.exists():
    raise SystemExit(f"manca {up}\n-> lancia prima 01_universe.py")
uni = pd.read_parquet(up)
keep = set(uni.loc[uni["incl"], "isin"].astype(str))
print(f"universo: {len(uni)} strumenti, {len(keep)} inclusi\n")

tot_r = tot_c = tot_rw = 0

for kind in ("linker", "nominal"):
    p = CACHE / f"ref_{kind}.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    bad = [i for i in df.index.astype(str) if i not in keep]
    print(f"ref_{kind:<8} {len(df):5d} righe -> {len(bad):4d} da togliere")
    if bad:
        ex = df.loc[df.index.astype(str).isin(bad)]
        col = "SECURITY_NAME" if "SECURITY_NAME" in ex.columns else None
        for i in bad[:6]:
            nm = str(ex.at[i, col])[:40] if col else ""
            print(f"    {i}  {nm}")
        if len(bad) > 6:
            print(f"    ... e altri {len(bad)-6}")
        tot_r += len(bad)
        if not DRY_RUN:
            df[~df.index.astype(str).isin(bad)].to_parquet(p)

corrotti = []
for p in sorted(CACHE.glob("*.parquet")):
    n = p.stem
    if not n.startswith(("px_mid_", "px_bid_", "px_ask_", "px_nom_", "ytm_")):
        continue
    df = pd.read_parquet(p)
    cols = [str(c) for c in df.columns]
    lng = [c for c in cols if c.lower() in LONG_COLS]
    if lng and len(lng) == len(cols):
        corrotti.append(n)
        print(f"{n:<20} {len(cols):5d} colonne, TUTTE long ({', '.join(cols)})")
        print(f"      -> file in formato LONG, inutilizzabile come pannello wide.")
        print(f"         Non lo tocco: cancellalo a mano se non ti serve.")
        continue
    if lng:
        print(f"{n:<20} {len(lng)} colonne SPURIE da merge long: {', '.join(lng)}")
        print(f"      -> rumore aggiunto, non dati persi: i {len(cols)-len(lng)} ISIN restano")
    junk_rows = 0
    if isinstance(df.index, pd.DatetimeIndex):
        junk_rows = int((df.index < ROW_FLOOR).sum())
        if junk_rows:
            print(f"{n:<20} {junk_rows} RIGHE spurie (indice < {ROW_FLOOR:%Y}) su {len(df)}"
                  f" -> {len(df)-junk_rows} restano")
            nz = int(df.loc[df.index < ROW_FLOOR,
                            [c for c in df.columns if str(c).lower() not in LONG_COLS]]
                     .notna().sum().sum())
            print(f"      dati utili in quelle righe: {nz}"
                  f"{'  (nessuno: scarto sicuro)' if nz == 0 else '  <<< CONTROLLA'}")
            tot_rw += junk_rows
    bad = [c for c in cols if c not in keep and c.lower() not in LONG_COLS]
    bad = lng + bad
    if (bad or junk_rows) and not DRY_RUN:
        d2 = df
        if junk_rows:
            d2 = d2[d2.index >= ROW_FLOOR]
        if bad:
            d2 = d2.drop(columns=[c for c in d2.columns if str(c) in set(bad)])
        d2.to_parquet(p)
    if bad:
        resta = len(df.columns) - len(bad)
        flag = "   <<< SVUOTA IL FILE" if resta == 0 else ""
        print(f"{n:<20} {len(df.columns):5d} colonne -> {len(bad):4d} da togliere "
              f"({resta} restano){flag}")
        # i nomi: senza questi non si distingue un ISIN tolto apposta da un residuo
        # di un run vecchio con un universo diverso.
        for c in bad[:8]:
            nz = int(df[c].notna().sum()) if c in df.columns else 0
            tag = "  [spuria long]" if c.lower() in LONG_COLS else "  [ISIN rimosso]"
            print(f"      {c}   {nz} valori non nulli{tag}")
        if len(bad) > 8:
            print(f"      ... e altri {len(bad)-8}")
        tot_c += len(bad)

print(f"\ntotale: {tot_r} righe di anagrafica, {tot_c} colonne e {tot_rw} righe di storico")
if corrotti:
    print(f"\n{len(corrotti)} file interamente in formato LONG, lasciati intatti:")
    print("  " + ", ".join(corrotti))
    print("  Se non ti servono (px_ask/px_bid servono solo alla versione trading):")
    print("    Remove-Item .\\data\\cache\\px_ask_*.parquet, .\\data\\cache\\px_bid_*.parquet")
if DRY_RUN:
    print("DRY_RUN=True: nulla e' stato modificato. Controlla l'elenco, poi metti False.")
else:
    print("cache allineata. Rigenera i pannelli con 04_basis_markets.py.")
