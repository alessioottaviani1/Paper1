"""15 - Z-SPREAD BLOOMBERG: scarico incrementale, resumibile, guidato dai dati che ci sono gia'.

RICHIEDE IL TERMINALE. A Bloomberg si chiede un campo solo, CAMPO, e nient'altro. I prezzi
NON si riscaricano: px_mid_<MKT>.parquet e ytm_<MKT>.parquet sono gia' in cache e qui
servono in lettura, per decidere COSA e QUANDO chiedere.

CHI SI SCARICA. Non un pool per euristica di distanza, ma i titoli che il matching sceglie
DAVVERO. La griglia della pipeline e' px.index & ytm.index (build_market), e i due pannelli
dicono esattamente quali titoli erano quotati in quale giorno: si simula qui, offline e a
costo zero, la scelta di lower/upper/nearest per ogni (data, linker) e si tiene l'unione.
E' la risposta esatta, non un'approssimazione, ed e' molto piu' stretta: un nominale a tre
anni di distanza che non viene mai selezionato non ha ragione di entrare nello scarico.

QUANDO. Le finestre sono i giorni in cui quel titolo ha davvero un prezzo sulla griglia.
Un anno senza prezzi non si chiede: uno Z-spread su una data senza prezzo non produrrebbe
comunque nessuna osservazione di base.

PERCHE' NON _update_wide. Quello considera COMPLETA una colonna che esiste, e la estende
solo in avanti da out.index.max(). Ma la doc del campo avverte: "the date range may
automatically be adjusted to a shorter period". Una richiesta 2003-2026 che tornasse solo
2018-2026 chiuderebbe quelle colonne per sempre, e il buco a monte non si riempirebbe mai,
in silenzio. Qui l'unita' di lavoro e' la coppia (titolo, ANNO).

COME RIPARTE. Il parquet viene riscritto dopo OGNI blocco. Un limite Bloomberg a meta'
costa al massimo l'ultimo blocco: si rilancia e riprende dai buchi rimasti.

VUOTO != BLOCCATO. Un titolo puo' non avere Z-spread in un certo anno: ritrattarlo come "da
riscaricare" farebbe girare lo script all'infinito. Ma "nessun dato" e' anche la faccia che
mostra un blocco. Il discriminante e' il resto del blocco: se altri titoli hanno risposto,
Bloomberg e' vivo e la coppia si segna come fatta. Se il blocco e' vuoto per intero
MAX_VUOTI volte di fila si interroga un CANARINO (una serie che deve esistere sempre): se
risponde i vuoti sono veri, se tace e' un limite giornaliero -> checkpoint e uscita.

VERIFICA FINALE. Avendo i prezzi come riferimento, la copertura non si giudica a occhio:
per ogni titolo si stampa quanti Z-spread sono arrivati rispetto ai giorni in cui il
titolo aveva un prezzo. Sotto SOGLIA_COP e' un troncamento, e si vede subito.

SCELTA DEL CAMPO. Z_SPRD_MID (SP037) e' pieno sui nominali e VUOTO sui linker: per gli
index-linked quell'analitica non viene stoccata. INDEX_Z_SPREAD_BP (BX326) ha storico su
entrambe le gambe, e a bdp sullo STESSO nominale nello stesso istante torna lo stesso
numero di SP037 (68.9 vs 68.9): non e' un campo alternativo con una convenzione propria,
e' lo stesso Z-spread sulla stessa curva swap, implementato in modo da coprire anche i
titoli i cui flussi vanno prima proiettati con l'inflazione.

Quindi BX326 si usa su ENTRAMBE le gambe, non una per campo. Mettere SP037 sul nominale e
BX326 sul linker farebbe rientrare dalla finestra il vizio dell'ASW differenziale: due
convenzioni diverse messe a sottrarre, con lo scarto fra definizioni che finisce dentro la
base. Con un campo solo quel problema non esiste.

Il pannello SP037 gia' scaricato non si butta: a fine run, se CONTROLLO e' valorizzato e
quel pannello esiste, si confrontano le due misure sulle celle (data, titolo) in comune.
Se coincidono su vent'anni di nominali, il cambio di campo e' validato sui dati invece che
sulla lettura di una definizione.
"""
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import bbg
from config import MARKETS, CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO    = "IT"
CAMPO      = "Z_SPRD_MID"          # SP037 @ BGN: l'unica combinazione che riproduce lo schermo
CONTROLLO  = ""                    # il pannello BX326 e' da buttare, non da confrontare
SOURCE     = "BGN"            # BGN: a CBBT questo campo non ha storico sui linker
ANNO_DA    = 2003
CHUNK      = 8                # titoli per richiesta: il campo si ricalcola ogni volta, e' lento
THROTTLE   = 0.5
MAX_VUOTI  = 3                # blocchi vuoti di fila prima di interrogare il canarino
SOGLIA_COP = 0.50             # copertura z/prezzo sotto cui segnalare il titolo

# ----------------------------------------------------------------- esecuzione
try:
    from xbbg import blp  # noqa: F401   solo per accertare che siamo sul terminale
except ImportError:
    raise SystemExit("xbbg non disponibile: questo script va lanciato sul terminale.")

m       = MARKETS[MERCATO]
nom_mkt = bbg.NOMINAL_POOL_ALIAS.get(MERCATO, MERCATO)
# il campo entra nel NOME del file: pannelli di campi diversi non si mescolano mai, e
# quello gia' scaricato resta intatto come controllo.
path    = CACHE / f"zsprd_{MERCATO}_{CAMPO}.parquet"
path_f  = CACHE / f"zsprd_{MERCATO}_{CAMPO}_fatti.parquet"
oggi    = pd.Timestamp.today().normalize()


def _mat(df: pd.DataFrame) -> pd.Series:
    """Scadenza: 'maturity' (dal file universo) con ripiego su MATURITY (bdp)."""
    s = pd.to_datetime(df["maturity"], errors="coerce") if "maturity" in df.columns \
        else pd.Series(pd.NaT, index=df.index)
    if "MATURITY" in df.columns:
        s = s.fillna(pd.to_datetime(df["MATURITY"], errors="coerce"))
    return s


ref_l = bbg.load("ref_linker");  ref_l = ref_l[ref_l["mkt"] == MERCATO]
ref_n = bbg.load("ref_nominal"); ref_n = ref_n[ref_n["mkt"] == nom_mkt]
if not len(ref_l):
    raise SystemExit(f"nessun linker {MERCATO} in ref_linker.parquet: lancia prima 01/02.")

# --- 1. la griglia su cui vive la pipeline (build_market) --------------------------
px  = bbg.load(f"px_mid_{MERCATO}")
ytm = bbg.load(f"ytm_{nom_mkt}")
px.index, ytm.index = pd.to_datetime(px.index), pd.to_datetime(ytm.index)
inizio = max(pd.Timestamp(f"{ANNO_DA}-01-01"), bbg.PULL_FLOOR)
grid = px.index.intersection(ytm.index).sort_values()
grid = grid[(grid >= inizio) & (grid <= oggi)]
if not len(grid):
    raise SystemExit("px_mid e ytm non si intersecano: controlla la cache.")
px, ytm = px.loc[grid], ytm.loc[grid]

mat_l = _mat(ref_l).reindex(px.columns).dropna()
mat_n = _mat(ref_n).reindex(ytm.columns).dropna()
px  = px[[c for c in px.columns if c in mat_l.index]]
ytm = ytm[[c for c in ytm.columns if c in mat_n.index]]

print(f"=== {MERCATO}: griglia {grid.min():%Y-%m-%d} -> {grid.max():%Y-%m-%d} "
      f"({len(grid)} date)")
print(f"    pannelli letti (NON si riscaricano): px_mid_{MERCATO} ({px.shape[1]} linker), "
      f"ytm_{nom_mkt} ({ytm.shape[1]} nominali)")

# --- 2. quali nominali il matching sceglie DAVVERO, data per data -------------------
# lower/upper = i due che bracciano la scadenza del linker FRA QUELLI QUOTATI QUEL GIORNO.
# Si simula sulla griglia vera invece di approssimare con una distanza di scadenza: il
# bracket del 2006 non e' quello di oggi, perche' nel frattempo ne sono scaduti e ne sono
# stati emessi, e una soglia fissa o lascia buchi o gonfia lo scarico.
o      = np.argsort(mat_n.values)
n_cols = np.asarray(mat_n.index)[o]
n_mats = mat_n.values[o].astype("datetime64[D]").astype(int)
l_cols = np.asarray(mat_l.index)
l_mats = mat_l.values.astype("datetime64[D]").astype(int)
M_n = ytm[n_cols].notna().values
M_l = px[l_cols].notna().values

usati, largh, orfani = set(), [], 0
for t in range(len(grid)):
    idx = np.flatnonzero(M_n[t])
    jl  = np.flatnonzero(M_l[t])
    if not len(idx) or not len(jl):
        continue
    mats = n_mats[idx]
    pos  = np.searchsorted(mats, l_mats[jl])
    for k in range(len(jl)):
        a, b = pos[k] - 1, pos[k]
        ok_a, ok_b = a >= 0, b < len(mats)
        if ok_a:
            usati.add(n_cols[idx[a]])
        if ok_b:
            usati.add(n_cols[idx[b]])
        if ok_a and ok_b:
            largh.append(mats[b] - mats[a])
        else:
            orfani += 1

sel_n = [c for c in n_cols if c in usati]
print(f"    nominali selezionati dal matching: {len(sel_n)}/{ytm.shape[1]} "
      f"({ytm.shape[1] - len(sel_n)} non vengono mai scelti, esclusi)")
if largh:
    q = np.percentile(largh, [50, 90, 99])
    print(f"    ampiezza del bracket (giorni): mediana {q[0]:.0f}, p90 {q[1]:.0f}, "
          f"p99 {q[2]:.0f}   [MAX_MISMATCH_DAYS della pipeline = 183]")
if orfani:
    print(f"    {orfani} coppie (data, linker) senza bracket su un lato: "
          f"restano fuori dalla misura interpolata")

uni = pd.concat([
    pd.DataFrame({"mat": mat_l, "kind": "linker"}),
    pd.DataFrame({"mat": mat_n[sel_n], "kind": "nominal"}),
])
bbid = pd.concat([ref_l["bb_id"], ref_n["bb_id"]])
uni["tick"] = bbid.reindex(uni.index).astype(str).str.strip() + f"@{SOURCE} Corp"
uni = uni[uni["tick"].notna() & ~uni["tick"].str.startswith(("nan@", "None@"))]
print(f"    da chiedere a Bloomberg: {int((uni['kind']=='linker').sum())} linker + "
      f"{len(sel_n)} nominali = {len(uni)} titoli, campo {CAMPO} @ {SOURCE}")
print(f"    esempio ticker: {uni['tick'].iloc[0]}\n")

# --- 3. quando: i giorni in cui quel titolo ha DAVVERO un prezzo --------------------
cop = {c: px.index[px[c].notna()] for c in px.columns}
cop.update({c: ytm.index[ytm[c].notna()] for c in sel_n})
cop = {c: d for c, d in cop.items() if c in uni.index and len(d)}

# --- 4. stato: cosa e' gia' in cache, cosa e' gia' stato chiesto --------------------
panel = pd.read_parquet(path) if path.exists() else pd.DataFrame()
if len(panel):
    panel.index = pd.to_datetime(panel.index)
    print(f"    ripresa: {panel.shape[1]} colonne, {int(panel.notna().sum().sum())} valori "
          f"in cache ({panel.index.min():%Y-%m} -> {panel.index.max():%Y-%m})")

fatti: set = set()
if path_f.exists():
    _f = pd.read_parquet(path_f)
    fatti = set(zip(_f["isin"], _f["anno"].astype(int)))
    print(f"    {len(fatti)} coppie (titolo, anno) gia' richieste: non si ripetono")

# --- 5. cosa manca ------------------------------------------------------------------
# anno chiuso: fatto se e' nel registro o se ha gia' dei valori.
# anno in corso: mai chiuso, si riprende dal giorno dopo l'ultimo Z-spread in cache.
# La finestra e' l'ANNO INTERO, non il primo/ultimo giorno quotato del titolo: agganciarla
# ai bordi del singolo bond frammenterebbe le richieste in blocchi da uno o due titoli
# invece che da CHUNK, moltiplicando le chiamate. Fuori dalla vita del titolo Bloomberg
# non torna nulla e non costa nulla: e' la stessa, unica richiesta. Il risparmio vero sta
# altrove -- nell'universo ristretto e negli ANNI SENZA PREZZO, che non si chiedono affatto.
bordi = {y: (max(grid.min(), pd.Timestamp(f"{y}-01-01")),
             min(grid.max(), pd.Timestamp(f"{y}-12-31")))
         for y in range(grid.min().year, grid.max().year + 1)}

lavoro = defaultdict(list)           # (anno, da, a) -> [isin, ...]
n_todo = 0
for isin, giorni in cop.items():
    for y, d in giorni.groupby(giorni.year).items():
        if (isin, y) in fatti and y < oggi.year:
            continue
        y0, y1 = bordi[y]
        d0 = y0
        if isin in panel.columns:
            s = panel[isin]
            s = s[(s.index >= y0) & (s.index <= y1)].dropna()
            if len(s):
                if y < oggi.year:
                    continue
                d0 = s.index.max() + pd.Timedelta(days=1)
                if d0 > d.max():
                    continue          # nessun giorno QUOTATO nuovo: non si chiede
        lavoro[(y, d0, y1)].append(isin)
        n_todo += 1

print(f"\n    da scaricare: {n_todo} coppie (titolo, anno) in {len(lavoro)} finestre")
if n_todo:
    _pa = pd.Series([y for (y, _, _), v in lavoro.items() for _ in v]).value_counts().sort_index()
    print("    per anno: " + "  ".join(f"{y}:{n}" for y, n in _pa.items()) + "\n")
else:
    # NON si esce: il ciclo sotto gira a vuoto e si arriva lo stesso alla verifica e al
    # controllo incrociato, che non costano chiamate e vanno poter rilanciare da soli.
    print("    niente da scaricare: si passa direttamente alla verifica.\n")


def _save():
    if len(panel):
        panel.sort_index().to_parquet(path)
    if fatti:
        pd.DataFrame(sorted(fatti), columns=["isin", "anno"]).to_parquet(path_f)


def _canarino() -> bool:
    """Bloomberg risponde ancora? Una serie che deve esistere sempre, finestra corta."""
    try:
        c = bbg._bdh([m.ils_ticker(10)], "PX_LAST",
                     (oggi - pd.Timedelta(days=30)).date(), oggi.date())
        return c is not None and not c.empty
    except Exception:
        return False


t0, n_vuoti = time.time(), 0
sospesi: set = set()      # blocchi tornati vuoti INTERI: veri buchi o blocco? non si sa ancora
for (y, d0, d1) in sorted(lavoro):
    isins = lavoro[(y, d0, d1)]
    n_blk = -(-len(isins) // CHUNK)
    for k in range(0, len(isins), CHUNK):
        blk = isins[k:k + CHUNK]
        t2c = {uni.at[i, "tick"]: i for i in blk}
        try:
            df = bbg._bdh(list(t2c), CAMPO, d0.date(), d1.date())
        except Exception as e:
            print(f"  {y} blocco {k//CHUNK+1}: errore {str(e)[:70]} -> checkpoint ed esco")
            _save()
            raise SystemExit(f"\nSalvati {int(panel.notna().sum().sum())} valori in "
                             f"{path.name}. Rilancia: riprende dai buchi.")

        if df is None or df.empty:
            # Non si segna ancora niente: un blocco vuoto per intero e' esattamente cio'
            # che si vede sia quando quei titoli non hanno Z-spread, sia quando Bloomberg
            # ha smesso di rispondere. Segnarlo subito darebbe per chiesto cio' che non e'
            # mai stato chiesto davvero.
            n_vuoti += 1
            sospesi |= {(i, y) for i in blk}
            print(f"  {y} blocco {k//CHUNK+1}/{n_blk}: vuoto ({n_vuoti})")
            if n_vuoti >= MAX_VUOTI:
                print("    -> interrogo il canarino...", end=" ", flush=True)
                if _canarino():
                    print(f"risponde: {len(sospesi)} coppie sono vuote per davvero")
                    fatti |= sospesi
                    sospesi, n_vuoti = set(), 0
                    _save()
                else:
                    print("tace: limite Bloomberg")
                    sospesi = set()      # non erano vuote: non gliele abbiamo mai chieste
                    _save()
                    raise SystemExit(
                        f"\nBloomberg non risponde piu' (probabile limite giornaliero).\n"
                        f"Salvati {int(panel.notna().sum().sum())} valori in {path.name}.\n"
                        f"Rilancia domani: riprende esattamente dai buchi rimasti.")
            time.sleep(THROTTLE)
            continue

        # un blocco ha risposto: Bloomberg era vivo anche prima, i vuoti sospesi erano veri
        fatti |= sospesi | {(i, y) for i in blk}
        sospesi, n_vuoti = set(), 0

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [t2c.get(c, c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = pd.to_datetime(df.index)
        panel = df if not len(panel) else panel.combine_first(df)
        _save()                                     # <-- CHECKPOINT dopo OGNI blocco
        print(f"  {y} blocco {k//CHUNK+1}/{n_blk}: +{int(df.notna().sum().sum()):5d} valori"
              f"   tot {int(panel.notna().sum().sum()):7d}   ({time.time()-t0:.0f}s)")
        time.sleep(THROTTLE)

if sospesi:
    print(f"\n{len(sospesi)} coppie vuote in sospeso -> canarino...", end=" ")
    if _canarino():
        print("risponde: vuote per davvero")
        fatti |= sospesi
    else:
        print("tace: non le segno, verranno richieste al prossimo lancio")
_save()

if not len(panel):
    raise SystemExit("\nniente in cache e niente scaricato: non c'e' nulla da verificare.")

# ----------------------------------------------------------------- verifica finale
print(f"\nfatto: {panel.shape[1]} colonne, {int(panel.notna().sum().sum())} valori, "
      f"{panel.index.min():%Y-%m-%d} -> {panel.index.max():%Y-%m-%d}")

# Avendo i prezzi come riferimento la copertura non si giudica a occhio: per ogni titolo,
# quanti Z-spread sono arrivati sui giorni in cui quel titolo AVEVA un prezzo.
rows = []
for isin, giorni in cop.items():
    s = panel[isin].reindex(giorni).notna().sum() if isin in panel.columns else 0
    rows.append((isin, uni.at[isin, "kind"], len(giorni), int(s), int(s) / len(giorni)))
r = pd.DataFrame(rows, columns=["isin", "kind", "n_px", "n_z", "cop"]).set_index("isin")
r["mat"] = uni["mat"]

print(f"\ncopertura Z-spread sui giorni con prezzo: "
      f"mediana {r['cop'].median():.0%}, minimo {r['cop'].min():.0%}")
for kind in ("linker", "nominal"):
    g = r[r["kind"] == kind]
    if len(g):
        print(f"  {kind:8s} {len(g):4d} titoli, {int(g['n_z'].sum()):7d} valori, "
              f"copertura mediana {g['cop'].median():.0%}")

male = r[r["cop"] < SOGLIA_COP].sort_values("cop")
if len(male):
    print(f"\n  {len(male)} titoli sotto il {SOGLIA_COP:.0%} "
          f"({int((male['kind']=='linker').sum())} linker):")
    for isin, x in male.head(15).iterrows():
        print(f"    {isin}  {x['kind']:8s} scad {x['mat']:%Y-%m-%d}  "
              f"{int(x['n_z']):5d}/{int(x['n_px']):5d} = {x['cop']:.0%}")
    print("  -> se sono pochi e sparsi e' illiquidita': Bloomberg non quota lo Z-spread")
    print("     tutti i giorni. Se sono tanti e concentrati su anni interi e' troncamento:")
    print(f"     togli quelle righe da {path_f.name} e rilancia per richiederli.")
else:
    print(f"\n  nessun titolo sotto il {SOGLIA_COP:.0%}: copertura piena.")

# --- controllo incrociato col campo precedente -------------------------------------
# Costa zero chiamate: il pannello dell'altro campo e' gia' su disco. Se le due misure
# coincidono dove si sovrappongono, il cambio di campo non ha introdotto un salto.
if CONTROLLO and CONTROLLO != CAMPO:
    p_ctrl = CACHE / f"zsprd_{MERCATO}_{CONTROLLO}.parquet"
    if not p_ctrl.exists():
        print(f"\ncontrollo: {p_ctrl.name} non c'e', confronto saltato")
    else:
        alt = pd.read_parquet(p_ctrl)
        alt.index = pd.to_datetime(alt.index)
        com_c = [c for c in panel.columns if c in alt.columns]
        if not com_c:
            print(f"\ncontrollo vs {CONTROLLO}: nessuna colonna in comune")
        else:
            A = panel[com_c].reindex(panel.index.union(alt.index))
            B = alt[com_c].reindex(A.index)
            d = (A - B).where(A.notna() & B.notna())
            v = d.stack().dropna()          # senza dropna i percentili tornano NaN
            print(f"\ncontrollo {CAMPO} vs {CONTROLLO}: {len(v)} celle in comune "
                  f"su {len(com_c)} titoli")
            if len(v):
                q = np.percentile(np.abs(v.values), [50, 95, 99])
                print(f"  differenza (bp): media {v.mean():+.3f}, sd {v.std():.3f}, "
                      f"|mediana| {q[0]:.3f}, |p95| {q[1]:.3f}, |p99| {q[2]:.3f}")
                peggio = d.abs().max().dropna().sort_values(ascending=False).head(5)
                print("  titoli con lo scarto massimo piu' grande:")
                for c, x in peggio.items():
                    print(f"    {c}  max |diff| {x:.2f} bp   "
                          f"({uni.at[c, 'kind'] if c in uni.index else '-'})")
                # il metro e' la base, non lo spread: una base sovrana sta su pochi bp,
                # quindi uno scarto fra definizioni di 1 bp e' gia' un decimo del segnale.
                if q[1] < 1.0:
                    print("  -> i due campi misurano la stessa cosa: il cambio e' validato")
                    print("     sui dati, non sulla definizione.")
                elif q[1] < 5.0:
                    print("  -> scarto piccolo ma non nullo. Guarda se e' RUMORE (segno che")
                    print("     cambia, nessuna struttura) o SISTEMATICO per titolo o per")
                    print("     periodo: nel primo caso e' timing di snapshot e si puo'")
                    print("     procedere, nel secondo entra dritto nella base e no.")
                else:
                    print("  -> NON coincidono. Prima di usare la base, capire da dove viene")
                    print("     lo scarto: se e' sistematico per titolo o per periodo, i due")
                    print("     campi non scontano sulla stessa curva e la differenza fra le")
                    print("     gambe non sarebbe una base.")

print(f"\nfile: {path}")
