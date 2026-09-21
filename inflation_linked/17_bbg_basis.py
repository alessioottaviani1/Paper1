"""17 - LE DUE BASI DA Z-SPREAD BLOOMBERG. Offline: legge il pannello, non chiama Bloomberg.

UN CAMPO SOLO, DUE GAMBE. Entrambe le misure sono differenze fra Z-spread presi dallo
STESSO campo (INDEX_Z_SPREAD_BP) sullo stesso istante. La curva swap su cui Bloomberg
sconta e' quindi la stessa nelle due gambe e si cancella nella differenza: quel che resta
e' il prezzo relativo fra linker e nominale, che e' la base. E' per questo che non si
mescolano due campi diversi -- lo scarto fra due definizioni non si cancella, e finirebbe
dentro la base indistinguibile dal segnale.

  PAIR    lambda = z(linker) - z(nominale gemello)
          Il gemello e' il nominale con la scadenza piu' vicina FRA QUELLI QUOTATI QUEL
          GIORNO, entro MAX_MISMATCH giorni. E' l'oggetto di FLL / Kita-Tortorice senza
          gamba STRIPS: massimamente tradabile (due titoli), ma con un disallineamento di
          scadenza residuo che va riportato, non nascosto.

  INTERP  lambda = z(linker) - [(1-w) z(nominale sotto) + w z(nominale sopra)]
          con w = (scad_linker - scad_sotto) / (scad_sopra - scad_sotto), lineare in
          scadenza. Disallineamento ZERO per costruzione. Il peso w NON e' un artificio
          statistico: e' il peso del barbell, cioe' quanto della gamba nominale sta sul
          titolo lungo. Serve gia' adesso, perche' rende la misura replicabile come trade
          invece che solo come numero.

PERCHE' TUTTE E DUE. La PAIR e' tradabile ma sporca di mismatch; la INTERP e' pulita ma
poggia su due titoli, e se il bracket e' largo l'interpolazione e' un'estrapolazione
mascherata. Dove concordano, la base non e' un artefatto di nessuna delle due scelte.
Dove divergono, il colpevole e' nei controlli qui sotto.

COSA NON SI FA QUI. Non si filtra niente dentro i pannelli: escono tutte le osservazioni,
e accanto esce bbgmatch_<MKT>.parquet con mismatch, ampiezza del bracket, peso w e vita
residua per ogni (data, linker). I filtri si applicano a valle, dove si vedono, invece di
essere cotti dentro i dati. Il riepilogo mostra quanto costano.

TIE-BREAK. A parita' di distanza di scadenza si sceglie per ISIN, non per dimensione
dell'emissione. Il pool della pipeline ordina per ["d", "AMT_OUTSTANDING"], cioe' a parita'
preferisce il titolo piu' grande -- che e' tipicamente il benchmark on-the-run, cioe'
proprio quello che tende a essere caro per convenience yield. Sarebbe un pollice sulla
bilancia nella direzione del risultato atteso.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO      = "IT"
CAMPO        = "Z_SPRD_MID"
MAX_MISMATCH = 183      # giorni: la tolleranza di basis.MAX_MISMATCH_DAYS
MAX_BRACKET  = 1095     # giorni: oltre, l'interpolazione poggia su due titoli troppo lontani
MIN_VITA     = 365      # giorni: MIN_VITA_GG della pipeline, usato solo nel riepilogo
SALVA        = True

# --- pulizia del pannello, decisa sulla QUALITA' e non sul risultato ----------------
ZERI_A_NAN   = True     # 0.00 esatto = dato mancante travestito, non uno spread nullo
MIN_ANNI_LNK = 0.0      # nessun filtro: la soglia a 8 anni era tarata sul pannello sbagliato
CORRIDOIO    = (-50.0, 800.0)
# Il censimento (22) mostra che la gamba linker esce dal corridoio nel 49.7% dei casi sotto
# i 6 mesi e nello 0.7% sopra gli 8 anni, mentre quella nominale sta sotto il 3% ovunque.
# Non e' amplificazione 1/duration comune: sarebbe visibile su entrambe. E' un errore
# proprio del calcolo sul linker -- che deve proiettare l'inflazione e rivalutare il
# rimborso -- diviso per una duration che si accorcia.
#
# IL FILTRO E' SULLA VITA RESIDUA, NON SUL VALORE. Scartare le osservazioni che "sembrano
# sbagliate" sarebbe selezione sulla variabile dipendente: l'errore qui e' unilaterale
# (tutta la coda sta a sinistra), quindi togliere le negative alzerebbe la media delle
# superstiti e renderebbe impossibile distinguere una base positiva vera da una base
# fabbricata buttando via le negative. La vita residua non sa nulla di quanto vale lo
# spread quel giorno: e' esogena, e la soglia si sceglie dove la qualita' della gamba
# linker raggiunge quella della nominale -- non dove la correlazione con la nostra misura
# viene meglio.

# ----------------------------------------------------------------- dati
path = CACHE / f"zsprd_{MERCATO}_{CAMPO}.parquet"
if not path.exists():
    raise SystemExit(f"manca {path.name}: lancia prima 15.")
Z = pd.read_parquet(path)
Z.index = pd.to_datetime(Z.index)
Z = Z.sort_index()

if ZERI_A_NAN:
    n0 = int((Z == 0.0).sum().sum())
    if n0:
        Z = Z.mask(Z == 0.0)
        print(f"    {n0} zeri esatti trattati come mancanti")

ref_l = bbg.load("ref_linker");  ref_l = ref_l[ref_l["mkt"] == MERCATO]
ref_n = bbg.load("ref_nominal")
mat = pd.concat([pd.to_datetime(ref_l["maturity"], errors="coerce"),
                 pd.to_datetime(ref_n["maturity"], errors="coerce")])
mat = mat[~mat.index.duplicated()]

c_l = [c for c in Z.columns if c in ref_l.index and pd.notna(mat.get(c))]
c_n = [c for c in Z.columns if c not in ref_l.index and pd.notna(mat.get(c))]
if not c_l or not c_n:
    raise SystemExit(f"servono linker e nominali nel pannello: ho {len(c_l)} e {len(c_n)}.")

c_n = sorted(c_n, key=lambda c: (mat[c], c))          # ordinati per scadenza, poi ISIN
m_n = np.array([mat[c].value for c in c_n]) // 86_400_000_000_000   # giorni epoch
m_l = np.array([mat[c].value for c in c_l]) // 86_400_000_000_000
Zn, Zl = Z[c_n].values, Z[c_l].values
giorni = Z.index.values.astype("datetime64[D]").astype(int)

print(f"=== {MERCATO}: basi da {CAMPO} ===")
print(f"    {len(c_l)} linker, {len(c_n)} nominali, {len(Z)} date "
      f"({Z.index.min():%Y-%m-%d} -> {Z.index.max():%Y-%m-%d})")
print(f"    gemello entro {MAX_MISMATCH}gg; bracket fino a {MAX_BRACKET}gg")

# --- qualita' per vita residua, gamba per gamba: e' cio' su cui si sceglie la soglia ---
LOc, HIc = CORRIDOIO
# broadcasting invece del doppio ciclo: 4700 date x 144 colonne sono 680.000 lookup
_mc = np.array([mat[c].to_datetime64() for c in Z.columns]).astype("datetime64[D]").astype(int)
_md = Z.index.values.astype("datetime64[D]").astype(int)
ANNI = pd.DataFrame((_mc[None, :] - _md[:, None]) / 365.25,
                    index=Z.index, columns=Z.columns)
fuori = ((Z < LOc) | (Z > HIc)) & Z.notna()
tagli = [0, 1, 2, 4, 6, 8, 10, 15, 99]
print(f"\n--- qualita': quota fuori dal corridoio [{LOc:.0f}, {HIc:.0f}] bp ---")
print(f"    {'ttm':<10}{'linker':>22}{'nominale':>22}")
for lo, hi in zip(tagli[:-1], tagli[1:]):
    riga = f"    {f'{lo}-{hi}a' if hi < 99 else f'> {lo}a':<10}"
    for cols in (c_l, c_n):
        m_ = (ANNI[cols] >= lo) & (ANNI[cols] < hi) & Z[cols].notna()
        n_ = int(m_.sum().sum())
        q_ = float((fuori[cols] & m_).sum().sum() / n_) if n_ else np.nan
        riga += f"{q_:>13.1%} ({n_:>6})" if n_ else f"{'-':>22}"
    print(riga)
print(f"    -> soglia scelta per la gamba linker: {MIN_ANNI_LNK:.0f} anni")

# Il filtro si applica solo alla gamba LINKER. I nominali restano interi: sono gia' puliti
# a ogni scadenza, e servono comunque come bracket.
pre = int(Z[c_l].notna().sum().sum())
Z[c_l] = Z[c_l].mask(ANNI[c_l] < MIN_ANNI_LNK)
post = int(Z[c_l].notna().sum().sum())
print(f"    filtro vita residua: {pre} -> {post} osservazioni linker "
      f"({1 - post/max(pre,1):.0%} scartate)")
Zn, Zl = Z[c_n].values, Z[c_l].values
print()

# ----------------------------------------------------------------- calcolo
righe = []
n_pari = 0
for t in range(len(Z)):
    idx = np.flatnonzero(~np.isnan(Zn[t]))
    jl  = np.flatnonzero(~np.isnan(Zl[t]))
    if not len(idx) or not len(jl):
        continue
    mm, zz = m_n[idx], Zn[t, idx]
    pos = np.searchsorted(mm, m_l[jl])
    for k, j in enumerate(jl):
        a, b = pos[k] - 1, pos[k]
        ok_a, ok_b = a >= 0, b < len(mm)
        zlk, mlk = Zl[t, j], m_l[j]

        # --- gemello: il piu' vicino per distanza di scadenza, qualunque lato ---------
        pair = np.nan; mis = np.nan; gem = None
        cand = [(abs(mm[i] - mlk), i) for i in (a, b) if (i == a and ok_a) or (i == b and ok_b)]
        if cand:
            d0 = min(c[0] for c in cand)
            if sum(1 for c in cand if c[0] == d0) > 1:
                n_pari += 1
            i_g = min((c for c in cand if c[0] == d0), key=lambda c: c_n[idx[c[1]]])[1]
            mis, gem = int(abs(mm[i_g] - mlk)), c_n[idx[i_g]]
            if mis <= MAX_MISMATCH:
                pair = zlk - zz[i_g]

        # --- interpolata: i due che bracciano, lineare in scadenza -------------------
        interp = np.nan; amp = np.nan; w = np.nan
        if ok_a and ok_b:
            amp = int(mm[b] - mm[a])
            if amp == 0:
                interp, w = zlk - 0.5 * (zz[a] + zz[b]), 0.5
            elif amp <= MAX_BRACKET:
                w = (mlk - mm[a]) / amp
                interp = zlk - ((1 - w) * zz[a] + w * zz[b])

        if np.isfinite(pair) or np.isfinite(interp):
            righe.append((giorni[t], c_l[j], pair, interp, mis, amp, w,
                          int(mlk - giorni[t]), gem))

D = pd.DataFrame(righe, columns=["d", "isin", "pair", "interp", "mismatch",
                                 "bracket", "w", "ttm", "gemello"])
D["date"] = pd.to_datetime(D["d"], unit="D")
D = D.drop(columns="d")
if not len(D):
    raise SystemExit("nessuna osservazione: controlla il pannello.")

P = D.pivot_table(index="date", columns="isin", values="pair")
I = D.pivot_table(index="date", columns="isin", values="interp")

# ----------------------------------------------------------------- riepilogo
def _st(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 2:
        return f"{len(x):>8}" + " " * 45
    q = np.percentile(x, [5, 50, 95])
    return (f"{len(x):>8}{x.mean():>9.2f}{x.std():>8.2f}{q[0]:>9.2f}{q[1]:>8.2f}"
            f"{q[2]:>9.2f}{x.skew():>8.2f}")


HDR = f"{'n':>8}{'media':>9}{'sd':>8}{'p5':>9}{'mediana':>8}{'p95':>9}{'skew':>8}"
print(f"--- 1. le due misure, tutto il campione (bp) ---")
print(f"    {'':<12}{HDR}")
print(f"    {'pair':<12}{_st(D['pair'])}")
print(f"    {'interp':<12}{_st(D['interp'])}")

Df = D[D["ttm"] > MIN_VITA]
print(f"\n--- 2. dopo il filtro di vita residua > {MIN_VITA}gg ---")
print(f"    {'':<12}{HDR}")
print(f"    {'pair':<12}{_st(Df['pair'])}")
print(f"    {'interp':<12}{_st(Df['interp'])}")

print(f"\n--- 3. quanto costano i vincoli di matching ---")
tot = len(D)
print(f"    osservazioni con almeno una misura      {tot:>8}")
print(f"    con gemello entro {MAX_MISMATCH}gg                  {int(D['pair'].notna().sum()):>8}"
      f"  ({D['pair'].notna().mean():.0%})")
print(f"    con bracket entro {MAX_BRACKET}gg                 {int(D['interp'].notna().sum()):>8}"
      f"  ({D['interp'].notna().mean():.0%})")
print(f"    con entrambe                            {int((D['pair'].notna() & D['interp'].notna()).sum()):>8}")
if n_pari:
    print(f"    {n_pari} pareggi di distanza risolti per ISIN (non per dimensione)")
q = np.percentile(D["mismatch"].dropna(), [50, 90, 99])
print(f"    mismatch del gemello (gg): mediana {q[0]:.0f}, p90 {q[1]:.0f}, p99 {q[2]:.0f}")
q = np.percentile(D["bracket"].dropna(), [50, 90, 99])
print(f"    ampiezza del bracket (gg): mediana {q[0]:.0f}, p90 {q[1]:.0f}, p99 {q[2]:.0f}")
w = D["w"].dropna()
print(f"    peso del barbell w: mediana {w.median():.2f}, "
      f"fuori [0,1] {int(((w < 0) | (w > 1)).sum())} volte (estrapolazione)")

print(f"\n--- 4. concordano fra loro? (solo dove esistono entrambe, ttm > {MIN_VITA}gg) ---")
E = Df[Df["pair"].notna() & Df["interp"].notna()]
if len(E) > 10:
    dd = E["pair"] - E["interp"]
    print(f"    {len(E)} osservazioni, correlazione {E['pair'].corr(E['interp']):+.3f}")
    print(f"    pair - interp: media {dd.mean():+.2f} bp, sd {dd.std():.2f}, "
          f"|mediana| {dd.abs().median():.2f}")
    alto = E[E["mismatch"] > 90]
    if len(alto) > 10:
        print(f"    dove il mismatch supera 90gg ({len(alto)} oss.): scarto medio "
              f"{(alto['pair'] - alto['interp']).mean():+.2f} bp")
        print(f"    -> se cresce col mismatch, la PAIR e' distorta dal disallineamento e")
        print(f"       la INTERP e' la misura da riportare come primaria.")
else:
    print("    troppo poche osservazioni in comune per concludere.")

print(f"\n--- 5. per anno (interp, ttm > {MIN_VITA}gg) ---")
print(f"    {'':<12}{HDR}")
for y, g in Df.groupby(Df["date"].dt.year):
    print(f"    {y:<12}{_st(g['interp'])}")

# --- 6. contro la nostra misura sui flussi ----------------------------------------
p_own = CACHE / f"basis_zspread_{MERCATO}.parquet"
print(f"\n--- 6. contro la nostra base sui flussi (basis_zspread_{MERCATO}) ---")
if not p_own.exists():
    print(f"    {p_own.name} non c'e': lancia 04_basis_markets per il confronto.")
else:
    O = pd.read_parquet(p_own); O.index = pd.to_datetime(O.index)
    ol = O.stack().rename("nostra")
    ol.index.names = ["date", "isin"]
    M = Df.set_index(["date", "isin"])[["pair", "interp", "ttm"]].join(ol, how="inner")
    if len(M) < 10:
        print("    nessuna cella in comune.")
    else:
        print(f"    {len(M)} celle in comune")
        for c in ("pair", "interp"):
            x = M[[c, "nostra"]].dropna()
            if len(x) < 10:
                continue
            d = x[c] - x["nostra"]
            print(f"    {c:<8} corr {x[c].corr(x['nostra']):+.3f}   "
                  f"livello: Bloomberg {x[c].mean():+.2f} vs nostra {x['nostra'].mean():+.2f} bp"
                  f"   scarto medio {d.mean():+.2f} (sd {d.std():.2f})")
        print("    Le due misure scontano su curve diverse -- swap per Bloomberg, sovrana")
        print("    fittata per la nostra -- ma in ENTRAMBE la curva e' comune alle due gambe")
        print("    e si cancella nella differenza. Quindi i livelli sono confrontabili, e una")
        print("    correlazione alta con uno scarto di livello stabile e' il risultato atteso:")
        print("    due strade indipendenti verso lo stesso prezzo relativo.")

if SALVA:
    P.to_parquet(CACHE / f"bbgbasis_pair_{MERCATO}.parquet")
    I.to_parquet(CACHE / f"bbgbasis_interp_{MERCATO}.parquet")
    D.to_parquet(CACHE / f"bbgmatch_{MERCATO}.parquet")
    print(f"\nsalvati: bbgbasis_pair_{MERCATO}.parquet ({P.shape[0]}x{P.shape[1]}), "
          f"bbgbasis_interp_{MERCATO}.parquet, bbgmatch_{MERCATO}.parquet")
    print(f"    bbgmatch tiene mismatch, bracket, w, ttm e gemello per ogni osservazione:")
    print(f"    e' li' che si filtra a valle, non dentro i pannelli.")
