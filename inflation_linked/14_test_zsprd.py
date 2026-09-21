"""14 - SONDA: quale CAMPO e quale SORGENTE, verificati contro valori letti a schermo.

RICHIEDE IL TERMINALE. Costa poco: pochi titoli, una finestra corta, una richiesta per
combinazione.

L'ERRORE CHE QUESTA VERSIONE RIPARA. La versione precedente provava i campi su UNA
sorgente, e le sorgenti alternative solo nel ramo in cui nessun campo copriva i linker.
Siccome INDEX_Z_SPREAD_BP@CBBT copriva, Z_SPRD_MID@BGN non e' mai stato provato. Campo e
sorgente non sono indipendenti: vanno provati in GRIGLIA.

E il criterio di validita' era la COPERTURA. Sbagliato. INDEX_Z_SPREAD_BP@CBBT tornava al
100% e sembrava perfetto, ma sul BTPei 2.1 09/15/16 il 31/08/2015 dava -681.40 mentre a
schermo, con Z_SPRD_MID@BGN, c'era +6.727 -- a parita' di prezzo, 101.860 su entrambi. La
copertura dice che arrivano dei numeri, non che siano quelli giusti: e' una condizione
necessaria scambiata per sufficiente, ed e' costata quattro ipotesi costruite su dati che
non misuravano cio' che credevo.

IL CRITERIO ORA SONO LE ANCORE. Valori letti sullo schermo, con la loro data e il loro
prezzo, scritti qui sotto. Una combinazione campo x sorgente e' buona se riproduce quei
numeri, non se risponde. Il prezzo serve da controllo di allineamento: se il prezzo non
torna, la data o il titolo non corrispondono e il confronto sullo spread non vale nulla.

COME AGGIUNGERE UN'ANCORA. Su HP / Historical Price Table del titolo, campi Mid Z-Spread e
Mid Price, si legge una riga e la si copia in ANCORE. Piu' titoli ci sono, meglio e': una
sola ancora puo' coincidere per caso, cinque su due titoli no.
"""
import numpy as np
import pandas as pd
import bbg
from config import MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO = "IT"
SOURCES = ["BGN", "CBBT", ""]        # "" = ticker nudo
CANDIDATI = [
    "Z_SPRD_MID",          # SP037 - quello dello schermo HP
    "INDEX_Z_SPREAD_BP",   # BX326
    "YAS_ZSPREAD",
    "OAS_SPREAD_MID",
]

# ANCORE: (isin, data, z a schermo, prezzo a schermo, sorgente dello schermo)
# BTPS 2.1 09/15/16 = IT0004682107, lette da HP con source BGN.
ANCORE = [
    ("IT0004682107", "2015-08-31",   6.727, 101.860, "BGN"),
    ("IT0004682107", "2015-09-01",  14.421, 101.920, "BGN"),
    ("IT0004682107", "2015-09-15",  14.662, 102.113, "BGN"),
    ("IT0004682107", "2015-08-26",   5.506, 101.685, "BGN"),
    ("IT0004682107", "2015-08-18",  18.595, 101.852, "BGN"),
]
TOL_Z  = 0.5     # bp di tolleranza sullo Z-spread
TOL_PX = 0.02    # punti di tolleranza sul prezzo

# ----------------------------------------------------------------- esecuzione
try:
    from xbbg import blp  # noqa: F401
except ImportError:
    raise SystemExit("xbbg non disponibile: questo script va lanciato sul terminale.")

nom_mkt = bbg.NOMINAL_POOL_ALIAS.get(MERCATO, MERCATO)
ref_l = bbg.load("ref_linker");  ref_l = ref_l[ref_l["mkt"] == MERCATO]
ref_n = bbg.load("ref_nominal"); ref_n = ref_n[ref_n["mkt"] == nom_mkt]
bb = pd.concat([ref_l["bb_id"], ref_n["bb_id"]]).astype(str).str.strip()
bb = bb[~bb.index.duplicated()]

anc = pd.DataFrame(ANCORE, columns=["isin", "data", "z", "px", "src"])
anc["data"] = pd.to_datetime(anc["data"])
manca = [i for i in anc["isin"].unique() if i not in bb.index or bb[i] in ("nan", "None")]
if manca:
    raise SystemExit(f"ancore su titoli senza bb_id: {manca}")

d0, d1 = anc["data"].min() - pd.Timedelta(days=5), anc["data"].max() + pd.Timedelta(days=5)
isins = list(anc["isin"].unique())
print(f"=== {MERCATO}: griglia campo x sorgente contro {len(anc)} ancore di schermo ===")
for i in isins:
    print(f"    {i}  bb_id {bb[i]}  ({int((anc['isin']==i).sum())} ancore)")
print(f"    finestra {d0:%Y-%m-%d} -> {d1:%Y-%m-%d}\n")


def _tick(isin: str, src: str) -> str:
    return f"{bb[isin]}@{src} Corp" if src else f"{bb[isin]} Corp"


def _serie(campo: str, src: str, isins: list) -> pd.DataFrame | None:
    t2c = {_tick(i, src): i for i in isins}
    try:
        df = bbg._bdh(list(t2c), campo, d0.date(), d1.date())
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [t2c.get(c, c) for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


# --- 0. il prezzo: le date sono allineate? ----------------------------------------
# Se il prezzo non torna, la data o il titolo non corrispondono, e confrontare lo spread
# non vuol dire niente. Questo controllo viene PRIMA di tutto il resto.
print("--- 0. allineamento: il PREZZO torna? ---")
px_ok = {}
for src in SOURCES:
    df = _serie("PX_MID", src, isins)
    if df is None:
        print(f"    PX_MID @{src or 'nudo':<6} nessuna risposta")
        continue
    err = []
    for _, a in anc.iterrows():
        v = df[a["isin"]].get(a["data"], np.nan) if a["isin"] in df.columns else np.nan
        err.append(abs(v - a["px"]) if pd.notna(v) else np.nan)
    err = np.array(err, dtype=float)
    n_ok = int(np.nansum(err <= TOL_PX))
    px_ok[src] = n_ok
    print(f"    PX_MID @{src or 'nudo':<6} {n_ok}/{len(anc)} prezzi coincidono "
          f"(scarto mediano {np.nanmedian(err):.4f})")
print("    Una sorgente con i prezzi giusti conferma che stiamo guardando le stesse date.\n")

# --- 1. la griglia ----------------------------------------------------------------
print("--- 1. campo x sorgente: quale riproduce le ancore? ---")
print(f"    {'campo':<20}{'source':<8}{'n':>5}{'coperti':>9}{'|err| med':>11}"
      f"{'|err| max':>11}{'ok':>5}")
best = []
for campo in CANDIDATI:
    for src in SOURCES:
        df = _serie(campo, src, isins)
        if df is None:
            print(f"    {campo:<20}@{(src or 'nudo'):<7}{'-':>5}{'vuoto':>9}")
            continue
        vals, errs = [], []
        for _, a in anc.iterrows():
            v = df[a["isin"]].get(a["data"], np.nan) if a["isin"] in df.columns else np.nan
            vals.append(v)
            errs.append(abs(v - a["z"]) if pd.notna(v) else np.nan)
        errs = np.array(errs, dtype=float)
        n_cop = int(np.isfinite(errs).sum())
        n_ok = int(np.nansum(errs <= TOL_Z))
        med = np.nanmedian(errs) if n_cop else np.nan
        mx = np.nanmax(errs) if n_cop else np.nan
        flag = "  <<<" if n_ok == len(anc) else ""
        print(f"    {campo:<20}@{(src or 'nudo'):<7}{len(anc):>5}{n_cop:>9}"
              f"{med:>11.2f}{mx:>11.2f}{n_ok:>5}{flag}")
        if n_cop:
            best.append((n_ok, -med, campo, src, vals))

# --- 2. la combinazione migliore, valore per valore -------------------------------
print(f"\n--- 2. dettaglio della combinazione migliore ---")
if not best:
    raise SystemExit("    nessuna combinazione risponde: controlla i ticker su FLDS/HP.")
best.sort(reverse=True)
n_ok, _, campo, src, vals = best[0]
print(f"    {campo} @ {src or 'nudo'}:  {n_ok}/{len(anc)} ancore riprodotte")
print(f"      {'data':<12}{'schermo':>10}{'scaricato':>12}{'scarto':>10}")
for (_, a), v in zip(anc.iterrows(), vals):
    sc = f"{v - a['z']:+.3f}" if pd.notna(v) else "-"
    vv = f"{v:.3f}" if pd.notna(v) else "-"
    print(f"      {a['data']:%Y-%m-%d}{a['z']:>10.3f}{vv:>12}{sc:>10}")

# --- 3. quella combinazione copre anche i NOMINALI e la storia lunga? -------------
print(f"\n--- 3. {campo} @ {src or 'nudo'}: copertura su entrambe le gambe ---")
px = bbg.load(f"px_mid_{MERCATO}"); ytm = bbg.load(f"ytm_{nom_mkt}")
sel_l = px.notna().sum().sort_values(ascending=False).head(3).index.tolist()
sel_n = ytm.notna().sum().sort_values(ascending=False).head(2).index.tolist()
prova = [(i, "linker") for i in sel_l if i in bb.index] + \
        [(i, "nominale") for i in sel_n if i in bb.index]
for etichetta, (w0, w1) in {"recente": (pd.Timestamp.today().normalize() -
                                        pd.Timedelta(days=180),
                                        pd.Timestamp.today().normalize()),
                            "2012": ("2012-01-02", "2012-06-29"),
                            "2005": ("2005-01-03", "2005-06-30")}.items():
    g0, g1 = pd.Timestamp(w0), pd.Timestamp(w1)
    t2c = {_tick(i, src): i for i, _ in prova}
    try:
        df = bbg._bdh(list(t2c), campo, g0.date(), g1.date())
    except Exception as e:
        print(f"    {etichetta:<9} errore {str(e)[:40]}"); continue
    if df is None or df.empty:
        print(f"    {etichetta:<9} vuoto"); continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [t2c.get(c, c) for c in df.columns]
    nbd = max(len(pd.bdate_range(g0, g1)), 1)
    cells = []
    for i, k in prova:
        c = int(df[i].notna().sum()) / nbd if i in df.columns else 0.0
        m = df[i].median() if i in df.columns and df[i].notna().any() else np.nan
        cells.append(f"{k[:3]}:{c:.0%}/{'' if pd.isna(m) else f'{m:.0f}'}")
    print(f"    {etichetta:<9} " + "  ".join(cells))
print("    copertura / mediana in bp. Una mediana fuori da [0, 600] e' un campanello:")
print("    la combinazione risponde ma non sta misurando uno Z-spread sovrano.")

print(f"\n--- da mettere in 15_fetch_zsprd.py ---")
print(f"    CAMPO  = \"{campo}\"")
print(f"    SOURCE = \"{src}\"")
print(f"    e RISCARICARE da zero: il pannello attuale va buttato, non integrato.")
