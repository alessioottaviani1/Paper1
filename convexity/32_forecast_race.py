"""32 - LA GARA DI PREVISIONE: quale venue sa qualcosa della volatilita' futura?

PERCHE' UNO SCRIPT AUTONOMO. Questo e' il risultato positivo piu' forte dell'intero progetto, e
oggi vive come pannello [5] di 22_vrp_battery.py, quinto in un file di robustezza. Un fatto
NULLO (le due venue non co-muovono) non porta un paper su una rivista top nel 2026; un fatto
POSITIVO lo puo' fare, e questo lo e':

    in tre mercati su quattro uno dei due prezzi della volatilita' NON E' una previsione di
    volatilita' -- e' fissato da qualcos'altro -- mentre l'altro previde con pendenza 0.85.

E' esattamente cio' che la lettura per clientele asserisce, e da' contenuto quantitativo al
"troppa o troppo poca curvatura" di Rebonato-Ronzani (2021) e al limite di previsione che
Rebonato-Putyatin (2018, sez. 13) diagnosticano nel proprio lavoro.

TRE COSE CHE MANCANO ALLA VERSIONE ATTUALE.
  (1) Gira su 4 mercati. Le quattro curve GOVERNATIVE sono evidenza gratuita gia' in casa:
      otto mercati, e in particolare il contrasto swap-vs-govt DENTRO la valuta anche qui.
  (2) Nessun R2 fuori campione. In campione, un regressore con pendenza 0.85 e' quasi garantito.
      Serve il confronto out-of-sample contro il benchmark ingenuo (RV trailing), alla
      Campbell-Thompson (2008): e' cio' che distingue "correla" da "prevede".
  (3) Nessun test di encompassing. La domanda del referee non e' "la BE aggiunge R2?" ma
      "l'errore di previsione di sigma_IV e' ortogonale a sigma_BE?" (Chong-Hendry 1986). Se lo
      e', l'opzione racchiude la curva e la curva non contiene informazione residua.

PANNELLI
  [1] Mincer-Zarnowitz separate e congiunta: RV(t,t+3m) ~ sigma_BE | sigma_IV | entrambe
  [2] R2 out-of-sample (Campbell-Thompson) contro benchmark RV trailing, finestra espandente
  [3] Encompassing di Chong-Hendry: errore di sigma_IV regredito su sigma_BE, e viceversa
  [4] La stessa gara su sigma_BE_perp (se 30_spanning.py e' stato eseguito)

CONVENZIONI. RV su finestra futura di 63 giorni lavorativi, annualizzata, in bp/anno (vol
normale), costruita dalla gamba 10Y di ciascun mercato -- la stessa convenzione della 6.5.
Sovrapposizione di 3 mesi nelle osservazioni mensili: t di Newey-West con 6 ritardi.

Output: results/32_forecast_race.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt

print("== 32 forecast race ==")
L = []; P = L.append
P("=== 32 GARA DI PREVISIONE: sigma_BE contro sigma_IV sulla volatilita' realizzata futura ===")
P("")

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
W   = pd.read_csv(PROC/"vols_monthly.csv",  index_col=0, parse_dates=True)
mid = pd.read_csv(PROC/"mids_daily.csv",    index_col=0, parse_dates=True)

HORIZON = 63          # giorni lavorativi ~ 3 mesi: orizzonte della swaption 3M
MINOOS  = 60          # minimo di osservazioni per la prima stima fuori campione

def ols_nw(y, X, lag=6):
    y = np.asarray(y, float); X = np.asarray(X, float); n, k = X.shape
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.pinv(X.T @ X)
    S = (e[:, None]*X).T @ (e[:, None]*X)
    for l in range(1, lag+1):
        w = 1 - l/(lag+1)
        G = (e[l:, None]*X[l:]).T @ (e[:-l, None]*X[:-l])
        S += w*(G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 1e-30))
    r2 = 1 - (e@e)/(((y-y.mean())@(y-y.mean())) + 1e-30)
    return b, b/se, r2, n

# ---------------------------------------------------------------- RV per mercato
def leg10(mkt):
    """Gamba 10Y del mercato. USTgovt: GSW SVENY10 giornaliero dal file Fed."""
    if mkt in MK: return MK[mkt][0][1]
    return None

def rv_fwd(mkt):
    """RV FUTURA su HORIZON giorni, annualizzata, bp/anno, indicizzata a fine mese t
    (usa solo dati da t+1 a t+HORIZON: e' la variabile dipendente, non un regressore)."""
    if mkt == "USTgovt":
        try:
            h = next(i for i, l in enumerate(open(GSW).read().splitlines()) if l.startswith("Date,"))
            g = pd.read_csv(GSW, skiprows=h)
            g["Date"] = pd.to_datetime(g["Date"]); g = g.set_index("Date")
            r = pd.to_numeric(g["SVENY10"], errors="coerce").dropna()/100.0
        except Exception:
            return None
    else:
        lg = leg10(mkt)
        if lg is None or lg not in mid.columns: return None
        r = (mid[lg]/100.0).dropna()
    dy = r.diff()
    fut = dy.shift(-1).rolling(HORIZON).std().shift(-(HORIZON-1))*np.sqrt(252)*1e4
    tr  = dy.rolling(HORIZON).std()*np.sqrt(252)*1e4
    return fut.resample("ME").last(), tr.resample("ME").last()

def iv3m10(mkt):
    c = f"{IVMAP[mkt]}_3M_10Y_NORM"
    return W[c] if c in W.columns else None

ORDER = ["USDswap","USTgovt","EUR","DEgovt","GBP","UKgovt","JPY","JPgovt"]
MKTS  = [m for m in ORDER if m in sbe.columns]

DATA = {}
for mkt in MKTS:
    o = rv_fwd(mkt); iv = iv3m10(mkt)
    if o is None or iv is None: continue
    fut, tr = o
    al = pd.concat([fut.rename("RV"), sbe[mkt].rename("BE"), iv.rename("IV"),
                    tr.rename("TR")], axis=1).dropna()
    if len(al) >= MINOOS + 24: DATA[mkt] = al
P(f"[0] mercati con serie complete: {sorted(DATA)}")
P("")

# ---------------------------------------------------------------- [1] Mincer-Zarnowitz
P("[1] MINCER-ZARNOWITZ: RV(t,t+3m) su ciascun prezzo, separatamente e congiuntamente")
P(f"{'mercato':9}{'b_BE':>8}{'[t]':>6}{'R2':>6} | {'b_IV':>8}{'[t]':>6}{'R2':>6} | "
  f"{'jBE':>8}{'[t]':>6}{'jIV':>8}{'[t]':>6}{'R2':>6}{'N':>6}")
for mkt, al in DATA.items():
    y = al.RV.values; n = len(al)
    bB, tB, r2B, _ = ols_nw(y, np.column_stack([np.ones(n), al.BE.values]))
    bI, tI, r2I, _ = ols_nw(y, np.column_stack([np.ones(n), al.IV.values]))
    bJ, tJ, r2J, _ = ols_nw(y, np.column_stack([np.ones(n), al.BE.values, al.IV.values]))
    P(f"{mkt:9}{bB[1]:8.2f}{tB[1]:6.1f}{r2B:6.2f} | {bI[1]:8.2f}{tI[1]:6.1f}{r2I:6.2f} | "
      f"{bJ[1]:8.2f}{tJ[1]:6.1f}{bJ[2]:8.2f}{tJ[2]:6.1f}{r2J:6.2f}{n:6d}")
P("    lettura: pendenza ~ 0 su sigma_BE con pendenza ~ 0.85 su sigma_IV, e sigma_BE che nella")
P("    congiunta perde significativita', significa che UNO dei due prezzi della volatilita' non")
P("    e' una previsione di volatilita'. E' cio' che la lettura per clientele asserisce.")
P("")

# ---------------------------------------------------------------- [2] R2 fuori campione
P("[2] R2 FUORI CAMPIONE (Campbell-Thompson) contro benchmark RV trailing, finestra espandente")
P("    R2oos = 1 - MSE(modello)/MSE(benchmark). Positivo = batte l'ingenuo. In campione un")
P("    regressore con pendenza 0.85 e' quasi garantito: questo pannello distingue previsione")
P("    da correlazione.")
P(f"{'mercato':9}{'R2oos BE':>11}{'R2oos IV':>11}{'R2oos both':>12}{'N oos':>7}")
def r2oos(al, cols):
    y = al.RV.values; n = len(al)
    num = den = 0.0; cnt = 0
    for i in range(MINOOS, n):
        tr = al.iloc[:i]; X = np.column_stack([np.ones(i)] + [tr[c].values for c in cols])
        try: b = np.linalg.lstsq(X, tr.RV.values, rcond=None)[0]
        except Exception: continue
        x1 = np.concatenate([[1.0], [al[c].values[i] for c in cols]])
        f = float(x1 @ b); bm = float(al.TR.values[i])
        num += (y[i]-f)**2; den += (y[i]-bm)**2; cnt += 1
    return (1 - num/den if den > 0 else np.nan), cnt
for mkt, al in DATA.items():
    a, c1 = r2oos(al, ["BE"]); b, _ = r2oos(al, ["IV"]); j, _ = r2oos(al, ["BE","IV"])
    P(f"{mkt:9}{a:+11.2f}{b:+11.2f}{j:+12.2f}{c1:7d}")
P("")

# ---------------------------------------------------------------- [3] encompassing
P("[3] ENCOMPASSING (Chong-Hendry): l'errore di una venue e' ortogonale all'altro prezzo?")
P("    e_IV = RV - fit(IV) regredito su sigma_BE: se b ~ 0, l'OPZIONE RACCHIUDE la curva.")
P("    e_BE = RV - fit(BE) regredito su sigma_IV: se b >> 0, la curva NON racchiude l'opzione.")
P(f"{'mercato':9}{'b(e_IV~BE)':>12}{'[t]':>7}{'b(e_BE~IV)':>12}{'[t]':>7}{'verdetto':>26}")
for mkt, al in DATA.items():
    y = al.RV.values; n = len(al)
    bI = np.linalg.lstsq(np.column_stack([np.ones(n), al.IV.values]), y, rcond=None)[0]
    eI = y - np.column_stack([np.ones(n), al.IV.values]) @ bI
    bB = np.linalg.lstsq(np.column_stack([np.ones(n), al.BE.values]), y, rcond=None)[0]
    eB = y - np.column_stack([np.ones(n), al.BE.values]) @ bB
    c1, t1, _, _ = ols_nw(eI, np.column_stack([np.ones(n), al.BE.values]))
    c2, t2, _, _ = ols_nw(eB, np.column_stack([np.ones(n), al.IV.values]))
    if abs(t1[1]) < 2 and abs(t2[1]) >= 2: v = "IV racchiude BE"
    elif abs(t1[1]) >= 2 and abs(t2[1]) < 2: v = "BE racchiude IV"
    elif abs(t1[1]) < 2 and abs(t2[1]) < 2: v = "nessuna informazione"
    else: v = "informazione complementare"
    P(f"{mkt:9}{c1[1]:12.3f}{t1[1]:7.1f}{c2[1]:12.3f}{t2[1]:7.1f}{v:>26}")
P("    'IV racchiude BE' e' l'esito piu' forte per il paper: la curva non contiene informazione")
P("    residua sulla volatilita' futura, quindi il suo prezzo e' fissato da altro -- la clientela.")
P("")

# ---------------------------------------------------------------- [4] sul residuo di 30
P("[4] LA STESSA GARA SU sigma_BE_perp (richiede 30_spanning.py)")
try:
    perp = pd.read_csv(PROC/"sigbe_perp_monthly.csv", index_col=0, parse_dates=True)
    P(f"{'mercato':9}{'b_BEperp':>11}{'[t]':>6}{'R2':>6}{'b_BE grezzo':>13}{'[t]':>6}{'N':>6}")
    for mkt, al in DATA.items():
        if mkt not in perp.columns: continue
        j = pd.concat([al.RV, perp[mkt].rename("BEp"), al.BE], axis=1).dropna()
        if len(j) < 60: continue
        n = len(j)
        bp, tp, r2p, _ = ols_nw(j.RV.values, np.column_stack([np.ones(n), j.BEp.values]))
        bg, tg, _,  _  = ols_nw(j.RV.values, np.column_stack([np.ones(n), j.BE.values]))
        P(f"{mkt:9}{bp[1]:11.2f}{tp[1]:6.1f}{r2p:6.2f}{bg[1]:13.2f}{tg[1]:6.1f}{n:6d}")
    P("    se il residuo ortogonale alla forma della curva continua a non prevedere nulla, il")
    P("    risultato [1] non e' un artefatto della componente spannata.")
except FileNotFoundError:
    P("    sigbe_perp_monthly.csv assente: eseguire prima 30_spanning.py.")
P("")

P("=== NOTA PER LA STESURA ===")
P("  Se [2] da' R2oos positivo per sigma_IV e negativo o nullo per sigma_BE, e [3] da'")
P("  'IV racchiude BE' in tre mercati su quattro, questo pannello E' il paper. Va in Sezione 2")
P("  come fatto guida, non in robustezza: e' un risultato POSITIVO, quantitativo e sorprendente,")
P("  e il quasi-zero del C3 diventa la sua corroborazione anziche' il contrario.")
P("  Il Giappone che inverte (curva che previde, opzione che non previde) non e' un'anomalia da")
P("  spiegare via: e' la firma della clientela YCC, e va presentato come tale.")
save_txt("32_forecast_race.txt", L); print("\n".join(L))
