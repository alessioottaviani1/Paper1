"""34 - IL PREMIO: il doppio sort mancante, e l'inferenza che t=1.7 non regge.

DUE PROBLEMI DISTINTI, ENTRAMBI SUL PUNTO PIU' DEBOLE DEL PAPER.

PROBLEMA 1: UN DOPPIO SORT MANCANTE. La 6.15 neutralizza il term premium con un doppio sort
(dentro ogni terzile del fattore forward locale, l'alto stress paga ancora piu' del basso:
+19.6 [1.5], +20.3 [2.2], +33.8 [2.6], +12.3 [1.7]). Quel disegno e' corretto e va replicato
sull'oggetto che la 30 ha reso saliente: la CURVATURA. La 30 ha mostrato che sigma_BE e'
spannato dalla forma della curva per il 49-69% -- il che rende immediata l'obiezione che il
ladder per terzili di stress stia leggendo dinamica di curva correlata con lo stress, non
vincolo di bilancio. La 6.13 mostra che i RENDIMENTI non caricano sul fattore curvatura
tradabile (R2 0.00-0.02), ma quello e' un test sull'esposizione media, non sul CONDIZIONAMENTO:
un R2 nullo in media e' compatibile con un ladder che ordina stati di curva. Il doppio sort
stress x curvatura e' il test che separa le due cose, e non esiste nel pacchetto.

PROBLEMA 2: L'INFERENZA NON E' ALL'ALTEZZA DELLA CLAIM. I rendimenti netti stanno a t di
1.7 / 2.7 / 1.7 / 1.5, e il pooled a 2.3 e' la media semplice di quattro serie trattate come
se fossero indipendenti. Non lo sono: sono la stessa strategia su quattro curve che si muovono
insieme, e la correlazione fra mercati gonfia il t aggregato. Tre correzioni, tutte standard:
    (a) pannello con effetti fissi di mercato ed errori standard CLUSTERIZZATI PER DATA, che
        e' la dimensione lungo cui la dipendenza corre;
    (b) Driscoll-Kraay, che ammette anche autocorrelazione oltre alla dipendenza cross-section;
    (c) permutazione a blocchi sullo spread ALTO-meno-BASSO, che non assume nulla sulla
        distribuzione ed e' l'inferenza giusta per un oggetto event-concentrated.
Se il premio sopravvive a (a)-(c) e' difendibile a t=2. Se il t crolla sotto inferenza
corretta, e' meglio saperlo adesso: e' esattamente il calcolo che un referee rifa.

TERZO PANNELLO, GRATUITO. Le quattro curve GOVERNATIVE hanno gia' sigma_BE e i rendimenti
lordi del pacchetto, e non entrano mai nell'analisi dei rendimenti (04 le salta per assenza
di spread dealer). Raddoppiano la cross-section. Vanno riportate come lorde e dichiarate tali:
non sostituiscono i quattro mercati swap, ma un premio che esiste in otto mercati su otto e'
un fatto diverso da un premio che esiste in quattro.

QUARTO PANNELLO. La 32 ha lasciato aperta la domanda naturale del referee: sigma_IV aggiunge
qualcosa alla VOLATILITA' REALIZZATA TRAILING, che e' il benchmark davvero duro? E sigma_BE?

PANNELLI
  [1] Doppio sort: terzili di stress x terzili di curvatura
  [2] Pannello pooled: FE di mercato, SE clusterizzati per data, Driscoll-Kraay
  [3] Permutazione a blocchi sullo spread ALTO-meno-BASSO, pooled
  [4] Le otto curve: il premio esiste anche sulle governative? (lordo, dichiarato)
  [5] Mincer-Zarnowitz con la RV trailing come regressore

Output: results/34_premium_robustness.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, nw_t, load_dealer_cds

print("== 34 premium robustness ==")
rng = np.random.default_rng(SEED)
L = []; P = L.append
P("=== 34 IL PREMIO: doppio sort su curvatura, e inferenza corretta ===")
P("")

sbe = pd.read_csv(PROC/"sigbe_monthly.csv",  index_col=0, parse_dates=True)
mid = pd.read_csv(PROC/"mids_daily.csv",     index_col=0, parse_dates=True)
W   = pd.read_csv(PROC/"vols_monthly.csv",   index_col=0, parse_dates=True)
try:
    S = pd.read_csv(PROC/"strat_monthly.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    P("strat_monthly.csv assente: eseguire 04_costs_net.py."); save_txt("34_premium_robustness.txt", L); raise SystemExit
RR  = pd.read_csv(PROC/"pack_returns_monthly.csv", index_col=0, parse_dates=True)
s2  = pd.read_csv(PROC/"s2be_monthly.csv",   index_col=0, parse_dates=True)
CDS, _, _ = load_dealer_cds()
SWAP = [m for m in ("USDswap","EUR","GBP","JPY") if m in S.columns]
P(f"[0] mercati swap con rendimenti netti: {SWAP}")

# stato della curva: stessa formula della 30
CRV = {}
for mkt, (legs, taus) in MK.items():
    z = mid[list(legs)].dropna().resample("ME").last()
    CRV[mkt] = 2*z.iloc[:,1] - z.iloc[:,0] - z.iloc[:,2]

def terciles(s):
    q1, q2 = s.quantile(1/3), s.quantile(2/3)
    return pd.Series(np.where(s <= q1, "L", np.where(s <= q2, "M", "H")), index=s.index)

# ------------------------------------------------------------------ [1] doppio sort
P("")
P("[1] DOPPIO SORT: stress x curvatura. Dentro ogni terzile di CURVATURA, l'alto stress")
P("    paga ancora piu' del basso? Se si', il ladder non sta leggendo dinamica di curva.")
P(f"{'mercato':9}{'terzile CRV':>13}{'LOW':>9}{'HIGH':>9}{'H-L':>9}{'[t H-L]':>10}{'n H':>6}")
DS = {}
for mkt in SWAP:
    r = S[mkt].dropna()
    st = CDS[CDSREGION[mkt]].reindex(r.index).ffill()
    cv = CRV[mkt].reindex(r.index).ffill()
    j = pd.concat([r.rename("r"), st.rename("s"), cv.rename("c")], axis=1).dropna()
    if len(j) < 60: continue
    j["ct"] = terciles(j.c)
    rows = []
    for lab in ("L","M","H"):
        g = j[j.ct == lab]
        if len(g) < 20: continue
        stg = terciles(g.s)                    # terzili di stress DENTRO il terzile di curva
        lo, hi = g.r[stg == "L"], g.r[stg == "H"]
        if len(lo) < 6 or len(hi) < 6: continue
        d = hi.mean() - lo.mean()
        # t sulla differenza fra medie, NW su ciascun braccio
        se = np.sqrt(hi.var()/max(len(hi),1) + lo.var()/max(len(lo),1))
        P(f"{mkt if lab=='L' else '':9}{lab:>13}{lo.mean():9.2f}{hi.mean():9.2f}"
          f"{d:9.2f}{d/max(se,1e-9):10.1f}{len(hi):6d}")
        rows.append(d)
    if rows: DS[mkt] = rows
P("    lettura: se H-L resta positivo nei tre terzili di curvatura in quasi tutti i mercati,")
P("    il premio e' condizionale allo stress e non alla forma della curva. Se il segno si")
P("    inverte dentro i terzili, il ladder stava ordinando stati di curva ed e' la scoperta.")
if DS:
    allv = np.array([x for v in DS.values() for x in v])
    P(f"    riepilogo: {int((allv>0).sum())}/{len(allv)} celle con H-L positivo, "
      f"media {allv.mean():+.2f} bp")
P("")

# ------------------------------------------------------------------ [2] pannello pooled
P("[2] PANNELLO POOLED: il t=2.3 aggregato regge a inferenza che ammette dipendenza?")
P("    Il pooled attuale e' la media di quattro serie correlate trattate come indipendenti.")
pan = S[SWAP].dropna(how="all").stack().rename("r").reset_index()
pan.columns = ["date","mkt","r"]
pan = pan.dropna()
y = pan.r.values
D = pd.get_dummies(pan.mkt, drop_first=False).values.astype(float)   # FE di mercato
b = np.linalg.lstsq(D, y, rcond=None)[0]
e = y - D @ b
alpha = float(np.mean(b))                     # media dei FE = premio medio di mercato
# --- SE clusterizzati per DATA (la dimensione della dipendenza)
def cluster_se(X, e, groups):
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        u = (X[m] * e[m][:,None]).sum(axis=0)[:,None]
        meat += u @ u.T
    V = XtXi @ meat @ XtXi
    return np.sqrt(np.maximum(np.diag(V), 0))
ones = np.ones((len(y),1))
b0 = np.linalg.lstsq(ones, y, rcond=None)[0]; e0 = y - ones @ b0
se_naive = np.std(e0, ddof=1)/np.sqrt(len(y))
se_date  = cluster_se(ones, e0, pan.date.values)[0]
se_mkt   = cluster_se(ones, e0, pan.mkt.values)[0]
# --- Driscoll-Kraay: media cross-section per data, poi NW sulla serie temporale
xt = pan.groupby("date").r.mean().sort_index()
P(f"    premio pooled: {float(b0[0]):.2f} bp/mese   N = {len(y):,} osservazioni "
  f"({pan.mkt.nunique()} mercati x {pan.date.nunique()} mesi)")
P(f"      t ingenuo (indipendenza)         {float(b0[0])/se_naive:5.1f}")
P(f"      t cluster per DATA               {float(b0[0])/se_date:5.1f}   <-- la dipendenza vera")
P(f"      t cluster per MERCATO            {float(b0[0])/se_mkt:5.1f}")
P(f"      t Driscoll-Kraay (media x-sec, NW6) {nw_t(xt):5.1f}")
P("    il numero da riportare nel paper e' quello clusterizzato per data o Driscoll-Kraay.")
P("    Se il t ingenuo e quello per data divergono molto, il pooled attuale sovrastima.")
P("")

# ------------------------------------------------------------------ [3] permutazione
P("[3] PERMUTAZIONE A BLOCCHI sullo spread ALTO-meno-BASSO, pooled")
P("    Nessuna assunzione distributiva: si ricircola lo STATO DI STRESS rispetto ai")
P("    rendimenti mantenendone l'autocorrelazione (shift circolare), e si guarda dove cade")
P("    lo spread osservato nella distribuzione nulla.")
def hml(r, st):
    t = terciles(st.reindex(r.index).ffill().dropna())
    r = r.reindex(t.index)
    return r[t=="H"].mean() - r[t=="L"].mean()
obs_pool, null_pool = [], []
for mkt in SWAP:
    r = S[mkt].dropna(); st = CDS[CDSREGION[mkt]].reindex(r.index).ffill().dropna()
    r = r.reindex(st.index).dropna(); st = st.reindex(r.index)
    if len(r) < 60: continue
    obs_pool.append(hml(r, st))
    nl = [hml(r, pd.Series(np.roll(st.values, k), index=st.index))
          for k in range(6, len(r)-6)]
    null_pool.append(np.array(nl))
if obs_pool:
    obs = float(np.mean(obs_pool))
    nmin = min(len(x) for x in null_pool)
    null = np.mean([x[:nmin] for x in null_pool], axis=0)
    p = float(np.mean(null >= obs))
    P(f"    spread H-L pooled osservato: {obs:+.2f} bp   |   p (shift circolare, unilaterale) = {p:.3f}")
    P(f"    distribuzione nulla: media {null.mean():+.2f}, p95 {np.quantile(null,0.95):+.2f}, "
      f"n shift {nmin}")
P("")

# ------------------------------------------------------------------ [4] otto mercati
P("[4] LE OTTO CURVE: il premio esiste anche sulle governative? (rendimenti LORDI)")
P("    Le governative non hanno spread dealer nel pacchetto, quindi 04 le salta. Qui entrano")
P("    LORDE e vanno dichiarate tali: non sostituiscono i quattro swap, ma raddoppiano la")
P("    cross-section su cui il fatto puo' essere affermato.")
P(f"{'mercato':9}{'lordo':>9}{'[t]':>7}{'LOW':>9}{'HIGH':>9}{'H-L':>9}{'n':>6}")
GR = {}
for mkt in [m for m in sbe.columns if m in RR.columns]:
    idx = s2[mkt].dropna().index if mkt in s2.columns else RR[mkt].dropna().index
    lg = MK[mkt][0][1] if mkt in MK else None
    if lg is None or lg not in mid.columns:
        continue                                  # USTgovt: gambe GSW, non nel mid daily
    dy = (mid[lg]/100.0).diff()
    trail = (dy.rolling(63).std()*np.sqrt(252)).resample("ME").last().reindex(idx)
    pos = np.sign(trail**2 - s2[mkt].reindex(idx)).fillna(0.0)
    g = (pos.shift(1)*RR[mkt].reindex(idx)).dropna()
    if len(g) < 60: continue
    GR[mkt] = g
    # load_dealer_cds() ritorna un dict: l'appartenenza si testa con 'in', non con .columns.
    # E le governative non hanno una chiave propria in CDSREGION: si mappano alla valuta.
    GOVTMAP = {"USTgovt": "USDswap", "DEgovt": "EUR", "UKgovt": "GBP", "JPgovt": "JPY"}
    reg = CDSREGION.get(GOVTMAP.get(mkt, mkt))
    st = CDS[reg].reindex(g.index).ffill() if (reg is not None and reg in CDS) else None
    if st is None or st.isna().all():
        P(f"{mkt:9}{g.mean():9.2f}{nw_t(g):7.1f}{'--':>9}{'--':>9}{'--':>9}{len(g):6d}"); continue
    t = terciles(st.dropna()); gg = g.reindex(t.index).dropna(); t = t.reindex(gg.index)
    lo, hi = gg[t=="L"], gg[t=="H"]
    P(f"{mkt:9}{g.mean():9.2f}{nw_t(g):7.1f}{lo.mean():9.2f}{hi.mean():9.2f}"
      f"{hi.mean()-lo.mean():9.2f}{len(gg):6d}")
if len(GR) >= 6:
    POOL8 = pd.DataFrame(GR).dropna(how="all").mean(axis=1)
    P(f"{'POOL':9}{POOL8.mean():9.2f}{nw_t(POOL8):7.1f}")
P("")

# ------------------------------------------------------------------ [5] MZ con RV trailing
P("[5] MINCER-ZARNOWITZ CON LA RV TRAILING COME REGRESSORE")
P("    Il benchmark duro non e' 'niente', e' la volatilita' realizzata recente. sigma_IV")
P("    aggiunge a essa? sigma_BE? E' la domanda che un referee fa dopo aver letto la 32.")
def ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X@b
    n, k = X.shape; XtXi = np.linalg.pinv(X.T@X)
    Sm = (e[:,None]*X).T @ (e[:,None]*X)
    for l in range(1, 7):
        w = 1-l/7; G = (e[l:,None]*X[l:]).T @ (e[:-l,None]*X[:-l]); Sm += w*(G+G.T)
    V = XtXi@Sm@XtXi
    return b, b/np.sqrt(np.maximum(np.diag(V),1e-30)), 1-(e@e)/(((y-y.mean())@(y-y.mean()))+1e-30)
P(f"{'mercato':9}{'b_TRAIL':>10}{'[t]':>6}{'b_IV':>9}{'[t]':>6}{'b_BE':>9}{'[t]':>6}{'R2':>7}{'n':>6}")
for mkt in [m for m in sbe.columns if m in MK]:
    lg = MK[mkt][0][1]
    if lg not in mid.columns: continue
    dy = (mid[lg]/100.0).dropna().diff()
    fut = (dy.shift(-1).rolling(63).std().shift(-62)*np.sqrt(252)*1e4).resample("ME").last()
    tr  = (dy.rolling(63).std()*np.sqrt(252)*1e4).resample("ME").last()
    c = f"{IVMAP[mkt]}_3M_10Y_NORM"
    if c not in W.columns: continue
    j = pd.concat([fut.rename("RV"), tr.rename("TR"), W[c].rename("IV"),
                   sbe[mkt].rename("BE")], axis=1).dropna()
    if len(j) < 60: continue
    b, t, r2 = ols(j.RV.values, np.column_stack([np.ones(len(j)), j.TR, j.IV, j.BE]))
    P(f"{mkt:9}{b[1]:10.2f}{t[1]:6.1f}{b[2]:9.2f}{t[2]:6.1f}{b[3]:9.2f}{t[3]:6.1f}"
      f"{r2:7.2f}{len(j):6d}")
P("    se b_IV resta significativa accanto alla RV trailing mentre b_BE no, il risultato")
P("    della 32 e' robusto al benchmark duro ed e' pronto per la Sezione 2.")
save_txt("34_premium_robustness.txt", L); print("\n".join(L))
