"""rp - libreria per l'analisi dei premi al rischio sui linker (richieste RR, ago 2026).

Layer di RICERCA sopra la pipeline inflation_linked: ne importa config/curves e ne legge
la cache (curve ufficiali BoE/GSW gia' validate). Tre blocchi:
  1. BEI e residuo di premio (BEI - inflazione attesa / spot), US vs UK
  2. sorprese d'inflazione -> variazioni dei rendimenti REALI (segmentazione)
  3. predicibilita' degli excess return reali: slope vs CP vs Cieslak-Povala vs
     predittore di COINTEGRAZIONE diretto (Rebonato-Nyholm JEF 2025: il potere di CP
     sta nell'identificare la relazione di cointegrazione tra i forward)

Convenzioni: curve in composto continuo, %, griglie in anni. Tutto mensile EOM.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))

import numpy as np
import pandas as pd
from config import CACHE                      # cache unica del progetto
import curves                                  # gsw_zero_panel, nss_yield

NU_CIP = 0.987          # smoothing del trend d'inflazione (Cieslak-Povala 2015, mensile)
NW_LAGS_12M = 18        # Newey-West per holding period 12m su dati mensili sovrapposti


# ------------------------------------------------------------------ caricamento curve
def boe(curve: str) -> pd.DataFrame:
    """Pannello BoE date x maturita' (cc, %). curve: 'nominal' | 'real' | 'inflation'."""
    df = pd.read_parquet(CACHE / f"boe_{curve}_spot_std.parquet")
    df.index = pd.to_datetime(df.index)
    df.columns = [float(c) for c in df.columns]
    return df.sort_index()


def gsw(kind: str, grid) -> pd.DataFrame:
    """Curva GSW valutata dai parametri pubblicati (cc, %). kind: 'nominal' | 'tips'."""
    raw = pd.read_parquet(CACHE / f"gsw_{kind}_raw.parquet")
    out = curves.gsw_zero_panel(kind, grid, raw)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def cpi(mkt: str) -> pd.Series:
    """Indice prezzi mensile (an_cpi_{mkt}.parquet, da r0_data): UK=RPI, US=CPI-U NSA."""
    df = pd.read_parquet(CACHE / f"an_cpi_{mkt}.parquet")
    s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    return s[~s.index.duplicated(keep="last")].sort_index()


def expinf_us() -> pd.DataFrame:
    """Aspettative d'inflazione USA dal modello Cleveland Fed (Haubrich-Pennacchi-
    Ritchken 2012 RFS), orizzonti 1..30 anni, mensile (an_expinf_US.parquet)."""
    df = pd.read_parquet(CACHE / "an_expinf_US.parquet")
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    df.columns = [float(c) for c in df.columns]
    return df.sort_index()


# ------------------------------------------------------------------ trasformazioni
def liquidity() -> pd.DataFrame:
    """Proxy di liquidita' alla Maffei: VIX (equity vol) e MOVE (Treasury vol), da
    an_liq.parquet (r0_data). La serie e' giornaliera: ricampionata a fine mese (EOM) e
    standardizzata (media 0, sd 1) per leggere i beta in unita' confrontabili."""
    df = pd.read_parquet(CACHE / "an_liq.parquet")
    df.index = pd.to_datetime(df.index)
    df = df.resample("ME").last()                     # giornaliero -> mensile EOM
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(how="all")
    return (df - df.mean()) / df.std()


def eom(df):
    """Campionamento a fine mese (ultima osservazione disponibile del mese)."""
    return df.resample("ME").last().dropna(how="all")


def yoy(idx: pd.Series) -> pd.Series:
    """Inflazione YoY (%) da un indice mensile."""
    return (idx / idx.shift(12) - 1.0) * 100.0


def ewma_trend(pi: pd.Series, nu: float = NU_CIP) -> pd.Series:
    """Trend d'inflazione alla Cieslak-Povala: media mobile esponenziale
    tau_t = (1-nu) * sum_i nu^i * pi_{t-i}."""
    return pi.dropna().ewm(alpha=1.0 - nu, adjust=True).mean()


def interp_cols(panel: pd.DataFrame, mats, warn: bool = True) -> pd.DataFrame:
    """Interpola linearmente il pannello sulle maturita' richieste (colonne float).
    FUORI dal range della curva np.interp clampa al bordo (estrapolazione piatta):
    viene segnalato, perche' e' una scelta da dichiarare, non un default silenzioso."""
    cols = np.array(sorted(panel.columns), float)
    outside = [float(m) for m in mats if m < cols.min() - 1e-9 or m > cols.max() + 1e-9]
    if outside and warn:
        print(f"    interp_cols: {outside} fuori dalla curva ({cols.min():.1f}-{cols.max():.1f}y) "
              f"-> valore clampato al bordo (estrapolazione piatta)")
    out = {}
    vals = panel[sorted(panel.columns)].values
    for m in mats:
        out[float(m)] = np.array([np.interp(m, cols, row) for row in vals])
    return pd.DataFrame(out, index=panel.index)


# ------------------------------------------------------------------ BEI
def bei_uk(mats=(5.0, 10.0)) -> pd.DataFrame:
    """BEI UK = curva d'inflazione implicita BoE (spot), gia' nominale-reale coerente."""
    return interp_cols(boe("inflation"), mats)


def bei_us(mats=(5.0, 10.0)) -> pd.DataFrame:
    """BEI US = GSW nominale - GSW TIPS reale, stessa griglia (cc, %)."""
    return gsw("nominal", mats) .sub(gsw("tips", mats), axis=0).dropna(how="all")


# ------------------------------------------------------------------ excess return (curva zero, cc)
def rx_panel(y: pd.DataFrame, mats, hold_m: int = 12,
             fund_mat: float | None = None) -> tuple[pd.DataFrame, float]:
    """Excess return (%) su zero-coupon sintetici dalla curva, campione mensile EOM,
    holding period hold_m mesi:  rx_t->t+h(n) = n*y_t(n) - (n-h)*y_{t+h}(n-h) - h*y_t(f)
    con h = hold_m/12 e f = fund_mat (default: la maturita' piu' corta disponibile --
    'excess sul reale corto'; per i reali il funding a 1 anno non esiste sulla curva).
    Overlapping mensile: inferenza con Newey-West (NW_LAGS_12M)."""
    ye = eom(y)
    h = hold_m / 12.0
    cols = np.array(sorted(ye.columns), float)
    f = float(fund_mat) if fund_mat else float(cols.min())
    need = sorted({float(m) for m in mats} | {float(m) - h for m in mats} | {f})
    yi = interp_cols(ye, need)
    out = {}
    for n in mats:
        n = float(n)
        rx = n * yi[n] - (n - h) * yi[n - h].shift(-hold_m) - h * yi[f]
        out[n] = rx
    return pd.DataFrame(out).dropna(how="all"), f


def forwards(y: pd.DataFrame, mats=(2, 3, 4, 5, 7)) -> pd.DataFrame:
    """Forward a un anno: f(n) = n*y(n) - (n-1)*y(n-1) (cc, %), SOLO dove la curva
    copre sia n che n-1. Niente estrapolazione sotto il minimo della curva: le curve
    REALI partono da ~2y (BoE 2.5y, GSW TIPS 2y), quindi f(1) non esiste e clampare al
    minimo duplicherebbe le colonne (matrice singolare in Johansen, CP mal condizionato).
    Le maturita' non coperte vengono scartate con avviso; default (2..7) adatto ai reali."""
    ye = eom(y)
    lo, hi = float(min(ye.columns)), float(max(ye.columns))
    ok = [float(n) for n in mats if (n - 1) >= lo - 1e-9 and n <= hi + 1e-9]
    drop = [n for n in mats if float(n) not in ok]
    if drop:
        print(f"    forwards: scarto {drop} (curva copre {lo:.1f}-{hi:.1f}y; serve n-1 >= {lo:.1f})")
    need = sorted({n for n in ok} | {n - 1 for n in ok})
    yi = interp_cols(ye, need)
    out = {n: n * yi[n] - (n - 1) * yi[n - 1] for n in ok}
    return pd.DataFrame(out)


# ------------------------------------------------------------------ econometria (numpy)
def ols(yv: np.ndarray, X: np.ndarray, const: bool = True):
    X = np.column_stack([np.ones(len(X)), X]) if const else np.asarray(X)
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    fit = X @ b
    e = yv - fit
    r2 = 1 - e.var() / yv.var() if yv.var() > 0 else np.nan
    return b, e, fit, r2, X


def nw_t(e: np.ndarray, X: np.ndarray, b: np.ndarray, lags: int):
    """t-stat Newey-West (kernel di Bartlett)."""
    n, k = X.shape
    u = X * e[:, None]
    S = u.T @ u / n
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = u[l:].T @ u[:-l] / n
        S += w * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    V = Q @ S @ Q / n
    se = np.sqrt(np.diag(V))
    return b / se


def adf_t(e: pd.Series, lags: int = 1) -> float:
    """t-ADF sul residuo (H0: radice unitaria). Critici ~ -3.43 (1%), -2.86 (5%).
    Usa statsmodels se disponibile, altrimenti DF aumentato a mano."""
    e = pd.Series(e).dropna()
    try:
        from statsmodels.tsa.stattools import adfuller
        return float(adfuller(e, maxlag=lags, autolag=None)[0])
    except Exception as exc:
        print(f"  !! adfuller non disponibile ({type(exc).__name__}): t-ADF calcolato con "
              f"DF manuale (approssimato). pip install statsmodels per il valore standard.")
        de = e.diff().dropna()
        X = [e.shift(1).loc[de.index].values]
        for l in range(1, lags + 1):
            X.append(de.shift(l).reindex(de.index).fillna(0).values)
        X = np.column_stack(X)
        b, res, _, _, Xc = ols(de.values, X)
        t = nw_t(res, Xc, b, 0)
        return float(t[1])


# ------------------------------------------------------------------ fattori predittivi
def cp_factor(rx_bar: pd.Series, F: pd.DataFrame):
    """Cochrane-Piazzesi (2005): rx medio sui forward -> il fitted e' il fattore CP."""
    df = pd.concat([rx_bar.rename("rx"), F], axis=1).dropna()
    b, e, fit, r2, X = ols(df["rx"].values, df[F.columns].values)
    return pd.Series(fit, index=df.index, name="CP"), b, r2


def cip_factor(y: pd.DataFrame, trend: pd.Series, rx_bar: pd.Series, mats):
    """Cieslak-Povala (2015): cicli = residui di y(n) su [1, trend]; il fattore e' il
    fitted di rx medio su [ciclo corto, media cicli lunghi]."""
    ye = interp_cols(eom(y), mats)
    tr = trend.reindex(ye.index).ffill()
    cyc = {}
    for n in mats:
        d = pd.concat([ye[float(n)].rename("y"), tr.rename("tau")], axis=1).dropna()
        b, e, *_ = ols(d["y"].values, d[["tau"]].values)
        cyc[float(n)] = pd.Series(e, index=d.index)
    C = pd.DataFrame(cyc)
    short = C[float(mats[0])]
    longm = C[[float(m) for m in mats[1:]]].mean(axis=1)
    df = pd.concat([rx_bar.rename("rx"), short.rename("c1"), longm.rename("cl")],
                   axis=1).dropna()
    b, e, fit, r2, X = ols(df["rx"].values, df[["c1", "cl"]].values)
    return pd.Series(fit, index=df.index, name="CiP"), C, r2


def coint_predictor(F: pd.DataFrame):
    """Predittore di cointegrazione diretto (Rebonato-Nyholm JEF 2025): la combinazione
    stazionaria dei forward quasi-unit-root. Engle-Granger: f(max) sui rimanenti; il
    residuo (ECM) e' il predittore; t-ADF riportato come verifica di stazionarieta'."""
    cols = sorted(F.columns)
    d = F[cols].dropna()
    b, e, fit, r2, X = ols(d[cols[-1]].values, d[cols[:-1]].values)
    resid = pd.Series(e, index=d.index, name="COINT")
    return resid, adf_t(resid), b


def predict_table(rx: pd.DataFrame, factors: dict, lags: int = NW_LAGS_12M) -> pd.DataFrame:
    """rx(n) e rx medio su ciascun fattore, separatamente: beta, t-NW, R2."""
    rows = []
    targets = {f"rx({int(n)}y)": rx[n] for n in rx.columns}
    targets["rx(media)"] = rx.mean(axis=1)
    for tname, tv in targets.items():
        for fname, fv in factors.items():
            d = pd.concat([tv.rename("y"), fv.rename("x")], axis=1).dropna()
            if len(d) < 30:
                continue
            b, e, fit, r2, X = ols(d["y"].values, d[["x"]].values)
            t = nw_t(e, X, b, lags)
            rows.append([tname, fname, len(d), b[1], t[1], r2])
    return pd.DataFrame(rows, columns=["target", "fattore", "n", "beta", "t_NW", "R2"])


# ------------------------------------------------------------------ sorprese d'inflazione
def surprises(mkt: str) -> pd.DataFrame:
    """Sorprese d'inflazione mensili. Colonne:
      rw  = pi_t - pi_{t-1} (random walk, proxy)
      ar1 = residuo AR(1) su pi (proxy)
      cons = actual - survey SE esiste an_surprise_{mkt}.parquet (consenso Bloomberg,
             colonne ['actual','survey'] o gia' ['surprise']) -- da verificare a terminale.
    Le proxy sono dichiarate come tali: il consenso e' l'upgrade quando disponibile."""
    pi = yoy(cpi(mkt)).dropna()
    out = pd.DataFrame({"rw": pi.diff()})
    d = pd.concat([pi.rename("p"), pi.shift(1).rename("l")], axis=1).dropna()
    b, e, *_ = ols(d["p"].values, d[["l"]].values)
    out["ar1"] = pd.Series(e, index=d.index)
    p = CACHE / f"an_surprise_{mkt}.parquet"
    if p.exists():
        s = pd.read_parquet(p)
        s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
        out["cons"] = (s["surprise"] if "surprise" in s.columns
                       else s["actual"] - s["survey"])
    return out


def reg_surprise(y_real: pd.DataFrame, surp: pd.DataFrame, mats,
                 lags: int = 6) -> pd.DataFrame:
    """Delta y_reale(n) mensile (bp) sulla sorpresa: beta per maturita' = lettura della
    segmentazione (RR-Giovanni: se la domanda di protezione sale con le sorprese, i
    rendimenti reali SCENDONO -> beta<0; pattern per scadenza = segmentazione)."""
    ye = interp_cols(eom(y_real), mats)
    dy = ye.diff() * 100.0                        # bp
    rows = []
    for sname in surp.columns:
        for n in mats:
            d = pd.concat([dy[float(n)].rename("dy"),
                           surp[sname].rename("s")], axis=1).dropna()
            if len(d) < 30:
                continue
            b, e, fit, r2, X = ols(d["dy"].values, d[["s"]].values)
            t = nw_t(e, X, b, lags)
            rows.append([sname, float(n), len(d), b[1], t[1], r2])
    return pd.DataFrame(rows, columns=["sorpresa", "mat", "n", "beta_bp", "t_NW", "R2"])


def coint_johansen_predictor(F: pd.DataFrame, rx_bar: pd.Series):
    """Predittore di cointegrazione GOLD STANDARD: Johansen (1991) MLE sui forward.
    Rango r dal trace test al 5%; i termini ECM beta'f_t (tutte le relazioni) vengono
    combinati regredendo rx medio sugli ECM (stessa costruzione in-sample del CP: cosi'
    il confronto CP vs COINT e' alla pari, come in Rebonato-Nyholm JEF 2025).
    Fallback Engle-Granger se statsmodels manca. Ritorna (fattore, r, adf_primo_ecm)."""
    cols = sorted(F.columns)
    d = F[cols].replace([np.inf, -np.inf], np.nan).dropna()
    # robustezza numerica: via colonne degeneri (varianza ~0) che rendono singolare la
    # matrice del test; con forward molto collineari e' la causa tipica di fallimento
    d = d.loc[:, d.std() > 1e-8]
    cols = list(d.columns)
    if d.shape[1] < 2 or len(d) < 40:
        print(f"  !! Johansen non applicabile ({d.shape[1]} serie, {len(d)} oss) -> Engle-Granger.")
        resid, adf, _ = coint_predictor(F)
        return resid.rename("COINT"), 1, adf
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError:
        print("  !! statsmodels ASSENTE: COINT ripiega su Engle-Granger (NON Johansen). "
              "pip install statsmodels e rilanciare.")
        resid, adf, _ = coint_predictor(F)
        return resid.rename("COINT"), 1, adf
    try:
        j = coint_johansen(d.values, det_order=0, k_ar_diff=1)
        r = int(np.sum(j.trace_stat > j.trace_stat_crit_vals[:, 1]))   # 5% (riportato)
        # predittore sull'INTERA base di direzioni di cointegrazione (n-1 autovettori):
        # con forward quasi-collineari il trace sotto-seleziona e il primo autovettore
        # e' la combo di minima varianza (rumore); la regressione supervisionata su rx
        # sull'intero spazio evita di perdere la direzione rilevante per i ritorni
        # (coerente con Rebonato-Nyholm: CP = la combinazione di cointegrazione).
        k = d.shape[1] - 1
        ecm = pd.DataFrame(d.values @ j.evec[:, :k], index=d.index,
                           columns=[f"ecm{i+1}" for i in range(k)])
    except Exception as exc:
        print(f"  !! Johansen FALLITO ({type(exc).__name__}: {exc}) -> ripiego su "
              f"Engle-Granger. Il COINT riportato NON e' Johansen.")
        resid, adf, _ = coint_predictor(F)
        return resid.rename("COINT"), 1, adf
    dd = pd.concat([rx_bar.rename("rx"), ecm], axis=1).dropna()
    b, e, fit, r2, X = ols(dd["rx"].values, dd[ecm.columns].values)
    fac = pd.Series(fit, index=dd.index, name="COINT")
    return fac, r, adf_t(ecm.iloc[:, 0])


def reg_surprise_event(y_real: pd.DataFrame, events: pd.DataFrame, mats,
                       window: int = 1) -> pd.DataFrame:
    """EVENT STUDY gold standard (Gurkaynak-Sack-Swanson 2005; Beechey-Wright 2009 sui
    TIPS): variazione del rendimento reale nella FINESTRA del rilascio (chiusura t-1 ->
    chiusura t+window-1, bp) regredita sulla sorpresa (actual - survey). events: indice
    = data di rilascio, colonne actual/survey oppure surprise."""
    s = (events["surprise"] if "surprise" in events.columns
         else events["actual"] - events["survey"]).dropna()
    ye = interp_cols(y_real.sort_index(), mats)
    rows = []
    for n in mats:
        yv = ye[float(n)].dropna()
        obs = []
        for dt, sv in s.items():
            prev = yv.index[yv.index < dt]
            nxt = yv.index[yv.index >= dt]
            if not len(prev) or len(nxt) < window:
                continue
            dy = (yv.loc[nxt[window - 1]] - yv.loc[prev[-1]]) * 100.0
            obs.append((sv, dy))
        if len(obs) < 20:
            continue
        arr = np.array(obs)
        b, e, fit, r2, X = ols(arr[:, 1], arr[:, [0]])
        t = nw_t(e, X, b, 3)
        rows.append([float(n), len(obs), b[1], t[1], r2])
    return pd.DataFrame(rows, columns=["mat", "n_eventi", "beta_bp", "t_NW", "R2"])


# ------------------------------------------------ specificazione Maffei-Rebonato (TIPS)
def isr(mkt: str, mats) -> pd.DataFrame:
    """Curva zero-coupon inflation swap (ISR) dal cache della pipeline: ils_{ticker}.
    UK=BPSWIT (RPI), US=USSWIT (CPI-U). Quotate annuali -> composto continuo (%),
    stessa convenzione delle curve BoE/GSW."""
    tick = {"UK": "BPSWIT", "US": "USSWIT"}[mkt]
    df = pd.read_parquet(CACHE / f"ils_{tick}.parquet")
    df.index = pd.to_datetime(df.index)
    df.columns = [float(c) for c in df.columns]
    df = df.sort_index()
    cc = np.log(1.0 + df / 100.0) * 100.0            # annuale -> cc
    return interp_cols(cc.dropna(how="all"), mats)


def lam_gamma(mkt: str, mats):
    """Maffei-Rebonato: lambda = ISR - BEI (mispricing/liquidita', il fattore latente
    sui linker); gamma = BEI - pi_hat con pi_hat = ISR (projected inflation di mercato)
    -> gamma = -lambda per costruzione quando pi_hat=ISR. Ritorna (BEI, ISR, lambda),
    tutti EOM, cc %."""
    b = eom(bei_us(mats) if mkt == "US" else bei_uk(mats))
    s = eom(isr(mkt, mats))
    idx = b.index.intersection(s.index)
    b, s = b.loc[idx], s.loc[idx]
    return b, s, (s - b)


def surprise_maffei(mkt: str, window_m: int = 120) -> pd.Series:
    """Sorpresa d'inflazione alla Maffei: YoY corrente - media mobile 10 anni (120 mesi)
    dello YoY. Nessun consenso, nessun download: dai soli CPI in cache."""
    pi = yoy(cpi(mkt)).dropna()
    return (pi - pi.rolling(window_m, min_periods=60).mean()).rename("surp")


def reg_lambda(lam: pd.DataFrame, surp: pd.Series, liq: pd.DataFrame | None = None,
               lags: int = 6) -> pd.DataFrame:
    """Regressione 'by T' di Maffei-Rebonato: lambda(T) [bp] su LIQUIDITA' e SORPRESA
    insieme, per scadenza -- la specificazione che risponde a RR ('TIPS yields depend
    on liquidity AND inflation surprises'). Riporta il beta della sorpresa AL NETTO
    della liquidita' (beta_surp) e, per riferimento, il beta della liquidita'.
    Se liq e' None ricade sulla sola sorpresa (univariata). Il pattern di beta_surp
    lungo T e' la segmentazione; il confronto R2(full) vs R2(solo-liq) isola il
    contributo delle sorprese."""
    rows = []
    for n in lam.columns:
        parts = [(lam[n] * 100).rename("l"), surp.rename("s")]
        cols = ["s"]
        if liq is not None:
            parts += [liq[c].rename(c) for c in liq.columns]
            cols += list(liq.columns)
        d = pd.concat(parts, axis=1).dropna()
        if len(d) < 30:
            continue
        b, e, fit, r2, X = ols(d["l"].values, d[cols].values)
        t = nw_t(e, X, b, lags)
        # R2 della sola liquidita' (per isolare il contributo incrementale delle sorprese)
        r2_liq = np.nan
        if liq is not None:
            dl = d[["l"] + list(liq.columns)].dropna()
            _, _, _, r2_liq, _ = ols(dl["l"].values, dl[list(liq.columns)].values)
        beta_liq = b[2] if liq is not None else np.nan
        t_liq = t[2] if liq is not None else np.nan
        rows.append([float(n), len(d), b[1], t[1], beta_liq, t_liq, r2, r2_liq])
    return pd.DataFrame(rows, columns=["mat", "n", "beta_surp", "t_surp",
                                       "beta_liq", "t_liq", "R2_full", "R2_liq_only"])
