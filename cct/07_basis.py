"""
07 - LE TRE BASI CCT-BTP.

COSTRUZIONE. Si compra il CCT a prezzo di mercato e si entra in un IRS alla pari (valore
iniziale nullo) in cui si paga Euribor 6M e si riceve il fisso K di pari scadenza. Il flusso
netto per periodo, con delta la frazione di rateo:

    max(E+s, 0)*delta  -  E*delta  +  K*delta

Se il floor NON e' attivo il netto e' (s+K)*delta: il pacchetto e' un'obbligazione a tasso
fisso con cedola (s+K), comprata al prezzo del CCT. Da qui YTM e prezzo confrontabili col BTP.

IL FLOOR NELLA COSTRUZIONE, NON NELLE ROBUSTEZZE. Se il floor e' attivo (E+s<0) la cedola
del CCT resta 0 ma la gamba variabile dello swap si paga comunque, e con E negativo pagarla
significa INCASSARE: il netto diventa (K-E)*delta, maggiore di (s+K)*delta. Quindi

    CCT swappato = obbligazione fissa a (s+K)  +  strip di floorlet su (E+s), strike 0

Il pacchetto incorpora una posizione LUNGA in un floor: lo rende piu' prezioso, il prezzo
sale, il rendimento sintetico scende, e la base misurata risulta distorta VERSO IL BASSO del
valore del floor. In v1 il floor non e' prezzato come opzione: si riporta la DISTANZA DAL
FLOOR (E+s) come diagnostica per cedola, e le coppie/date esposte sono marcate. La
decomposizione base_misurata = base_vera + valore_floor resta esplicita.

DUE REGIMI, DUE CONVERSIONI IN FISSO. Tutti i CCT entrano, ma la gamba variabile non e'
la stessa e non puo' essere trattata allo stesso modo.

  CCTeu  - cedola annua = Euribor6M + s. La gamba variabile dello swap EUR standard E'
           Euribor6M: l'asset swap converte esattamente, sintetico = s + K_eur(tau) con
           K_eur il tasso par interpolato sulla griglia EUSA. Conversione DI MERCATO.

  CCT-BOT- cedola annua = y_BOT + 2s (lo spread si somma dopo il dimezzamento, quindi in
           termini annui pesa doppio). Non esiste un mercato swap contro BOT, MA non serve:
           y_BOT e' il tasso a breve DELLO STESSO EMITTENTE che paga la cedola. Un floater
           indicizzato alla curva su cui viene scontato prezza alla pari a ogni reset, quindi
           il fisso equivalente alla gamba variabile e' il PAR YIELD della curva sovrana
           gia' fittata in 06. Sintetico = 2s + K_sov(tau). Conversione DA MODELLO.

           Perche' NON si usa un IRS su Euribor per i CCT-BOT: la gamba variabile sarebbe
           credito bancario europeo contro un indice che e' credito sovrano italiano. Nel
           novembre 2011 il divario valeva oltre 400 bp (BOT 6m sopra il 6%, Euribor 1.7%):
           finirebbe tutto dentro la "base". La colonna K_source distingue i due casi, e
           ogni risultato va letto sapendo che per i CCT-BOT la conversione e' model-based:
           e' una misura piu' debole di quella dei CCTeu, non equivalente.

LE MISURE (in spazio PREZZI e RENDIMENTI, come Fleckenstein-Longstaff 2020):
  (1) ISIN vs ISIN    - solo in spazio RENDIMENTI. Il confronto e' diretto: la ricchezza o
                        economicita' del singolo BTP fa parte della base e NON va rimossa.
                        NIENTE versione in prezzo: due titoli con cedole diverse (differenza
                        mediana 2.88 punti) differiscono in prezzo di 13 punti a 5 anni e 24
                        a 10 con mispricing NULLO, solo per effetto cedola. FL confrontano i
                        prezzi perche' il loro replicante ha flussi IDENTICI; qui non lo sono.
  (3) ISIN vs curva   - sintetico CCT contro un titolo con gli STESSI flussi prezzato sulla
                        curva nominale. Flussi identici, quindi il confronto in prezzo E'
                        valido ed e' la misura primaria (gold standard FL).

CONTEGGIO DEI FLUSSI: ceil, non round. Servono TUTTE le cedole successive alla data di
valutazione. Con tau=3.2 anni le cedole residue sono 7 (3.2, 2.7, ..., 0.2) ma round(6.4)=6:
si perde la cedola piu' vicina, che vale ~2 punti di prezzo e ~10 bp di rendimento. round
sottostima ogni volta che la parte frazionaria di tau*freq e' sotto 0.5, cioe' in META' delle
osservazioni -- ed era l'intera origine del divario fra la misura (1) e la (3).

DIAGNOSTICA wedge_y. Per la misura (1) si riporta accanto la differenza di YTM che la SOLA
curva implicherebbe fra sintetico e BTP a mispricing nullo -- cioe' la parte meccanica dovuta
a cedola e scadenza diverse (0.6 bp a 3 anni, 1.5 a 5, 2.9 a 7, 5.6 a 10). Non viene sottratta:
sta accanto, cosi' si vede quanta parte della differenza grezza e' meccanica.

Output: PROC/basis_daily.csv, PROC/basis_curve.csv, results/07_basis.txt
"""
import numpy as np, pandas as pd
from scipy.optimize import brentq
from config import *
from utils import save_txt

YEARS_SWAP_ARR = np.array(YEARS_SWAP, float)

# --- ammissibilita' della misura (1) --------------------------------------------
# Lo YTM e' iperselettivo al prezzo quando la duration e' piccola: 0.5 punti di prezzo
# valgono 10 bp su un quinquennale e 167 bp su un titolo a 0.3 anni. Lo stesso vale per il
# disallineamento di scadenza, che conta in proporzione alla vita residua. La misura (1)
# resta calcolata per tutti, ma e' marcata utilizzabile solo dove i due effetti sono piccoli.
# La misura (3) NON e' toccata: usa la scadenza esatta del CCT, quindi il disallineamento
# non esiste per costruzione. E' un'altra ragione per cui e' la misura primaria.
M1_MAX_MISMATCH_FRAC = 0.05     # disallineamento <= 5% della vita residua
M1_MIN_TAU           = 1.0      # e almeno un anno di vita residua

def coupon_dates_back(issue, maturity, freq=CPN_FREQ):
    """Date cedolari a ritroso dalla scadenza, incluse quelle passate (servono al rateo)."""
    from dateutil.relativedelta import relativedelta
    step = 12 // freq
    lim = pd.Timestamp(issue) if pd.notna(issue) else pd.Timestamp(maturity) - relativedelta(years=50)
    ds, d = [], pd.Timestamp(maturity)
    while d > lim:
        ds.append(d); d -= relativedelta(months=step)
    return np.array(sorted(ds))

def accrued_fixed(d, dates, coupon, freq=CPN_FREQ):
    """
    Rateo ACT/ACT-ICMA alla data d, sulle date cedolari EFFETTIVE.

    Sostituisce l'approssimazione precedente, che ricavava il rateo da una griglia
    sintetica di mezzi anni: quando round(tau*freq) arrotondava in giu' il tempo alla
    prossima cedola risultava > 1/freq e il codice assegnava UNA CEDOLA INTERA di rateo
    invece della frazione maturata. Accadeva in circa meta' delle osservazioni e valeva
    50-60 bp di errore di rendimento -- l'intero divario fra la misura (1) e la (3).
    """
    if not coupon: return 0.0
    nxt = dates[dates > d]
    if len(nxt) == 0: return 0.0
    nxt = nxt[0]
    prv = dates[dates <= d]
    prv = prv[-1] if len(prv) else nxt - pd.DateOffset(months=12 // freq)
    per = (nxt - prv).days
    return 0.0 if per <= 0 else float(coupon) / freq * ((d - prv).days / per)

def ytm(price_dirty, taus, amts, guess=0.03):
    """Rendimento a scadenza (composto continuo) di un flusso fisso, dal prezzo sporco."""
    f = lambda y: float(np.sum(amts * np.exp(-y * taus))) - price_dirty
    try:
        if f(-0.05) * f(0.60) > 0: return np.nan
        return brentq(f, -0.05, 0.60, maxiter=200, xtol=1e-10)
    except Exception:
        return np.nan

def nss(tau, p):
    b0, b1, b2, b3, t1, t2 = p
    tau = np.maximum(np.asarray(tau, float), 1e-8)
    x1, x2 = tau / t1, tau / t2
    f1 = (1 - np.exp(-x1)) / x1; f2 = (1 - np.exp(-x2)) / x2
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))

def par_yield_curve(p_par, tau, freq=CPN_FREQ):
    """
    Par yield della curva sovrana a scadenza tau: il tasso fisso c tale che
    sum(c/freq * DF_i) + DF(tau) = 1. E' il fisso equivalente a una gamba variabile
    indicizzata al tasso a breve della curva stessa -- quindi la conversione in fisso
    dei CCT-BOT, che sono indicizzati al rendimento BOT del medesimo emittente.
    """
    n = max(int(np.ceil(tau * freq - 1e-9)), 1)          # ceil, non round: v. nota in __main__
    t = np.array([(i + 1) / freq for i in range(n)], float)
    t = t - (t[-1] - tau); t = t[t > 0]
    if len(t) == 0: return np.nan
    df = np.exp(-nss(t, p_par) / 100.0 * t)
    ann = float(np.sum(df)) / freq
    return np.nan if ann <= 0 else float((1.0 - df[-1]) / ann) * 100.0

def swap_rate(row_irs, tau):
    """Tasso swap alla scadenza tau, interpolato sulla griglia EUSA."""
    v = row_irs.dropna()
    if len(v) < 3: return np.nan
    yy = np.array([float(str(i).replace("irs", "").replace("y", "")) for i in v.index])
    o = np.argsort(yy)
    return float(np.interp(tau, yy[o], v.values[o]))

if __name__ == "__main__":
    print("== 07 basis ==")
    PAIRS = pd.read_csv(PROC/"pairs_cct_btp.csv", parse_dates=["CCT_mat","BTP_mat","basis_start","basis_end"])
    SCHED = pd.read_csv(PROC/"coupon_schedule.csv", parse_dates=["accr_start","pay_date","fixing_date"])
    CRV   = pd.read_csv(PROC/"curve_params.csv", index_col=0, parse_dates=True)
    MKT   = pd.read_csv(PROC/"curves_market.csv", index_col=0, parse_dates=True)
    PXC   = pd.read_csv(PROC/"px_cct.csv", index_col=0, parse_dates=True)
    PXB   = pd.read_csv(PROC/"px_btp.csv", index_col=0, parse_dates=True)
    SB    = pd.read_csv(PROC/"static_btp.csv", parse_dates=["maturity","issue"]).set_index("isin")
    IRSC  = [c for c in MKT.columns if c.startswith("irs")]

    # date cedolari dei BTP, precomputate una volta per titolo (servono al rateo)
    BTP_DATES = {}
    for isin, r in SB.iterrows():
        if pd.notna(r["maturity"]):
            try: BTP_DATES[isin] = coupon_dates_back(r.get("issue"), r["maturity"])
            except Exception: pass

    # spread cedolare per CCT (dal calendario cedolare: costante per titolo)
    spread = SCHED.groupby("isin")["spread_pct"].median()
    # ultima cedola nota e prossima, per il rateo del CCT
    sched_by = {k: v.sort_values("pay_date") for k, v in SCHED.groupby("isin")}

    print(f"  coppie: {len(PAIRS)} "
          f"({(PAIRS.regime=='CCTeu').sum()} CCTeu, {(PAIRS.regime=='CCT-BOT').sum()} CCT-BOT)")

    rows = []
    for _, pr in PAIRS.iterrows():
        cct, btp = pr["CCT_ISIN"], pr["BTP_ISIN"]
        if cct not in PXC.columns or btp not in PXB.columns: continue
        s = spread.get(cct, np.nan)
        if not np.isfinite(s): continue
        sc = sched_by.get(cct)
        if sc is None: continue
        b = SB.loc[btp] if btp in SB.index else None
        if b is None or pd.isna(b["coupon"]): continue

        idx = (PXC[cct].dropna().index.intersection(PXB[btp].dropna().index)
                       .intersection(CRV.index).intersection(MKT.index))
        idx = idx[(idx >= pr["basis_start"]) & (idx <= pr["basis_end"])]
        if len(idx) == 0: continue

        for d in idx:
            tau_c = (pr["CCT_mat"] - d).days / 365.25
            tau_b = (pr["BTP_mat"] - d).days / 365.25
            if tau_c < 0.25 or tau_b < 0.25: continue
            p_par = CRV.loc[d, ["b0","b1","b2","b3","t1","t2"]].values.astype(float)
            if not np.isfinite(p_par).all(): continue

            # --- conversione in fisso, diversa per regime (v. docstring)
            if pr["regime"] == "CCTeu":
                K = swap_rate(MKT.loc[d, IRSC], tau_c); k_src = "market_irs"
                cpn_syn = s + K                      # spread sommato al tasso ANNUO
            else:
                K = par_yield_curve(p_par, tau_c);    k_src = "curve_par"
                cpn_syn = 2.0 * s + K                # spread sommato al SEMESTRALE -> doppio
            if not np.isfinite(K): continue
            n = max(int(np.ceil(tau_c * CPN_FREQ - 1e-9)), 1)
            t_syn = np.array([(i + 1) / CPN_FREQ for i in range(n)])
            t_syn = t_syn - (t_syn[-1] - tau_c)                       # allinea l'ultimo flusso
            t_syn = t_syn[t_syn > 0]
            a_syn = np.full(len(t_syn), cpn_syn / CPN_FREQ); a_syn[-1] += 100.0

            # rateo del CCT dalla cedola in corso
            past = sc[sc.pay_date <= d]; nxt = sc[sc.pay_date > d]
            acc_c = 0.0
            if len(nxt):
                st = past.pay_date.iloc[-1] if len(past) else nxt.accr_start.iloc[0]
                per = (nxt.pay_date.iloc[0] - st).days
                cpn_real = nxt.coupon_semi_pct.iloc[0]
                if per > 0 and np.isfinite(cpn_real):
                    acc_c = float(cpn_real) * ((d - st).days / per)
            p_cct = float(PXC.loc[d, cct]) + acc_c

            # --- BTP appaiato
            cb = float(b["coupon"])
            nb = max(int(np.ceil(tau_b * CPN_FREQ - 1e-9)), 1)
            t_b = np.array([(i + 1) / CPN_FREQ for i in range(nb)])
            t_b = t_b - (t_b[-1] - tau_b); t_b = t_b[t_b > 0]
            a_b = np.full(len(t_b), cb / CPN_FREQ); a_b[-1] += 100.0
            bd = BTP_DATES.get(btp)
            acc_b = accrued_fixed(d, bd, cb) if bd is not None else np.nan
            if not np.isfinite(acc_b): continue
            p_btp = float(PXB.loc[d, btp]) + acc_b

            # --- valutazione sulla curva nominale del giorno
            z_syn = nss(t_syn, p_par) / 100.0
            p_fair_syn = float(np.sum(a_syn * np.exp(-z_syn * t_syn)))    # sintetico "giusto"
            z_b = nss(t_b, p_par) / 100.0
            p_fair_btp = float(np.sum(a_b * np.exp(-z_b * t_b)))

            y_syn      = ytm(p_cct, t_syn, a_syn)
            y_btp      = ytm(p_btp, t_b, a_b)
            y_fair     = ytm(p_fair_syn, t_syn, a_syn)
            y_fair_btp = ytm(p_fair_btp, t_b, a_b)
            # cuneo meccanico: differenza di YTM che la sola curva implica fra i due titoli
            # (cedole e scadenze diverse), a mispricing nullo. Diagnostica, NON correzione.
            wedge = ((y_fair - y_fair_btp) * 1e4
                     if np.isfinite(y_fair) and np.isfinite(y_fair_btp) else np.nan)

            # --- distanza dal floor (diagnostica v1)
            dist = np.nan
            if len(nxt) and np.isfinite(nxt.param_ann.iloc[0]):
                dist = float(nxt.param_ann.iloc[0]) + s if pr["regime"] == "CCTeu" else np.nan

            rows.append({"date": d, "CCT_ISIN": cct, "BTP_ISIN": btp, "regime": pr["regime"],
                "tau_cct": tau_c, "tau_btp": tau_b, "mismatch_d": pr["mismatch_days"],
                # Il disallineamento va letto RELATIVO alla vita residua, non in giorni:
                # 31 giorni sono l'1.7% di un quinquennale e il 12% di un titolo a 0.7 anni.
                # Sotto la soglia la misura (1) e' dominata dal disallineamento, non dalla base.
                "mismatch_frac": abs(pr["mismatch_days"]) / max(tau_c * 365.25, 1.0),
                "m1_usable": bool(abs(pr["mismatch_days"]) / max(tau_c * 365.25, 1.0) <= M1_MAX_MISMATCH_FRAC
                                  and tau_c >= M1_MIN_TAU),
                "swap_K": K, "K_source": k_src, "spread": s,
                "cpn_syn": cpn_syn, "cpn_btp": cb,
                "p_cct_dirty": p_cct, "p_btp_dirty": p_btp,
                "p_fair_syn": p_fair_syn, "p_fair_btp": p_fair_btp,
                # (1) ISIN vs ISIN, solo rendimenti (in prezzo non ha significato: v. docstring)
                "basis1_y": (y_syn - y_btp) * 1e4 if np.isfinite(y_syn) and np.isfinite(y_btp) else np.nan,
                "wedge_y": wedge,
                # (3) ISIN vs curva: flussi identici, valida in ENTRAMBI gli spazi
                "basis3_y": (y_syn - y_fair) * 1e4 if np.isfinite(y_syn) and np.isfinite(y_fair) else np.nan,
                "basis3_p": p_cct - p_fair_syn,
                # residuo del BTP sulla curva: diagnostica, NON una correzione
                "btp_resid_p": p_btp - p_fair_btp,
                "floor_dist": dist})

    B = pd.DataFrame(rows)
    B.to_csv(PROC/"basis_daily.csv", index=False)

    L=[]; P=L.append
    P("=== 07 BASI CCT-BTP ===")
    P(f"osservazioni: {len(B):,} | coppie {B.CCT_ISIN.nunique()} CCT x {B.BTP_ISIN.nunique()} BTP")
    P(f"periodo: {B.date.min().date()} -> {B.date.max().date()}")
    P("conversione in fisso: " + str(B.K_source.value_counts().to_dict()))
    P("  market_irs = asset swap su Euribor (CCTeu, conversione esatta di mercato)")
    P("  curve_par  = par yield della curva sovrana (CCT-BOT, conversione da modello)")
    for reg, g in B.groupby("regime"):
        P(f"\n{'='*66}\n{reg}  ({len(g):,} oss., {g.CCT_ISIN.nunique()} CCT, "
          f"{g.date.min().date()} -> {g.date.max().date()})\n{'='*66}")
        P("  MISURA PRIMARIA (3) ISIN vs curva -- flussi identici, gold standard FL:")
        for lab, c, u in [("prezzo", "basis3_p", "per 100"), ("yield ", "basis3_y", "bp")]:
            v = g[c].dropna()
            if len(v): P(f"    {lab}: mediana {v.median():8.3f} {u} | media {v.mean():8.3f} | "
                         f"IQR [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}] | n {len(v):,}")
        v = g["basis1_y"].dropna(); w = g["wedge_y"].dropna()
        gu = g[g.m1_usable]
        vu = gu["basis1_y"].dropna()
        P("  MISURA (1) ISIN vs ISIN, solo rendimenti:")
        if len(v): P(f"    tutte le oss.   : mediana {v.median():8.1f} bp | "
                     f"IQR [{v.quantile(.25):.1f}, {v.quantile(.75):.1f}] | n {len(v):,}")
        if len(vu): P(f"    campione AMMISS.: mediana {vu.median():8.1f} bp | "
                      f"IQR [{vu.quantile(.25):.1f}, {vu.quantile(.75):.1f}] | n {len(vu):,}"
                      f"  ({len(vu)/max(len(v),1):.0%})")
        if len(vu):
            dd = (gu.basis1_y - gu.basis3_y).dropna()
            P(f"    scarto da (3) sul campione ammissibile: mediana {dd.median():.1f} bp, "
              f"|scarto|<25bp nel {(dd.abs()<25).mean():.0%} dei casi")
        if len(w): P(f"    cuneo meccanico : mediana {w.median():8.1f} bp | "
                     f"IQR [{w.quantile(.25):.1f}, {w.quantile(.75):.1f}]  (NON sottratto)")
        v = g["btp_resid_p"].dropna()
        if len(v): P(f"    [diagn.] residuo BTP sulla curva: mediana {v.median():.3f}, sd {v.std():.3f}")
    P("\n--- serie annuale, misura primaria (3), per regime ---")
    P("    prezzo = punti per 100 di nominale (negativo = CCT sotto il fair -> economico)")
    P("    bp     = differenza di rendimento (positivo = CCT rende di piu' -> economico)")
    B["yr"] = B.date.dt.year
    P(f"  {'anno':>6}{'CCT-BOT p':>11}{'CCT-BOT bp':>12}{'CCTeu p':>10}{'CCTeu bp':>11}{'n':>8}")
    nan = float("nan")
    for y, g in B.groupby("yr"):
        a, b = g[g.regime == "CCT-BOT"], g[g.regime == "CCTeu"]
        ap, ay = a.basis3_p.dropna(), a.basis3_y.dropna()
        bp_, by = b.basis3_p.dropna(), b.basis3_y.dropna()
        if len(ap) + len(bp_) > 50:
            P(f"  {y:>6}{(ap.median() if len(ap) else nan):>11.3f}{(ay.median() if len(ay) else nan):>12.1f}"
              f"{(bp_.median() if len(bp_) else nan):>10.3f}{(by.median() if len(by) else nan):>11.1f}"
              f"{len(ap)+len(bp_):>8,}")
    fl = B["floor_dist"].dropna()
    if len(fl):
        P(f"\nFLOOR (solo CCTeu): distanza E+s mediana {fl.median():.3f}%, minimo {fl.min():.3f}%")
        P(f"  osservazioni con floor ATTIVO (E+s<0): {(fl<0).sum():,} ({(fl<0).mean():.2%})")
        P(f"  entro 25bp dal floor: {(fl<0.25).sum():,} ({(fl<0.25).mean():.2%})")
        P("  [v1] il floor NON e' prezzato come opzione: dove la distanza e' piccola la base")
        P("       misurata e' distorta verso il basso del valore del floorlet.")
    P("\n=== COERENZA FRA LE DUE MISURE ===")
    P("  identita' attesa:  basis1 = basis3 + cuneo + (quanto il BTP si discosta dalla curva)")
    B["btp_off_curve_bp"] = B.basis1_y - B.basis3_y - B.wedge_y
    for reg, g in B.groupby("regime"):
        d1, d3 = g.basis1_y.dropna(), g.basis3_y.dropna()
        off = g.btp_off_curve_bp.dropna()
        P(f"  {reg:8s}: basis1 {d1.median():7.1f} | basis3 {d3.median():7.1f} | "
          f"differenza {d1.median()-d3.median():7.1f} bp")
        P(f"  {'':8s}  di cui BTP fuori curva: mediana {off.median():.1f} bp, "
          f"IQR [{off.quantile(.25):.1f}, {off.quantile(.75):.1f}]")
    P("  [attesa] le due misure devono essere VICINE: il BTP appaiato sta quasi sulla curva,")
    P("           quindi confrontarsi con lui o con la curva alla sua scadenza e' quasi lo stesso.")

    P("\nCUNEI, riportati non corretti:")
    P(f"  livello cedolare: |cpn_syn - cpn_btp| mediana "
      f"{(B.cpn_syn - B.cpn_btp).abs().median():.2f} punti pct")
    P(f"  disallineamento di scadenza: |mismatch| mediana {B.mismatch_d.abs().median():.0f} gg")
    P(f"\n[saved] {PROC/'basis_daily.csv'}")
    save_txt("07_basis.txt", L); print("\n".join(L))
