"""
04 - Determinazione delle cedole: le tre regole di indicizzazione, codificate dalle
     schede ufficiali MEF (CCT_.pdf, CCTeu.pdf). E' il modulo su cui poggia la replica
     esatta: se la cedola e' sbagliata, la base e' sbagliata, e l'errore e' invisibile.

TRE REGIMI, NON DUE.

  (A) CCT emessi PRIMA del 1-1-1995
      Media aritmetica dei rendimenti all'emissione dei BOT ANNUALI collocati nel bimestre
      che precede di un mese il godimento della cedola; pagamento 8 mesi dopo la
      determinazione. Regime diverso e piu' rumoroso: i CCT emessi 1988-94 scadono
      1995-2001 e sono quindi DENTRO il campione esteso.

  (B) CCT emessi DAL 1-1-1995 (BOT-indexed)
      cedola_sem = round( 0.5 * y_BOT6m(ultima asta prima del godimento) + spread , 0.05 )
      Lo spread si somma DOPO il dimezzamento: in termini annui pesa il doppio.
      Spread per epoca di emissione: 0.50 fino al 1-8-1993, 0.30 dal 1-10-1993 al
      1-9-1996, 0.15 dal 1-11-1996. Convenzione ACT/ACT.

  (C) CCTeu (dal 2010)
      tasso_annuo = round(Euribor6M(-2 gg lav. dal primo giorno di godimento), 3 dec) + spread
      cedola = tasso_annuo * (giorni_effettivi_semestre / 360) * nominale
      Convenzione ACT/360, Modified Following unadjusted.

  Floor (tutti): Circolare MEF 5619 del 21-3-2016, cedola posta a zero se il parametro
  negativo erode e supera lo spread. In v1 il floor e' APPLICATO alla cedola realizzata
  (e' un fatto, non un'opzione) ma NON e' prezzato come opzione nella valutazione
  forward-looking: le date esposte sono marcate e il caveat e' esplicito.

Output: PROC/coupon_schedule.csv (una riga per CCT x data cedolare) + results/05_indexation.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import coupon_dates, save_txt

def cct_spread(issue_date):
    """Spread di legge in punti percentuali, per epoca di emissione."""
    d = pd.Timestamp(issue_date)
    for lim, sp in CCT_SPREAD_BP:
        if d <= pd.Timestamp(lim): return sp
    return CCT_SPREAD_BP[-1][1]

def round_to(x, step):
    return np.round(np.asarray(x, float) / step) * step

def coupon_cct_bot(y_bot_ann, spread, apply_floor=True):
    """(B) cedola semestrale in % del nominale. y_bot_ann in punti percentuali."""
    c = round_to(0.5 * np.asarray(y_bot_ann, float) + spread, CCT_ROUND_STEP)
    return np.maximum(c, 0.0) if apply_floor else c

def coupon_ccteu(eur6m_ann, spread, d_start, d_end, apply_floor=True):
    """(C) cedola semestrale in % del nominale. Convenzione ACT/360 via QuantLib."""
    rate = round_to(np.asarray(eur6m_ann, float), EURIBOR_ROUND) + spread
    if apply_floor: rate = np.maximum(rate, 0.0)
    try:
        from qlutils import accrual_fraction
        frac = accrual_fraction(d_start, d_end, "CCTeu")
    except Exception:
        frac = (pd.Timestamp(d_end) - pd.Timestamp(d_start)).days / 360.0
    return rate * frac

def build_schedule(cct_static, bot_yields=None, eur6m=None, bbg_spreads=None, bot12=None):
    """
    Genera lo scadenzario cedolare per ogni CCT. Le serie dei parametri (aste BOT 6m,
    fixing Euribor) vanno passate come Series indicizzate per data; se assenti, la riga
    e' generata comunque con il parametro a NaN, cosi' si vede subito cosa manca.
    """
    rows = []
    for _, c in cct_static.iterrows():
        if pd.isna(c["issue"]) or pd.isna(c["maturity"]): continue
        iss, mat = pd.Timestamp(c["issue"]), pd.Timestamp(c["maturity"])
        regime = c["regime"]
        if regime == "CCT-BOT":
            rule = "B" if iss >= pd.Timestamp(CCT_RULE_CHANGE) else "A"
            spread = cct_spread(iss)
        else:
            rule, spread = "C", np.nan
            if bbg_spreads is not None and c["isin"] in bbg_spreads.index:
                v = float(bbg_spreads.loc[c["isin"]])
                # FLT_SPREAD arriva in BASIS POINT (range osservato 30-250): in punti
                # percentuali. La soglia distingue i due casi se in futuro cambiasse unita'.
                spread = v / 100.0 if abs(v) > 10 else v
        cds = coupon_dates(iss, mat)
        for k, pay in enumerate(cds):
            # NOTA (E14): per il primo periodo la scheda CCTeu fa decorrere il godimento
            # dalla data di REGOLAMENTO dell'emissione, non dall'issue date. Differenza:
            # fixing spostato di ~2gg e frazione ACT/360 leggermente diversa sulla sola
            # prima cedola dei neo-emessi (pochi bp). Da raffinare con la settlement date.
            start = cds[k-1] if k > 0 else iss
            if rule == "C":
                try:   # -2 giorni lavorativi TARGET (calendario Euribor): il BDay generico
                    from qlutils import CAL, to_ql, from_ql   # sbaglia di un giorno attorno
                    import QuantLib as ql                      # alle feste TARGET (es. 1 maggio)
                    fix_dt = from_ql(CAL.advance(to_ql(start), -EURIBOR_LAG_BD, ql.Days))
                except Exception:
                    fix_dt = start - pd.tseries.offsets.BDay(EURIBOR_LAG_BD)
                par = np.nan if eur6m is None else _asof(eur6m, fix_dt)
                # fixing FUTURO: la cedola non e' determinata -- NaN, non lo spot travestito
                if eur6m is not None and pd.Timestamp(fix_dt) > eur6m.dropna().index.max():
                    par = np.nan
                cpn = coupon_ccteu(par, spread, start, pay) if np.isfinite(par) and np.isfinite(spread) else np.nan
            elif rule == "B":
                par = np.nan if bot_yields is None else _asof(bot_yields, start)
                cpn = float(coupon_cct_bot(par, spread)) if np.isfinite(par) else np.nan
                fix_dt = start
            else:  # regola pre-1995: media BOT annuali del bimestre che precede di un mese
                w0, w1 = start - pd.DateOffset(months=3), start - pd.DateOffset(months=1)
                src = bot12 if bot12 is not None else bot_yields   # ANNUALI se disponibili,
                par = np.nan if src is None else _mean_window(src, w0, w1)  # altrimenti proxy 6M
                cpn = float(coupon_cct_bot(par, spread)) if np.isfinite(par) else np.nan
                fix_dt = w1
            rows.append({"isin": c["isin"], "regime": regime, "rule": rule,
                         "accr_start": start.date(), "pay_date": pay.date(),
                         "fixing_date": pd.Timestamp(fix_dt).date(), "spread_pct": spread,
                         "param_ann": par, "coupon_semi_pct": cpn,
                         "days": (pay - start).days,
                         "dc": DC_CCTEU if rule == "C" else DC_CCT,
                         "floor_exposed": pd.Timestamp(FLOOR_FLAG_FROM) <= start <= pd.Timestamp(FLOOR_FLAG_TO)})
    return pd.DataFrame(rows)

def _asof(s, d):
    s = s.dropna(); s = s[s.index <= pd.Timestamp(d)]
    return float(s.iloc[-1]) if len(s) else np.nan

def _mean_window(s, a, b):
    w = s.dropna(); w = w[(w.index >= pd.Timestamp(a)) & (w.index <= pd.Timestamp(b))]
    return float(w.mean()) if len(w) else np.nan

if __name__ == "__main__":
    print("== 05 indexation ==")
    C = pd.read_csv(PROC/"static_cct.csv", parse_dates=["maturity", "issue"])
    bot = eur = spr = None
    p = PROC/"bot_auction_6m.csv"
    if p.exists():
        bot = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    p12 = PROC/"bot_auction_12m.csv"
    bot12 = (pd.read_csv(p12, index_col=0, parse_dates=True).iloc[:, 0].dropna()
             if p12.exists() else None)
    p = PROC/"curves_market.csv"
    if p.exists():
        cm = pd.read_csv(p, index_col=0, parse_dates=True)
        if "euribor6m" in cm: eur = cm["euribor6m"]
    import glob
    sb = ([str(PROC/"static_bbg.csv")] if (PROC/"static_bbg.csv").exists() else []) \
         + sorted(glob.glob(str(PROC/"static_bbg_*.csv")))
    if sb:
        d = pd.concat([pd.read_csv(f, index_col=0) for f in sb])
        if "flt_spread" in [c.lower() for c in d.columns]:
            col = [c for c in d.columns if c.lower() == "flt_spread"][0]
            d.index = [str(i).split("@")[0].split(" ")[0] for i in d.index]
            spr = pd.to_numeric(d[col], errors="coerce")

    S = build_schedule(C, bot, eur, spr, bot12)
    L = []; P = L.append
    P("=== 05 SCADENZARIO CEDOLARE ===")
    P(f"CCT processati: {S['isin'].nunique()} | righe cedolari: {len(S):,}")
    P(f"per regola: {S.rule.value_counts().to_dict()}   (A=pre-1995, B=BOT post-1995, C=CCTeu)")
    P(f"spread applicati (CCT-BOT): {sorted(S[S.rule.isin(['A','B'])].spread_pct.dropna().unique())}")
    P(f"spread CCTeu da Bloomberg: {'presenti' if spr is not None else 'MANCANTI (tirare FLT_SPREAD)'}")
    P(f"parametro di indicizzazione: {S.param_ann.notna().sum():,}/{len(S):,} righe popolate")
    if S.param_ann.isna().all():
        P("  [!] nessun parametro: servono bot_auction_6m.csv (aste BOT 6m) e euribor6m in curves_market.csv")
    P(f"cedole esposte al floor (accrual 2015-11 -> 2022-07): {S.floor_exposed.sum():,}")
    P(f"convenzioni: {S.dc.value_counts().to_dict()}")
    # regola A: BOT ANNUALI se bot_auction_12m.csv esiste (norma pre-1995), altrimenti
    # PROXY dai 6M -- flag esplicito perche' nulla usi il proxy in silenzio.
    # onesto: "12m" solo dove il parametro e' stato DAVVERO trovato nella serie annuale;
    # "no_data" dove nessuna serie BOT copre il periodo (cedole pre-1996: Bloomberg non
    # ha prezzi BOT prima del 1996, quindi la regola A pre-1995 non e' ricostruibile).
    src_lbl = "12m" if bot12 is not None else "6m_PROXY"
    S["param_series"] = np.select(
        [S["rule"].ne("A"),
         S["rule"].eq("A") & S["param_ann"].notna(),
         S["rule"].eq("A") & S["param_ann"].isna()],
        ["native", src_lbl, "no_data"],
        default="native")
    S.to_csv(PROC/"coupon_schedule.csv", index=False); P(f"[saved] {PROC/'coupon_schedule.csv'}")
    save_txt("05_indexation.txt", L); print("\n".join(L))
