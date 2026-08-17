"""
02 - Appaiamento ISIN-vs-ISIN con vincolo di convivenza.

REGOLA (congelata). Per ogni CCT, fra i BTP con |mismatch di scadenza| <= MAX_MISMATCH_D:
  1. tieni solo quelli con COPERTURA >= MIN_COVERAGE della finestra utile del CCT
     -- la finestra utile e' [max(inizio campione, emissione CCT), scadenza CCT], e la
     copertura e' la frazione di quella finestra in cui ANCHE il BTP e' vivo. Senza questo
     vincolo la base non e' sfruttabile: non puoi tenere una gamba che non esiste ancora.
  2. fra i superstiti, minimo |mismatch|;
  3. a parita', preferisci il BTP che scade PRIMA del CCT (preferenza rivelata di FLL);
  4. a ulteriore parita', l'on-the-run (emesso piu' di recente).
Se nessun BTP raggiunge la copertura minima, si tiene quello a copertura massima e la
coppia e' marcata 'partial': la serie parte comunque dalla data in cui entrambi vivono.

Output: PROC/pairs_cct_btp.csv + results/04_pairing.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

print("== 04 pairing ==")
C = pd.read_csv(PROC/"static_cct.csv", parse_dates=["maturity","issue"])
B = pd.read_csv(PROC/"static_btp.csv", parse_dates=["maturity","issue"])
B = B.dropna(subset=["issue","maturity"])

def build(start, universe_label):
    S0 = pd.Timestamp(start); END = pd.Timestamp(END_SAMPLE)
    rows, dropped = [], []
    for _, c in C[C.maturity > S0].iterrows():
        eff = max(c.issue, S0) if pd.notna(c.issue) else S0
        tot = (min(c.maturity, END) - eff).days
        if tot <= 0: continue
        cand = B[(B.maturity - c.maturity).abs() <= pd.Timedelta(days=MAX_MISMATCH_D)].copy()
        if cand.empty:
            dropped.append((c["isin"], c.maturity.date(), "nessun BTP entro soglia")); continue
        cand["mm"] = (cand.maturity - c.maturity).dt.days
        st = np.maximum(cand.issue.values.astype("datetime64[D]"), np.datetime64(eff.date()))
        en = np.minimum(cand.maturity.values.astype("datetime64[D]"),
                        np.datetime64(min(c.maturity, END).date()))
        cand["covg"] = np.clip((en - st).astype(int), 0, None) / tot
        ok = cand[cand["covg"] >= MIN_COVERAGE]
        flag = "ok"
        if ok.empty:
            ok = cand.loc[[cand["covg"].idxmax()]]; flag = "partial"
        o = np.lexsort((-ok.issue.values.astype("int64"), (ok.mm.values > 0).astype(int),
                        np.abs(ok.mm.values)))[0]
        b = ok.iloc[o]
        bs = max(pd.Timestamp(b.issue), eff); be = min(c.maturity, b.maturity, END)
        months = (be.to_period("M") - bs.to_period("M")).n
        if months < MIN_MONTHS_PAIR:
            dropped.append((c["isin"], c.maturity.date(), f"finestra {months}m < {MIN_MONTHS_PAIR}")); continue
        rows.append({"CCT_ISIN": c["isin"], "CCT_mat": c.maturity.date(), "CCT_issue":
                     c.issue.date() if pd.notna(c.issue) else None, "regime": c.regime,
                     "BTP_ISIN": b["isin"], "BTP_mat": b.maturity.date(), "BTP_issue": b.issue.date(),
                     "BTP_coupon": b.coupon, "mismatch_days": int(b.mm),
                     "coverage": round(float(b["covg"]), 3), "basis_start": bs.date(),
                     "basis_end": be.date(), "months": months, "flag": flag,
                     "floor_exposed": not (be < pd.Timestamp(FLOOR_FLAG_FROM) or
                                           bs > pd.Timestamp(FLOOR_FLAG_TO))})
    return pd.DataFrame(rows), dropped

L = []; P = L.append
P("=== 04 APPAIAMENTO CCT -> BTP ===")
P(f"regola: |mismatch|<={MAX_MISMATCH_D}gg, copertura>={MIN_COVERAGE:.0%}, min {MIN_MONTHS_PAIR} mesi")
best = None
for lab, s in [("PRIMARIO (1999)", START_PRIMARY), ("ESTESO (1995)", START_EXTENDED)]:
    Pr, dr = build(s, lab)
    P(f"\n--- {lab} ---")
    P(f"  coppie: {len(Pr)} | scartate: {len(dr)} | BTP distinti: {Pr.BTP_ISIN.nunique() if len(Pr) else 0}")
    if len(Pr):
        P(f"  per regime: {Pr.regime.value_counts().to_dict()}")
        P(f"  mismatch |gg|: mediana {Pr.mismatch_days.abs().median():.0f}, "
          f"0-31 {(Pr.mismatch_days.abs()<=31).sum()}, 32-92 {(Pr.mismatch_days.abs()>31).sum()}")
        P(f"  copertura: mediana {Pr.coverage.median():.3f} | full {(Pr.flag=='ok').sum()} | partial {(Pr.flag=='partial').sum()}")
        P(f"  mesi osservabili per coppia: mediana {Pr.months.median():.0f}, totale {Pr.months.sum():,}")
        P(f"  coppie esposte al floor CCTeu (2015-2022): {Pr.floor_exposed.sum()}")
    for d in dr[:5]: P(f"    scartata {d[0]} (mat {d[1]}): {d[2]}")
    if s == START_EXTENDED: best = Pr
best.to_csv(PROC/"pairs_cct_btp.csv", index=False)
P(f"\n[saved] {PROC/'pairs_cct_btp.csv'} (campione esteso; filtrare per data nei passi successivi)")
save_txt("04_pairing.txt", L); print("\n".join(L))
