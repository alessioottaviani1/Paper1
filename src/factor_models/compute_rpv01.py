"""
RPV01 (risky annuity) per tutti i CDS / CDS-index di CDS.xlsx via ISDA Standard Model (QuantLib).
- single name: spread 1Y+5Y -> hazard 2 segmenti -> RPV01_1Y e RPV01_5Y
- index:       spread 5Y     -> hazard 1 segmento  -> RPV01_5Y
- sconto: curva PIATTA al tasso breve EUR (EUR001M), second-order (validato ~1% vs BBG).
  NB: i nomi USD usano lo stesso sconto EUR (approssimazione second-order, washed dalla standardizzazione).
- recovery: 40% (tutti senior).  Pre-Big-Bang (2004-2009) le quote sono par spread -> bootstrap diretto.
"""
import re, numpy as np, pandas as pd, openpyxl, QuantLib as ql

RECOVERY = 0.40

def _isnum(x):
    try: float(x); return True
    except: return False

def imm_prev(d):
    cand = [ql.Date(20, m, d.year()) for m in (3,6,9,12)] + [ql.Date(20,12,d.year()-1)]
    return max([x for x in cand if x <= d])

def risky_annuity(pydate, sprd_by_tenor, target, r_disc, R=RECOVERY):
    d = ql.Date(pydate.day, pydate.month, pydate.year)
    ql.Settings.instance().evaluationDate = d
    disc = ql.YieldTermStructureHandle(ql.FlatForward(d, float(r_disc), ql.Actual365Fixed()))
    helpers = [ql.SpreadCdsHelper(ql.QuoteHandle(ql.SimpleQuote(s/1e4)), ql.Period(t), 1,
               ql.WeekendsOnly(), ql.Quarterly, ql.Following, ql.DateGeneration.CDS2015,
               ql.Actual360(), R, disc) for t, s in sorted(sprd_by_tenor.items())]
    hz = ql.DefaultProbabilityTermStructureHandle(ql.PiecewiseFlatHazardRate(d, helpers, ql.Actual365Fixed()))
    mat = ql.cdsMaturity(d, ql.Period(target), ql.DateGeneration.CDS2015)
    sched = ql.Schedule(imm_prev(d), mat, ql.Period(ql.Quarterly), ql.WeekendsOnly(),
                        ql.Following, ql.Unadjusted, ql.DateGeneration.CDS2015, False)
    cds = ql.CreditDefaultSwap(ql.Protection.Buyer, 1e7, 1e-4, sched, ql.Following, ql.Actual360())
    cds.setPricingEngine(ql.IsdaCdsEngine(hz, R, disc))
    return abs(cds.couponLegNPV())/1e-4/1e7


if __name__ == "__main__":
    CDS_XLSX = "CDS.xlsx"          # <-- path al tuo CDS.xlsx
    BBG_XLSX = "r1.xlsx"           # <-- path al bbg.xlsx (per EUR001M); metti il tuo

    # ---------- 1. parse spreads da Sheet1 ----------
    wb = openpyxl.load_workbook(CDS_XLSX, read_only=True, data_only=True); ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True)); tick, fld = rows[3], rows[5]
    dates = pd.to_datetime([r[0] for r in rows[6:]], errors="coerce")
    def col(j):
        return pd.Series([(float(r[j]) if _isnum(r[j]) else np.nan) for r in rows[6:]], index=dates)
    def short(tk):
        iss = re.split(r" CDS", tk)[0].strip()
        ten = "5Y" if " 5Y" in tk else ("1Y" if " 1Y" in tk else "?")
        return iss.replace(" ", "_"), ten
    spreads, rpv_bbg = {}, {}        # key=(issuer,tenor) -> Series
    for j in range(len(fld)):
        if fld[j] == "PX_LAST" and tick[j]:
            iss, ten = short(str(tick[j]).strip())
            spreads[(iss, ten)] = col(j)
            if j+1 < len(fld) and fld[j+1] == "SW_CNV_RISK":
                rpv_bbg[(iss, ten)] = col(j+1).abs()
    issuers = sorted(set(k[0] for k in spreads))

    # ---------- 2. sconto EUR001M ----------
    wb2 = openpyxl.load_workbook(BBG_XLSX, read_only=True, data_only=True); ws2 = wb2["rates_fx"]
    r2 = list(ws2.iter_rows(values_only=True)); t2 = r2[3]
    je = [i for i in range(len(t2)) if t2[i] and "EUR001M" in str(t2[i])][0]
    disc_rate = pd.Series([float(r[je]) if _isnum(r[je]) else np.nan for r in r2[6:]],
                          index=pd.to_datetime([r[0] for r in r2[6:]], errors="coerce")).ffill()/100.0

    # ---------- 3. loop: RPV01 per strumento e tenor ----------
    out = {}
    for iss in issuers:
        s1 = spreads.get((iss, "1Y")); s5 = spreads.get((iss, "5Y"))
        idx = (s5 if s5 is not None else s1).index
        for ten in (["1Y", "5Y"] if s1 is not None else ["5Y"]):
            if (iss, ten) not in spreads: continue
            res = pd.Series(index=idx, dtype=float)
            for dt in idx:
                r = disc_rate.get(dt, np.nan)
                if np.isnan(r): continue
                sb = {}
                if s1 is not None and dt in s1.index and not np.isnan(s1[dt]): sb["1Y"] = s1[dt]
                if s5 is not None and dt in s5.index and not np.isnan(s5[dt]): sb["5Y"] = s5[dt]
                if ten not in sb: continue              # serve lo spread del tenor target
                try: res[dt] = risky_annuity(dt, sb, ten, r)
                except Exception: res[dt] = np.nan
            out[f"{iss}_{ten}"] = res

    rpv = pd.DataFrame(out).sort_index()
    rpv.index.name = "Date"

    # ---------- 4. validazione vs BBG SW_CNV_RISK (overlap) ----------
    print("VALIDAZIONE ISDA vs BBG SW_CNV_RISK (mediana errore % sull'overlap):")
    for (iss, ten), bser in rpv_bbg.items():
        col_ = f"{iss}_{ten}"
        if col_ in rpv:
            b = bser.dropna(); m = rpv[col_].reindex(b.index).dropna()
            common = b.index.intersection(m.index)
            if len(common):
                err = ((m[common]-b[common]).abs()/b[common]*100)
                print(f"  {col_:18s} n={len(common):5d}  err_mediano={err.median():.2f}%  (mio={m[common].median():.3f} bbg={b[common].median():.3f})")

    # ---------- 5. salva ----------
    rpv.to_excel("/mnt/user-data/outputs/cds_rpv01.xlsx")
    print("\nRPV01 panel:", rpv.shape, "-> /mnt/user-data/outputs/cds_rpv01.xlsx")
    print("colonne:", list(rpv.columns))
    print("copertura:", rpv.index.min().date(), "->", rpv.index.max().date())
