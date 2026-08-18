"""17 - TEST B: chi detiene i CCT, e la clientela di habitat MISURATA per contrasto.

FATTO VERIFICATO (conti che quadrano, consistenze 2026-Q1, mld):
  banche 88.5 | estero ~12.5 | BCE 7.2 | famiglie 3.7 | assicur 2.3 | NFC 2.0 | fondi 0.3
  totale in circolazione ~116. Le banche detengono il 76% dei CCT.

SPECIFICITA' (quota CCT sul portafoglio titoli di Stato di ogni settore):
  BANCHE 22.4% | NFC 3.2% | famiglie 1.1% | assicur 1.0%.
  Le banche sono l'UNICO settore che SOVRAPPESA i CCT -> sono la clientela di habitat.
  Storicamente hanno sempre avuto la quota CCT piu' alta (47% nel 2008 -> 22% oggi).
  Il retail (famiglie) NON ha mai sovrappesato i CCT piu' delle banche -> non era l'habitat.

INTERPRETAZIONE: le banche sono un habitat CON capacita' di arbitraggio -- tengono i CCT
strutturalmente (immunizzazione passivo a vista) E sfruttano i disallineamenti (comprano
quando i CCT sono a sconto: da qui il segno +positivo di d_holdings sulla base). E' la
Teoria 1 (Vayanos-Vila habitat + Gromb-Vayanos vincolo) nello STESSO agente.

DATI (Bankit/): CCT per settore (stock trim) + "altri titoli medio/lungo amm.centrale"
(=BTP+altri, per settore) -> permette la QUOTA CCT per settore. Banche anche mensile.

TEST:
  B.0 -- SPECIFICITA': quale settore sovrappesa i CCT? (identifica l'habitat per contrasto)
  B.1 -- la base risponde alle detenzioni bancarie di CCT? (mensile, potenza)
  B.2 -- IDENTIFICAZIONE: al netto di stress sovrano e QE (banca centrale)?
  B.3 -- la QUOTA CCT bancaria (specificita' nel tempo) predice la base?

Richiede: basis_daily.csv (07), mechanism_panel.csv (10), file Bankit/.
Output: results/17_holdings_test.txt
"""
import numpy as np, pandas as pd, re, glob
from config import *
from utils import save_txt

ONLY_CCTEU = True

# match per DESCRIZIONE. CCT per settore + "altri titoli medio/lungo" (BTP-ish) per settore.
SEC_KEYS = {
    "banche":        ["istituzioni finanziarie monetarie", "government securities - CCT"],
    "assicurazioni": ["assicuraz"],
    "fondi_comuni":  ["fondi comuni"],
    "altri_interm":  ["altri intermediari"],
    "nfc":           ["societ", "non finanz"],
    "famiglie":      ["households", "famiglie"],
    "bce":           ["central bank"],
}


def _read(f):
    raw = open(f, encoding="utf-8", errors="replace").read().splitlines()
    if not raw:
        return "", None
    sep = ";" if ";" in raw[0] else ","
    desc = raw[0].split(sep, 1)[1].strip('"') if sep in raw[0] else raw[0]
    rows = []
    for l in raw[1:]:
        p = l.split(sep)
        if len(p) >= 2 and re.match(r'"?\d{4}-\d{2}-\d{2}', p[0]):
            v = p[1].strip('"').replace(",", "")
            try:
                rows.append((pd.to_datetime(p[0].strip('"')),
                             float(v) if v not in ("", "..", "-") else np.nan))
            except Exception:
                pass
    return desc, (pd.DataFrame(rows, columns=["date", "value"]).set_index("date")["value"].sort_index() if rows else None)


def _index():
    roots = [ROOT, ROOT.parent, Path.cwd()]
    files = []
    for r in roots:
        files += glob.glob(str(r / "**" / "REPORT_*.csv"), recursive=True)
    cct, alt, bankm = {}, {}, {}
    for f in sorted(set(files)):
        d, s = _read(f)
        if s is None:
            continue
        dl = d.lower()
        # settore
        sec = None
        if "assicuraz" in dl: sec = "assicurazioni"
        elif "fondi comuni" in dl: sec = "fondi_comuni"
        elif "altri intermediari" in dl: sec = "altri_interm"
        elif "istituzioni finanziarie monetarie" in dl: sec = "banche"
        elif ("societ" in dl and "non finanz" in dl): sec = "nfc"
        elif "households" in dl or ("famiglie" in dl and "held" not in dl): sec = "famiglie"
        elif "central bank" in dl: sec = "bce"
        # banche mensile da bilanci (CCT)
        if "Banks:" in d and "CCT" in d:
            bankm["cct"] = s
        if "Banks:" in d and "BTP" in d:
            bankm["btp"] = s
        if sec is None:
            continue
        if "CCT" in d and ("detenuti" in dl or "sottoscritti" in dl or "held by" in dl):
            cct[sec] = s
        elif "altri titoli" in dl or "medio/lungo" in dl:
            alt[sec] = s
    return cct, alt, bankm


if __name__ == "__main__":
    print("== 17 test B: chi detiene i CCT + habitat per contrasto ==")
    L = []; P = L.append
    P("=== 17 TEST B -- CHI DETIENE I CCT + CLIENTELA DI HABITAT PER CONTRASTO ===")

    cct, alt, bankm = _index()
    P(f"settori con detenzioni CCT: {sorted(cct.keys())}")
    P(f"settori con 'altri titoli' (BTP-ish): {sorted(alt.keys())}")
    P(f"banche mensili (bilanci): CCT={'si' if 'cct' in bankm else 'no'}, BTP={'si' if 'btp' in bankm else 'no'}")
    if "banche" not in cct:
        P("[!] detenzioni CCT per settore non trovate. Copiare i REPORT_*.csv (Bankit) in THESIS.")
        print("\n".join(L)); save_txt("17_holdings_test.txt", L); raise SystemExit

    # ===================================================== B.0 SPECIFICITA'
    P("\n" + "=" * 80)
    P("B.0 -- SPECIFICITA': quale settore SOVRAPPESA i CCT? (habitat per contrasto)")
    P("  quota CCT = CCT / (CCT + altri titoli di Stato) nel portafoglio del settore.")
    P("  quota alta = quel settore concentra domanda sui CCT = clientela di habitat.")
    P("=" * 80)
    d0 = max(s.dropna().index.max() for s in cct.values())
    P(f"  {'settore':16s} {'CCT(mld)':>9s} {'BTP-ish':>9s} {'quota CCT':>10s}")
    shares = {}
    for sec in sorted(set(cct) & set(alt)):
        c = cct[sec][cct[sec].index <= d0].dropna()
        a = alt[sec][alt[sec].index <= d0].dropna()
        if len(c) and len(a):
            cv, av = c.iloc[-1] / 1000, a.iloc[-1] / 1000
            sh = cv / (cv + av) * 100
            shares[sec] = sh
            P(f"  {sec:16s} {cv:9.1f} {av:9.1f} {sh:9.1f}%")
    if shares:
        top = max(shares, key=shares.get)
        P(f"  >>> settore che sovrappesa i CCT: {top.upper()} ({shares[top]:.1f}%) = habitat floater")

    # ===================================================== base
    B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
    if ONLY_CCTEU:
        B = B[B.regime == "CCTeu"].copy()
    tau_c = "tau_cct" if "tau_cct" in B.columns else "tau"
    B["ym"] = B.date.dt.to_period("M")
    sov = None
    try:
        M = pd.read_csv(PROC/"mechanism_panel.csv", parse_dates=["date"])
        M["ym"] = M.date.dt.to_period("M")
        sov = M.groupby("ym")["sov_swap"].mean()
    except Exception:
        pass

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        P("[statsmodels assente]"); print("\n".join(L)); save_txt("17_holdings_test.txt", L); raise SystemExit

    def run(d, f, keys, lab):
        try:
            r = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["CCT_ISIN"]})
            P(f"  {lab:46s} " + "  ".join(
                f"{k}={r.params.get(k,np.nan):+.4f}[{r.tvalues.get(k,np.nan):+.2f}]" for k in keys)
              + f"   n {int(r.nobs):,}")
            return {k: (r.params.get(k, np.nan), r.tvalues.get(k, np.nan)) for k in keys}
        except Exception as e:
            P(f"  {lab:46s} fallita ({str(e)[:42]})"); return {}

    # ===================================================== B.1 base ~ detenzioni bancarie CCT (mensile)
    P("\n" + "=" * 80)
    P("B.1 -- la base risponde alle detenzioni bancarie di CCT? (mensile, potenza)")
    P("  segno + = banche comprano quando i CCT sono a sconto (habitat con arbitraggio);")
    P("  segno - = banche sostengono il prezzo (habitat puro). Il segno E' informativo.")
    P("=" * 80)
    if "cct" in bankm:
        d_bank = (np.log(bankm["cct"].replace(0, np.nan)).diff() * 100)
        d_bank.index = d_bank.index.to_period("M")
        Bm = (B.groupby(["CCT_ISIN", "ym"])
                .agg(basis_y=("basis3_y", "mean"), tau=(tau_c, "mean")).reset_index())
        Bm["d_bank_cct"] = Bm.ym.map(d_bank.groupby(level=0).last().to_dict())
        if sov is not None:
            Bm["sov_swap"] = Bm.ym.map(sov.to_dict())
        D = Bm.dropna(subset=["basis_y", "d_bank_cct"])
        P(f"  panel mensile: {len(D):,} oss. | {D.CCT_ISIN.nunique()} CCT")
        run(D, "basis_y ~ d_bank_cct + tau", ["d_bank_cct"], "  (1) grezza")
        run(D, "basis_y ~ d_bank_cct + tau + C(CCT_ISIN)", ["d_bank_cct"], "  (2) + eff. fissi CCT")

        # ================================================= B.2 identificazione
        P("\n" + "=" * 80)
        P("B.2 -- IDENTIFICAZIONE: al netto di stress sovrano e QE (banca centrale)?")
        P("=" * 80)
        if "bce" in cct:
            d_qe = cct["bce"].diff(); d_qe.index = d_qe.index.to_period("M")
            Bm["d_qe"] = Bm.ym.map(d_qe.groupby(level=0).last().to_dict())
        have_qe = "d_qe" in Bm.columns and Bm.d_qe.notna().sum() > 50
        if "sov_swap" in Bm.columns and Bm.sov_swap.notna().sum() > 50:
            cols = ["d_bank_cct", "sov_swap"] + (["d_qe"] if have_qe else [])
            dd = Bm.dropna(subset=["basis_y"] + cols).copy()
            P(f"  corr(d banche CCT, stress) = {dd.d_bank_cct.corr(dd.sov_swap):+.2f}")
            run(dd, "basis_y ~ " + " + ".join(cols) + " + tau", cols,
                "  (3) banche + stress" + (" + QE" if have_qe else ""))
            X = dd[["sov_swap"] + (["d_qe"] if have_qe else [])].values
            X = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X, dd.d_bank_cct.values, rcond=None)[0]
            dd["orth"] = dd.d_bank_cct.values - X @ beta
            run(dd, "basis_y ~ orth + tau + C(CCT_ISIN)", ["orth"],
                "  (4) residuo banche al netto stress" + ("+QE" if have_qe else ""))
            P("  [se il residuo predice la base, il canale bancario e' distinto da stress e QE]")

    # ===================================================== B.3 quota CCT bancaria nel tempo
    P("\n" + "=" * 80)
    P("B.3 -- la QUOTA CCT bancaria (specificita' nel tempo) predice la base?")
    P("  quota alta = banche concentrate sui CCT (habitat forte) -> base sostenuta.")
    P("=" * 80)
    if "banche" in cct and "banche" in alt:
        cb = cct["banche"]; ab = alt["banche"]
        j = pd.concat([cb, ab], axis=1, keys=["cct", "alt"]).dropna()
        j["share"] = j.cct / (j.cct + j.alt) * 100
        j["ym"] = j.index.to_period("Q")
        # base trimestrale
        B["q"] = B.date.dt.to_period("Q")
        Bq = B.groupby(["CCT_ISIN", "q"]).agg(basis_y=("basis3_y", "mean"),
                                              tau=(tau_c, "mean")).reset_index()
        sh_q = j.set_index(j.index.to_period("Q"))["share"]
        Bq["bank_share"] = Bq.q.map(sh_q.to_dict())
        if sov is not None:
            sov_q = sov.groupby(sov.index.astype("period[Q]")).mean() if hasattr(sov.index, 'astype') else None
        Dq = Bq.dropna(subset=["basis_y", "bank_share"])
        P(f"  panel trimestrale: {len(Dq):,} oss. | quota banche {j.share.min():.0f}%-{j.share.max():.0f}%")
        run(Dq, "basis_y ~ bank_share + tau + C(CCT_ISIN)", ["bank_share"],
            "  base ~ quota CCT bancaria")
        P("  [segno - atteso: quota bancaria alta -> CCT piu' sostenuti -> base meno positiva.")
        P("   E' la specificita' dell'habitat nel tempo, il test piu' vicino alla teoria]")

    P("\n  NOTA: conti verificati (banche 76% dei CCT, unico settore che li sovrappesa).")
    P("   La clientela di habitat sono le BANCHE (non il retail, uscito). Habitat con")
    P("   arbitraggio: tengono i CCT strutturalmente e sfruttano i disallineamenti.")
    print("\n".join(L)); save_txt("17_holdings_test.txt", L)
