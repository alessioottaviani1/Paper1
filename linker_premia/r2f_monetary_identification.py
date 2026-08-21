"""r2f - IL TEST CHE CHIUDE LA STORIA ECONOMICA: l'amplificazione sovrana IT-DE del
breakeven sopravvive AL NETTO della politica monetaria BCE?

Da r2c/r2d sappiamo:
  - T2: il differenziale (lambda_IT - lambda_DE) reagisce alle sorprese d'inflazione,
    IT-DE +1.32, sopravvive a tutto. E' il fatto piu' robusto del progetto.
  - T4: il canale fiscale PURO e' falsificato (la Germania reagisce quando dovrebbe
    essere muta) -> la lettura corretta e' "floor monetario comune + amplificazione
    sovrana": la BCE muove TUTTI i breakeven (anche DE), e l'Italia reagisce di PIU'.

MA l'amplificazione IT-DE sopravvive NON e' ancora IDENTIFICATA. Puo' essere:
  (a) genuina amplificazione del rischio sovrano all'inflazione, oppure
  (b) semplicemente la BCE che, rispondendo all'inflazione, muove gli spread sovrani
      (politica monetaria, non rischio-paese autonomo).

QUESTO TEST separa (a) da (b). Controlla la reazione di IT-DE alle sorprese d'inflazione
per le SORPRESE DI POLITICA MONETARIA BCE (Target, Forward Guidance, QE dall'EA-MPD,
costruite in ecb_factors.py) E per l'effetto diretto della BCE sullo spread sovrano
(IT10Y-DE10Y nella finestra della riunione).

  Se il coefficiente di IT-DE sulla sorpresa d'inflazione SOPRAVVIVE netto dei fattori
  monetari -> l'amplificazione e' rischio-paese genuino, NON politica monetaria.
  STORIA ECONOMICA CHIUSA: il breakeven sovrano reagisce all'inflazione oltre la BCE.

  Se SPARISCE -> era la BCE che muoveva gli spread. La storia si chiude diversamente.

NOTA identificazione: Altavilla et al. (WP 2281) trovano che il primo stadio della
trasmissione monetaria "non e' diverso tra i grandi paesi euro" sui rendimenti NOMINALI.
Qui testiamo i BREAKEVEN (non i nominali): se l'eterogeneita' IT-DE sopravvive al
controllo monetario, e' un fenomeno oltre la trasmissione standard -> contributo.

Richiede: ecb_factors_monthly.parquet (da ecb_factors.py) + CPI nazionali (da r0).
Output: an_monetary_identification.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np
import pandas as pd
import rp
from config import CACHE

MATS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)
REL_DAY = {"IT": 17, "FR": 15, "DE": 15}
NAT_KEY = {"IT": "IT_NAT", "FR": "FR_NAT", "DE": "DE_NAT"}
CRISIS = ("2011-01-01", "2012-12-31")


# ------------------------------------------------------------------ lambda differenziale (come r2c T2)
def _release_grid(mkt, months):
    f = CACHE / f"cpi_release_dates_{mkt}.csv"
    if f.exists():
        rel = pd.to_datetime(pd.read_csv(f).iloc[:, 0]).sort_values()
        out = {}
        for m in months:
            nxt = rel[(rel > m) & (rel <= m + pd.offsets.MonthEnd(2))]
            if len(nxt):
                out[m] = nxt.iloc[0]
        return pd.Series(out)
    d = REL_DAY[mkt]
    return pd.Series({m: (m + pd.offsets.MonthBegin(1) + pd.Timedelta(days=d - 1)
                          + pd.offsets.BusinessDay(0)) for m in months})


def _sample_at(daily, grid):
    daily = daily.sort_index()
    return pd.DataFrame({m: daily.asof(g) for m, g in grid.items()
                         if g >= daily.index.min()}).T.dropna(how="all")


def _lambda(mkt):
    bei_d, isr_d = rp.bei_euro(mkt, MATS), rp.isr(mkt, MATS)
    months = pd.date_range(bei_d.index.min().normalize().replace(day=1),
                           bei_d.index.max(), freq="ME")
    grid = _release_grid(mkt, months)
    bei, s = _sample_at(bei_d, grid), _sample_at(isr_d, grid)
    idx = bei.index.intersection(s.index)
    return (s.loc[idx] - bei.loc[idx])


def _lambda_diff():
    """lambda_IT - lambda_DE (l'amplificazione sovrana), per scadenza, EOM."""
    lam_it, lam_de = _lambda("IT"), _lambda("DE")
    idx = lam_it.index.intersection(lam_de.index)
    cols = [c for c in lam_it.columns if c in lam_de.columns]
    return lam_it.loc[idx, cols] - lam_de.loc[idx, cols]


def _cpi_national(mkt):
    path = CACHE / f"an_cpi_{NAT_KEY[mkt]}.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path).iloc[:, 0]
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _surprise(idx_series):
    pi = rp.yoy(idx_series).dropna()
    return (pi - pi.rolling(120, min_periods=120).mean()).dropna()


# ------------------------------------------------------------------ fattori BCE mensili
def _ecb_monthly():
    f = CACHE / "ecb_factors_monthly.parquet"
    if not f.exists():
        return None
    M = pd.read_parquet(f)
    M.index = pd.to_datetime(M.index) + pd.offsets.MonthEnd(0)
    return M[~M.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------ regressione multivariata NW
def _reg_multi(y, Xdf, lags=6):
    """y (bp) su Xdf (piu' regressori, gia' allineati). Ritorna dict nome->(beta,t)."""
    d = pd.concat([y.rename("y"), Xdf], axis=1).dropna()
    if len(d) < 30:
        return None, len(d)
    Xcols = list(Xdf.columns)
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in Xcols])
    b, e, _, r2, _ = rp.ols(d["y"].values, X[:, 1:])   # rp.ols aggiunge la costante
    t = rp.nw_t(e, X, b, lags)
    names = ["const"] + Xcols
    return {n: (b[i], t[i]) for i, n in enumerate(names)}, len(d), r2


def run_tenor(diff, surp, ecb, tenor, label):
    """Un tenore: IT-DE(tenor) ~ InfS, poi + fattori BCE, affiancati."""
    y = (diff[tenor] * 100).dropna()   # bp
    rows = []
    # (1) baseline: solo sorpresa d'inflazione (replica T2 a questo tenore)
    r1 = _reg_multi(y, surp.rename("infS").to_frame().reindex(y.index))
    # (2) + fattori monetari BCE (target, fg, qe)
    X2 = pd.concat([surp.rename("infS"),
                    ecb["target"].rename("mp_target"),
                    ecb["fg"].rename("mp_fg"),
                    ecb["qe"].rename("mp_qe")], axis=1).reindex(y.index)
    r2_ = _reg_multi(y, X2)
    # (3) + effetto diretto BCE sullo spread sovrano
    X3 = pd.concat([X2, ecb["sov_spread_shock"].rename("mp_sovshock")], axis=1).reindex(y.index)
    r3 = _reg_multi(y, X3)

    def _g(res, key):
        if res is None or res[0] is None:
            return (np.nan, np.nan, 0)
        d = res[0]; n = res[1]
        return (d.get(key, (np.nan, np.nan))[0], d.get(key, (np.nan, np.nan))[1], n)

    b1, t1, n1 = _g(r1, "infS")
    b2, t2, n2 = _g(r2_, "infS")
    b3, t3, n3 = _g(r3, "infS")
    print(f"\n  [{label}] tenore {tenor}y")
    print(f"    (1) IT-DE ~ InfS                       infS = {b1:+.2f} [{t1:+.1f}]   n={n1}")
    print(f"    (2) + BCE (target,fg,qe)               infS = {b2:+.2f} [{t2:+.1f}]   n={n2}")
    print(f"    (3) + shock diretto spread sovrano     infS = {b3:+.2f} [{t3:+.1f}]   n={n3}")
    # verdetto: quanto resta di infS passando da (1) a (3)
    keep = b3 / b1 if b1 else np.nan
    surv = (abs(t3) >= 1.96 and np.sign(b3) == np.sign(b1))
    print(f"    >>> InfS resta il {keep*100:.0f}% netto della BCE, "
          f"{'SIGNIFICATIVO' if surv else 'non significativo'}")
    for spec, res in (("baseline", r1), ("+BCE", r2_), ("+sovshock", r3)):
        if res and res[0]:
            for k, (bb, tt) in res[0].items():
                rows.append([label, tenor, spec, k, bb, tt, res[1]])
    return rows, (b1, b3, surv)


# ------------------------------------------------------------------ test single-market (falsificazione DE)
def run_single_market(mkt, ecb, tenor):
    """lambda_mkt(tenor) ~ sorpresa NAZIONALE del paese, netto BCE.
    Per IT: deve REGGERE (l'Italia reagisce alla sua inflazione).
    Per DE: deve essere ~0 (la Germania non ha rischio sovrano -> canale muto).
    Ritorna (b_baseline, b_netBCE, t_netBCE, n)."""
    lam = _lambda(mkt)
    if tenor not in lam.columns:
        return None
    y = (lam[tenor] * 100).dropna()   # bp
    nat = _cpi_national(mkt)
    if nat is None:
        return None
    surp = _surprise(nat).reindex(y.index)
    # baseline
    r1 = _reg_multi(y, surp.rename("infS").to_frame())
    # netto BCE (target, fg, qe + shock diretto)
    X = pd.concat([surp.rename("infS"),
                   ecb["target"].rename("mp_target"),
                   ecb["fg"].rename("mp_fg"),
                   ecb["qe"].rename("mp_qe"),
                   ecb["sov_spread_shock"].rename("mp_sovshock")], axis=1).reindex(y.index)
    r3 = _reg_multi(y, X)
    def _g(res, key):
        if res is None or res[0] is None:
            return (np.nan, np.nan, 0)
        return (res[0].get(key, (np.nan, np.nan))[0], res[0].get(key, (np.nan, np.nan))[1], res[1])
    b1, _, _ = _g(r1, "infS")
    b3, t3, n3 = _g(r3, "infS")
    return (b1, b3, t3, n3)


def diagnose_bce_inflation(surp, ecb, diff, tenor=10):
    """Perche' il coefficiente su InfS SALE aggiungendo i controlli BCE? Ipotesi: la BCE
    reagisce all'inflazione smorzando il breakeven differenziale -> controllando per la BCE
    si libera l'effetto pieno. Verifica: (a) corr(InfS, fattori BCE); (b) segno dei
    coefficienti BCE nella regressione del differenziale (devono smorzare)."""
    print("\n" + "=" * 78)
    print("DIAGNOSTICA -- perche' il coefficiente su InfS SALE con i controlli BCE?")
    print("=" * 78)
    # (a) correlazione tra sorpresa d'inflazione e fattori BCE
    print("  (a) corr(sorpresa inflazione, fattori BCE):")
    for f in ["target", "fg", "qe", "sov_spread_shock"]:
        d = pd.concat([surp.rename("infS"), ecb[f].rename(f)], axis=1).dropna()
        if len(d) > 20:
            c = d.corr().iloc[0, 1]
            print(f"      corr(InfS, {f:16s}) = {c:+.2f}")
    # (b) segno dei coefficienti BCE nel differenziale a 10y
    if tenor in diff.columns:
        y = (diff[tenor] * 100).dropna()
        X = pd.concat([surp.rename("infS"),
                       ecb["target"].rename("mp_target"),
                       ecb["fg"].rename("mp_fg"),
                       ecb["qe"].rename("mp_qe"),
                       ecb["sov_spread_shock"].rename("mp_sovshock")], axis=1).reindex(y.index)
        res = _reg_multi(y, X)
        if res and res[0]:
            print(f"\n  (b) coefficienti nella regressione IT-DE({tenor}y) completa:")
            for k, (b, t) in res[0].items():
                if k != "const":
                    print(f"      {k:16s} = {b:+.3f} [{t:+.1f}]")
            print("      [se i fattori BCE hanno segno opposto a InfS o smorzano il breakeven,")
            print("       spiega perche' controllarli libera l'effetto pieno dell'inflazione]")


if __name__ == "__main__":
    ecb = _ecb_monthly()
    if ecb is None:
        raise SystemExit("[!] ecb_factors_monthly.parquet assente: lanciare prima ecb_factors.py")
    print(f"[ecb] fattori mensili: {len(ecb)} mesi, {ecb.index.min().date()} -> {ecb.index.max().date()}")

    diff = _lambda_diff()
    print(f"[lambda] differenziale IT-DE: {len(diff)} mesi, tenori {list(diff.columns)}")

    # sorpresa d'inflazione: nazionale italiana se disponibile, altrimenti euro aggregata
    nat_it = _cpi_national("IT")
    if nat_it is not None:
        surp = _surprise(nat_it)
        print("[surp] sorpresa d'inflazione = CPI NAZIONALE italiano (r2d-style)")
    else:
        surp = rp.surprise_maffei("IT")
        print("[surp] sorpresa d'inflazione = HICP euro aggregata (nazionale assente)")

    surp = surp.reindex(diff.index).dropna()

    # allinea i fattori BCE all'indice EOM del differenziale
    ecb_al = ecb.reindex(diff.index)

    print("\n" + "=" * 78)
    print("TEST r2f -- l'amplificazione IT-DE sopravvive al netto della politica monetaria BCE?")
    print("=" * 78)
    print("  (1) baseline replica T2 (IT-DE ~ sorpresa d'inflazione)")
    print("  (2) aggiunge i 3 fattori BCE (Target, Forward Guidance, QE)")
    print("  (3) aggiunge lo shock diretto BCE sullo spread sovrano (IT10Y-DE10Y in finestra)")

    all_rows = []
    verdicts = []
    for tenor in (5, 10):
        if tenor in diff.columns:
            rows, v = run_tenor(diff, surp, ecb_al, tenor, "IT-DE")
            all_rows += rows
            verdicts.append((tenor, v))

    # media sul tratto centrale (piu' robusta del singolo tenore)
    print("\n" + "=" * 78)
    print("VERDETTO -- la storia economica dei linker e' chiusa?")
    print("=" * 78)
    for tenor, (b1, b3, surv) in verdicts:
        print(f"  {tenor}y: InfS baseline {b1:+.2f} -> netto BCE {b3:+.2f}  "
              f"({'SOPRAVVIVE' if surv else 'sparisce'})")
    any_surv = any(v[1][2] for v in verdicts)
    print()
    if any_surv:
        print("  >>> l'amplificazione IT-DE SOPRAVVIVE al netto della politica monetaria BCE.")
        print("      Il breakeven sovrano reagisce all'inflazione OLTRE la trasmissione")
        print("      monetaria standard (che Altavilla trova omogenea tra paesi sui nominali).")
        print("      -> RISCHIO-PAESE genuino isolato. STORIA ECONOMICA CHIUSA e pubblicabile:")
        print("         'sovereign amplification of inflation risk, net of ECB policy'.")
    else:
        print("  >>> l'amplificazione IT-DE SPARISCE al netto della BCE: era la politica")
        print("      monetaria a muovere gli spread sovrani. La storia va riformulata")
        print("      (il canale e' monetario, non rischio-paese autonomo).")

    # ---------- TEST DI FALSIFICAZIONE: IT reagisce, DE deve essere MUTA (netto BCE)
    print("\n" + "=" * 78)
    print("TEST DI FALSIFICAZIONE -- IT reagisce alla sua inflazione, DE no? (netto BCE)")
    print("=" * 78)
    print("  la storia 'rischio-paese' richiede: IT reagisce alla sorpresa ITALIANA,")
    print("  ma DE NON reagisce alla sorpresa TEDESCA (la Germania non ha rischio sovrano).")
    print("  se DE fosse anch'essa forte -> non e' rischio-paese ma un fenomeno comune.")
    for tenor in (5, 10):
        print(f"\n  --- tenore {tenor}y ---")
        for mkt in ("IT", "DE"):
            r = run_single_market(mkt, ecb_al, tenor)
            if r is None:
                print(f"    {mkt}: dati insufficienti (CPI nazionale o tenore assente)")
                continue
            b1, b3, t3, n3 = r
            role = "deve REGGERE" if mkt == "IT" else "deve essere ~0 (controllo)"
            flag = ""
            if mkt == "DE":
                flag = "  <-- MUTO, ok" if abs(t3) < 1.96 else "  <-- REAGISCE (attenzione!)"
            elif mkt == "IT":
                flag = "  <-- regge, ok" if abs(t3) >= 1.96 and b3 > 0 else "  <-- debole"
            print(f"    lambda_{mkt} ~ InfS_{mkt} netto BCE = {b3:+.2f} [{t3:+.1f}]  n={n3}  "
                  f"({role}){flag}")
    print("\n  >>> se IT regge e DE e' muto -> canale rischio-paese BLINDATO e simmetrico:")
    print("      l'Italia reagisce alla sua inflazione oltre la BCE, la Germania no.")

    # ---------- DIAGNOSTICA: perche' il coefficiente sale con i controlli
    diagnose_bce_inflation(surp, ecb_al, diff, tenor=10)

    if all_rows:
        pd.DataFrame(all_rows, columns=["dep", "tenor", "spec", "regressor", "beta", "t", "n"]
                     ).to_csv(CACHE / "an_monetary_identification.csv", index=False)
    print("\nsalvato: an_monetary_identification.csv")
