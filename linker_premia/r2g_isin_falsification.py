"""r2g - IL TEST DI FALSIFICAZIONE, GOLD STANDARD: breakeven a livello ISIN, non su curva.

Perche' ISIN e non curva. Il test r2f usa il breakeven su CURVA (nominale - reale NSS).
La curva reale tedesca e' fittata su POCHI Bund€i (5-6 titoli): a certi nodi (es. 5y) puo'
essere mal vincolata o estrapolata, e il rumore di fitting puo' correlarsi spuriamente con
l'inflazione -> un falso "la Germania reagisce". A livello ISIN il breakeven e' calcolato
dai flussi DEL SINGOLO TITOLO (total_ytm - IRR reale, da breakeven_{mkt}.parquet): nessuna
curva, nessuna estrapolazione, nessun rumore di fit propagato tra scadenze. E' il dato
pulito per decidere se la Germania e' davvero muta.

Il costo: la scadenza SCORRE nel tempo (un bond 2033 e' a 10y nel 2023, a 8y nel 2025). Si
gestisce controllando per la MATURITA' RESIDUA nella regressione (approccio pooled).

STRUTTURA:
  1. DIAGNOSTICA densita' curva DE: quanti Bund€i esistono per periodo e come si distribuiscono
     in scadenza -> dimostra SE e QUANDO la curva a 5y era mal vincolata (il sospetto di r2f).
  2. TEST ISIN pooled: breakeven_bond ~ sorpresa_nazionale + maturita' residua, netto BCE.
     Per IT deve reggere, per DE deve essere muto. Confronto diretto con r2f (su curva).
  3. BENCHMARK US e UK: curve reali FITTE (decine di TIPS/gilt) -> breakeven ben misurato.
     Controllo di sanita': come si comporta un breakeven pulito su sovrani sicuri?

Richiede: breakeven_{IT,DE}.parquet (pipeline con CON_BREAKEVEN=True) + ref_linker (maturita')
+ ecb_factors_monthly + CPI nazionali. US/UK: curve rp.bei_us/bei_uk (gia' disponibili).
Output: an_isin_falsification.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np
import pandas as pd
import rp
from config import CACHE
try:
    import bbg
except Exception:
    bbg = None

REL_DAY = {"IT": 17, "FR": 15, "DE": 15, "US": 12, "UK": 15}
NAT_KEY = {"IT": "IT_NAT", "FR": "FR_NAT", "DE": "DE_NAT"}


# ------------------------------------------------------------------ util comuni
def _surprise_from(idx_series):
    pi = rp.yoy(idx_series).dropna()
    return (pi - pi.rolling(120, min_periods=120).mean()).dropna()


def _cpi_national(mkt):
    path = CACHE / f"an_cpi_{NAT_KEY.get(mkt, mkt)}.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path).iloc[:, 0]
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _ecb_monthly():
    f = CACHE / "ecb_factors_monthly.parquet"
    if not f.exists():
        return None
    M = pd.read_parquet(f)
    M.index = pd.to_datetime(M.index) + pd.offsets.MonthEnd(0)
    return M[~M.index.duplicated(keep="last")].sort_index()


def _ref_maturities():
    """MATURITY per ISIN dei linker (da ref_linker)."""
    if bbg is None:
        return None
    try:
        ref = bbg.load("ref_linker")
    except Exception:
        return None
    mcol = next((c for c in ("MATURITY", "MTY", "maturity") if c in ref.columns), None)
    kcol = next((c for c in ("MKT", "MARKET", "mkt") if c in ref.columns), None)
    if mcol is None:
        return None
    out = pd.DataFrame({"maturity": pd.to_datetime(ref[mcol], errors="coerce")}, index=ref.index)
    if kcol:
        out["mkt"] = ref[kcol]
    return out


def _load_breakeven_isin(mkt):
    """breakeven_{mkt}.parquet: pannello date x ISIN, breakeven per singolo titolo (bp o %)."""
    f = CACHE / f"breakeven_{mkt}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _ols_nw(y, Xdf, lags=6):
    d = pd.concat([y.rename("y"), Xdf], axis=1).dropna()
    if len(d) < 40:
        return None, len(d)
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in Xdf.columns])
    b, e, _, r2, _ = rp.ols(d["y"].values, X[:, 1:])
    t = rp.nw_t(e, X, b, lags)
    names = ["const"] + list(Xdf.columns)
    return {n: (b[i], t[i]) for i, n in enumerate(names)}, len(d)


# ================================================================== 1) diagnostica densita' curva DE
def diagnose_de_density():
    print("\n" + "=" * 78)
    print("1) DIAGNOSTICA -- la curva reale DE e' abbastanza fitta a 5y e 10y?")
    print("=" * 78)
    mat = _ref_maturities()
    if mat is None:
        print("  [!] ref_linker non disponibile: impossibile contare i Bund€i. Salto.")
        return
    de = mat[mat.get("mkt", pd.Series(index=mat.index)).astype(str).str.upper().str.contains("DE", na=False)] \
        if "mkt" in mat.columns else mat
    if de.empty:
        # fallback: prova a filtrare per ISIN tedeschi (DE...) 
        de = mat[mat.index.astype(str).str.startswith("DE")]
    if de.empty:
        print("  [!] nessun linker DE identificato in ref_linker. Salto.")
        return
    mats = de["maturity"].dropna().sort_values()
    print(f"  Bund€i totali in anagrafica: {len(mats)}")
    print(f"  scadenze: {', '.join(m.strftime('%Y') for m in mats)}")
    # per alcune date campione, quanti bond vivi e come bracketano 5y/10y
    print("\n  copertura ai nodi nel tempo (bond vivi che bracketano il nodo +-2y):")
    for yr in (2013, 2016, 2019, 2022, 2025):
        asof = pd.Timestamp(f"{yr}-06-30")
        alive = mats[(mats > asof)]                       # non ancora scaduti
        ttm = (alive - asof).dt.days / 365.25             # anni a scadenza
        n5 = ((ttm >= 3) & (ttm <= 7)).sum()              # bracketano 5y
        n10 = ((ttm >= 8) & (ttm <= 12)).sum()            # bracketano 10y
        f5 = "  <-- 5y POCO vincolato" if n5 < 2 else ""
        f10 = "  <-- 10y poco vincolato" if n10 < 2 else ""
        print(f"    {yr}: {len(alive)} Bund€i vivi | attorno 5y: {n5}{f5} | attorno 10y: {n10}{f10}")
    print("\n  [se a 5y ci sono <2 bond che bracketano in molti periodi, il breakeven DE 5y")
    print("   su CURVA e' mal vincolato -> il t+5.1 di r2f e' probabilmente artefatto di fit,")
    print("   e il test ISIN (che usa i titoli veri) e' quello da credere]")


# ================================================================== 2) test ISIN pooled
def test_isin_pooled(mkt, ecb, surp):
    """breakeven_bond(t,isin) ~ surp + maturita' residua, netto BCE. Pooled su tutti i titoli."""
    bkev = _load_breakeven_isin(mkt)
    mat = _ref_maturities()
    if bkev is None:
        print(f"  {mkt}: breakeven_{mkt}.parquet ASSENTE (rigira 04 con CON_BREAKEVEN=True)")
        return None
    if mat is None:
        print(f"  {mkt}: ref_linker assente (serve la maturita' residua)")
        return None
    # long format: (date, isin, bkev)
    long = bkev.stack().rename("bkev").reset_index()
    long.columns = ["date", "isin", "bkev"]
    long["date"] = pd.to_datetime(long["date"])
    long = long.merge(mat[["maturity"]], left_on="isin", right_index=True, how="left")
    long["ttm"] = (long["maturity"] - long["date"]).dt.days / 365.25
    long = long[(long["ttm"] > 1) & (long["ttm"] < 30)].dropna(subset=["bkev", "ttm"])
    # EOM per il merge con surp/ecb mensili
    long["eom"] = long["date"] + pd.offsets.MonthEnd(0)
    long = long.merge(surp.rename("infS"), left_on="eom", right_index=True, how="left")
    for c in ["target", "fg", "qe", "sov_spread_shock"]:
        long = long.merge(ecb[c].rename(f"mp_{c}"), left_on="eom", right_index=True, how="left")
    long = long.dropna(subset=["infS"]).reset_index(drop=True)
    if len(long) < 100:
        print(f"  {mkt}: troppo pochi dati ISIN dopo il merge ({len(long)})")
        return None
    # breakeven -> bp (se in frazione/percentuale, la mediana assoluta e' <1)
    yv = long["bkev"] * 100 if long["bkev"].abs().median() < 1 else long["bkev"]
    base = long[["infS", "ttm", "mp_target", "mp_fg", "mp_qe", "mp_sov_spread_shock"]].copy()
    base.insert(0, "y", yv.values)
    r1, n1 = _ols_nw(base["y"], base[["infS", "ttm"]])
    r3, n3 = _ols_nw(base["y"], base[["infS", "ttm", "mp_target", "mp_fg", "mp_qe",
                                      "mp_sov_spread_shock"]])
    def _g(res, k):
        if res is None:
            return (np.nan, np.nan)
        return res.get(k, (np.nan, np.nan))
    b1, t1 = _g(r1, "infS"); b3, t3 = _g(r3, "infS")
    print(f"  {mkt} (ISIN, pooled, n={n3}):")
    print(f"     baseline  bkev ~ InfS + ttm             InfS = {b1:+.2f} [{t1:+.1f}]")
    print(f"     netto BCE + target/fg/qe/sovshock       InfS = {b3:+.2f} [{t3:+.1f}]")
    return (mkt, b1, t1, b3, t3, n3)


# ================================================================== 3) benchmark US/UK (curva fitta)
def test_curve_benchmark(mkt, ecb):
    """US/UK: curve reali FITTE. Breakeven su curva a 5y/10y ~ surp nazionale, netto BCE.
    Controllo di sanita': come reagisce un breakeven ben misurato su sovrano sicuro?"""
    try:
        bei = rp.bei_us((5.0, 10.0)) if mkt == "US" else rp.bei_uk((5.0, 10.0))
    except Exception as e:
        print(f"  {mkt}: BEI non calcolabile ({str(e)[:40]})")
        return None
    bei = rp.eom(bei) if hasattr(rp, "eom") else bei
    bei.index = pd.to_datetime(bei.index) + pd.offsets.MonthEnd(0)
    surp = rp.surprise_maffei(mkt)   # sorpresa nazionale del paese (US/UK)
    surp.index = pd.to_datetime(surp.index) + pd.offsets.MonthEnd(0)
    out = []
    for ten in (5.0, 10.0):
        if ten not in bei.columns:
            continue
        y = (bei[ten] * 100).dropna()
        X = pd.concat([surp.rename("infS"),
                       ecb["target"].rename("mp_target"),
                       ecb["fg"].rename("mp_fg"),
                       ecb["qe"].rename("mp_qe")], axis=1).reindex(y.index)
        # NB: i fattori BCE non sono lo shock giusto per US/UK (altra banca centrale),
        # quindi qui il controllo BCE e' solo indicativo; il punto e' il BASELINE.
        r1, n1 = _ols_nw(y, surp.rename("infS").to_frame().reindex(y.index))
        b1, t1 = (r1.get("infS", (np.nan, np.nan)) if r1 else (np.nan, np.nan))
        print(f"  {mkt} {ten:g}y (curva fitta): bkev ~ InfS_{mkt}  InfS = {b1:+.2f} [{t1:+.1f}]  n={n1}")
        out.append((mkt, ten, b1, t1, n1))
    return out


if __name__ == "__main__":
    ecb = _ecb_monthly()
    if ecb is None:
        raise SystemExit("[!] ecb_factors_monthly.parquet assente: lanciare prima ecb_factors.py")

    # 1) densita' curva DE
    diagnose_de_density()

    # 2) test ISIN pooled per IT e DE
    print("\n" + "=" * 78)
    print("2) TEST ISIN (gold standard) -- breakeven per TITOLO, non su curva")
    print("=" * 78)
    print("  IT deve REGGERE, DE deve essere MUTO. Se DE e' muto anche a livello ISIN,")
    print("  il t+5.1 di r2f era artefatto della curva -> storia rischio-paese CONFERMATA.")
    rows = []
    for mkt in ("IT", "DE"):
        nat = _cpi_national(mkt)
        if nat is None:
            print(f"  {mkt}: CPI nazionale assente (an_cpi_{NAT_KEY[mkt]})")
            continue
        surp = _surprise_from(nat)
        surp.index = pd.to_datetime(surp.index) + pd.offsets.MonthEnd(0)
        r = test_isin_pooled(mkt, ecb, surp)
        if r:
            rows.append(r)

    # 3) benchmark US/UK
    print("\n" + "=" * 78)
    print("3) BENCHMARK US/UK -- curve reali FITTE (decine di titoli), breakeven pulito")
    print("=" * 78)
    print("  controllo di sanita': un breakeven ben misurato su sovrano sicuro reagisce")
    print("  alla propria inflazione? (aiuta a interpretare il 'DE reagisce a 5y')")
    bench = []
    for mkt in ("US", "UK"):
        b = test_curve_benchmark(mkt, ecb)
        if b:
            bench += b

    # verdetto
    print("\n" + "=" * 78)
    print("VERDETTO ISIN -- la Germania e' muta con i TITOLI VERI?")
    print("=" * 78)
    de_row = next((r for r in rows if r[0] == "DE"), None)
    it_row = next((r for r in rows if r[0] == "IT"), None)
    if de_row and it_row:
        _, _, _, b3_de, t3_de, _ = de_row
        _, _, _, b3_it, t3_it, _ = it_row
        it_ok = abs(t3_it) >= 1.96 and b3_it > 0
        de_muto = abs(t3_de) < 1.96
        print(f"  IT (ISIN, netto BCE): InfS = {b3_it:+.2f} [{t3_it:+.1f}]  ({'regge' if it_ok else 'debole'})")
        print(f"  DE (ISIN, netto BCE): InfS = {b3_de:+.2f} [{t3_de:+.1f}]  ({'MUTO' if de_muto else 'reagisce'})")
        if it_ok and de_muto:
            print("\n  >>> con i titoli veri (ISIN) IT regge e DE e' MUTO: la storia rischio-paese")
            print("      e' CONFERMATA e il t+5.1 su curva di r2f era artefatto di fitting.")
            print("      STORIA ECONOMICA BLINDATA sul dato pulito.")
        elif it_ok and not de_muto:
            print("\n  >>> anche con i titoli veri DE reagisce: NON e' artefatto di curva, e' un")
            print("      canale comune che tocca anche la Germania. La storia va riformulata")
            print("      (rischio-paese al lungo + componente comune al breve).")
    else:
        print("  [dati ISIN incompleti: genera breakeven_{IT,DE}.parquet con 04 CON_BREAKEVEN=True]")

    # salvataggio
    allrows = [("isin", *r) for r in rows] + [("bench", m, t, b, tt, n) for (m, t, b, tt, n) in bench]
    if allrows:
        pd.DataFrame(rows, columns=["mkt", "b_base", "t_base", "b_netBCE", "t_netBCE", "n"]
                     ).to_csv(CACHE / "an_isin_falsification.csv", index=False)
    print("\nsalvato: an_isin_falsification.csv")
