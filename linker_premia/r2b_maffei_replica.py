"""r2b - Replica ESATTA della specificazione Maffei (tesi 15/6/2026) per la domanda
RR n.2: "do we find the same for Linkers?". Nessuna invenzione: ogni scelta segue la
tesi, sezioni 4-6; dove un dato UK non esiste (CPI SA, calendario ONS) il fallback e'
dichiarato a schermo.

Differenze implementate rispetto a r2 (che resta la nostra specificazione):
  1. campionamento alla DATA DI RILASCIO del CPI, non a fine mese (tesi, sez. 4):
     proxy = primo giorno lavorativo >= giorno D del mese successivo al mese di
     riferimento (US: D=12, UK: D=15). Override con le date esatte se esiste
     CACHE/cpi_release_dates_{mkt}.csv (una colonna di date, es. da Bloomberg ECO).
  2. sorpresa su CPI DESTAGIONALIZZATO per gli US (tesi, sez. 6.1) se an_cpi_US_SA
     esiste (r0 aggiornato); UK resta RPI NSA (nessuna serie SA ufficiale): dichiarato.
  3. media mobile 120 mesi PIENA (min_periods=120, "without any truncation").
  4. liquidita' = PC1 di (VIX, MOVE, GVLQUSD, on/off-run 10y) standardizzate
     (tesi, sez. 6.1.1); robustezza con VIX da solo. PCA su cio' che an_liq contiene.
  5. regressioni SEPARATE come eq. (7) della tesi: lambda~PCA, lambda~VIX,
     ISR~PCA, ISR~VIX (test di validazione 7.1.2: atteso beta=0), lambda~InfS;
     per gli US anche gamma_hat = BEI - pi_hat(Cleveland) ~ PCA.
  6. griglia tenor della tesi [2..10,12,15,20] (UK dal 2.5: limite curva reale BoE)
     e sub-sample di confronto ago-2007 -> set-2024.
Output: an_maffei_replica.csv + tabella a schermo con la Tab. V della tesi a fianco.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np
import pandas as pd
import rp
from config import CACHE

MATS = {"US": (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20),
        "UK": (2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20),
        "IT": (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20),
        "FR": (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20),
        "DE": (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)}
REL_DAY = {"US": 12, "UK": 15, "IT": 17, "FR": 15, "DE": 15}   # proxy release HICP/CPI
MAFFEI_SPAN = ("2007-08-01", "2024-09-30")
MAFFEI_TABV = ("[Maffei Tab.V, lambda~InfS, US 2007-2024]  2y -1.12 (-1.3) | "
               "5y -0.64 (-0.8) | 10y -1.16 (-2.7) | 12y -1.40 (-3.9) | 20y -2.45 (-7.2)")


def release_grid(mkt: str, months: pd.DatetimeIndex) -> pd.Series:
    """Per ogni mese di riferimento m: la data in cui il mercato riceve il CPI di m.
    Se esiste il calendario esatto (cpi_release_dates_{mkt}.csv) usa quello, mappando
    ogni release al mese di riferimento precedente; altrimenti proxy dichiarata."""
    f = CACHE / f"cpi_release_dates_{mkt}.csv"
    if f.exists():
        rel = pd.to_datetime(pd.read_csv(f).iloc[:, 0]).sort_values()
        out = {}
        for m in months:                       # release di m = prima data nel mese m+1
            nxt = rel[(rel > m) & (rel <= m + pd.offsets.MonthEnd(2))]
            if len(nxt):
                out[m] = nxt.iloc[0]
        print(f"  [{mkt}] calendario release ESATTO: {f.name} ({len(out)} mesi)")
        return pd.Series(out)
    d = REL_DAY[mkt]
    out = {m: (m + pd.offsets.MonthBegin(1) + pd.Timedelta(days=d - 1)
               + pd.offsets.BusinessDay(0)) for m in months}
    print(f"  [{mkt}] release PROXY: primo bd >= {d} del mese successivo "
          f"(override: {f.name} da Bloomberg ECO)")
    return pd.Series(out)


def sample_at(daily: pd.DataFrame, grid: pd.Series) -> pd.DataFrame:
    """Ultimo valore disponibile <= data di release; indicizzato al MESE di riferimento."""
    daily = daily.sort_index()
    rows = {m: daily.asof(g) for m, g in grid.items() if g >= daily.index.min()}
    return pd.DataFrame(rows).T.dropna(how="all")


def pca1(df: pd.DataFrame) -> pd.Series:
    """PC1 di variabili standardizzate (tesi 6.1.1); segno orientato su MOVE/VIX.
    Robusto: scarta colonne a varianza nulla/insufficiente, allinea le righe PRIMA di
    standardizzare, e ricade sulla proxy migliore se resta una colonna sola."""
    d = df.copy()
    # 1. scarta colonne quasi tutte-NaN o costanti (std=0 -> divisione degenere)
    keep = [c for c in d.columns if d[c].notna().sum() >= 30 and d[c].std(skipna=True) > 1e-9]
    dropped = [c for c in d.columns if c not in keep]
    if dropped:
        print(f"  [PCA] scarto proxy inutilizzabili (NaN/costanti): {dropped}")
    d = d[keep]
    # 2. allinea le righe (tutte le proxy presenti nello stesso mese) PRIMA di standardizzare
    d = d.dropna()
    if d.shape[1] < 2 or len(d) < 30:
        col = d.columns[0] if d.shape[1] else df.columns[0]
        print(f"  [PCA] proxy insufficienti per la PCA: uso {col} standardizzata come fattore")
        base = df[col].dropna()
        return ((base - base.mean()) / base.std()).rename("PCA")
    z = (d - d.mean()) / d.std()
    C = np.cov(z.values.T)
    if not np.all(np.isfinite(C)):
        raise ValueError(f"[PCA] covarianza non finita su {list(z.columns)}")
    ev, vec = np.linalg.eigh(C)
    pc = pd.Series(z.values @ vec[:, -1], index=z.index, name="PCA")
    ref = "MOVE" if "MOVE" in z.columns else z.columns[0]
    if pc.corr(z[ref]) < 0:
        pc = -pc
    share = ev[-1] / ev.sum() * 100
    print(f"  PC1 su {list(z.columns)}: varianza spiegata {share:.1f}%  (tesi: 64.1% su 4 proxy)")
    return pc


def sep_reg(dep: pd.DataFrame, x: pd.Series, lab: str, lags: int = 6) -> pd.DataFrame:
    """Regressione separata by-T (tesi eq. 7), OLS con t di Newey-West."""
    rows = []
    for n in dep.columns:
        d = pd.concat([(dep[n] * 100).rename("y"), x.rename("x")], axis=1).dropna()
        if len(d) < 30:
            continue
        b, e, _, r2, X = rp.ols(d["y"].values, d[["x"]].values)
        t = rp.nw_t(e, X, b, lags)
        rows.append([lab, float(n), len(d), b[1], t[1], r2])
    return pd.DataFrame(rows, columns=["reg", "mat", "n", "beta", "t", "R2"])


def show(t: pd.DataFrame, head: str):
    if t.empty:
        print(f"  {head}: (nessun dato)")
        return
    print(f"  {head}:  " + "  ".join(
        f"{r.mat:g}y {r.beta:+.2f}({r.t:+.1f})" for r in t.itertuples()))


def show_tab(t: pd.DataFrame, head: str):
    """Tabella per tenore con beta, t e R2 -- il formato delle Tab. III e V della tesi.
    L'R2 non e' decorativo: l'argomento di segmentazione di Maffei (sez. 7.1.1) si regge
    sulla VARIANZA SPIEGATA che sfuma al crescere del tenore per la liquidita', mentre
    la dipendenza dalle sorprese si rafforza sul lungo."""
    if t.empty:
        print(f"  {head}: (nessun dato)")
        return
    print(f"  {head}")
    print("         " + "".join(f"{('%gy' % r.mat):>9s}" for r in t.itertuples()))
    print("    beta " + "".join(f"{r.beta:>9.3f}" for r in t.itertuples()))
    print("    t    " + "".join(f"{r.t:>9.1f}" for r in t.itertuples()))
    print("    R2   " + "".join(f"{r.R2:>9.3f}" for r in t.itertuples()))


def run(mkt: str, span=None):
    tag = f"{mkt}" + (f" {span[0][:7]}->{span[1][:7]}" if span else " full")
    print(f"\n=== {tag} | replica Maffei (release-date, MA120 piena, reg. separate) ===")
    # 1. lambda, BEI, ISR giornalieri -> campionati alla release
    fn = {"US": rp.bei_us, "UK": rp.bei_uk}.get(mkt, lambda m: rp.bei_euro(mkt, m))
    bei_d, isr_d = fn(MATS[mkt]), rp.isr(mkt, MATS[mkt])
    months = pd.date_range(bei_d.index.min().normalize().replace(day=1),
                           bei_d.index.max(), freq="ME")
    grid = release_grid(mkt, months)
    bei, s = sample_at(bei_d, grid), sample_at(isr_d, grid)
    idx = bei.index.intersection(s.index)
    bei, s = bei.loc[idx], s.loc[idx]
    lam = s - bei
    # 2. sorpresa: CPI SA per US se disponibile, MA120 piena. Euro: CPI del paese in cache.
    src = mkt
    if mkt == "US" and (CACHE / "an_cpi_US_SA.parquet").exists():
        src = "US_SA"
    elif mkt == "US":
        print("  [!] an_cpi_US_SA assente: sorpresa su CPURNSA (Maffei usa SA) -- "
              "aggiornare r0 e riscaricare")
    if mkt in ("IT", "FR", "DE") and not (CACHE / "cpi_CPTFEMU.parquet").exists():
        print(f"  [!] cpi_CPTFEMU assente: sorpresa euro non calcolabile. Salto {mkt}.")
        return pd.DataFrame()
    if mkt in ("IT", "FR", "DE"):
        print("  [nota] sorpresa su HICP EUROZONA (CPTFEMU), comune ai tre: la sorpresa"
              " e' AGGREGATA, non nazionale -> misura la reazione del breakeven di questo"
              " paese allo shock d'inflazione dell'AREA, non del paese.")
    pi = rp.yoy(rp.cpi(src)).dropna()
    surp = (pi - pi.rolling(120, min_periods=120).mean()).dropna()
    surp = surp.reindex(lam.index)                    # mese di riferimento m
    # 3. liquidita': PCA su cio' che c'e' + VIX, alla release
    liq = None
    try:
        raw = pd.read_parquet(CACHE / "an_liq.parquet")
        raw.index = pd.to_datetime(raw.index)
        lg = sample_at(raw, grid).reindex(lam.index)
        pc = pca1(lg)
        vix = ((lg["VIX"] - lg["VIX"].mean()) / lg["VIX"].std()
               if "VIX" in lg.columns else None)
        liq = pd.concat([pc, vix.rename("VIX") if vix is not None else None], axis=1)
        if lg.shape[1] < 4:
            print(f"  [!] an_liq ha solo {list(lg.columns)}: per la PCA a 4 della tesi "
                  "servono anche GVLQUSD e G0111Z (r0 aggiornato)")
    except FileNotFoundError:
        print("  [!] an_liq assente: salto le regressioni di liquidita' (lanciare r0)")
    if span:
        cut = slice(pd.Timestamp(span[0]), pd.Timestamp(span[1]))
        lam, s, bei, surp = lam.loc[cut], s.loc[cut], bei.loc[cut], surp.loc[cut]
        if liq is not None:
            liq = liq.loc[cut]
    # 4. le regressioni separate della tesi (eq. 7)
    tabs = []
    if liq is not None:
        tabs.append(sep_reg(lam, liq["PCA"], "lambda~PCA"))
        show_tab(tabs[-1], "lambda ~ PCA   [bp/sd]   (tesi Tab. III)")
        if "VIX" in liq.columns:
            tabs.append(sep_reg(lam, liq["VIX"], "lambda~VIX"))
            show(tabs[-1], "lambda ~ VIX   [bp/sd]")
        tabs.append(sep_reg(s, liq["PCA"], "ISR~PCA"))
        show(tabs[-1], "ISR ~ PCA  (validazione 7.1.2: atteso ~0)")
        if "VIX" in liq.columns:
            tabs.append(sep_reg(s, liq["VIX"], "ISR~VIX"))
            show(tabs[-1], "ISR ~ VIX  (validazione 7.1.2: atteso ~0)")
    tabs.append(sep_reg(lam, surp, "lambda~InfS"))
    show_tab(tabs[-1], "lambda ~ InfS  [bp/unit]  (tesi Tab. V)")
    if mkt == "US":
        print("  " + MAFFEI_TABV)
        try:                                        # gamma_hat = BEI - pi_hat(Cleveland)
            pih = rp.interp_cols(rp.expinf_us(), MATS["US"], warn=False)
            pih = pih.reindex(lam.index, method="ffill")
            gam = bei - pih
            if liq is not None:
                tabs.append(sep_reg(gam, liq["PCA"], "gamma~PCA"))
                show(tabs[-1], "gamma_hat ~ PCA  [bp/sd] (tesi eq. 7, riga 1)")
        except Exception as e:
            print(f"  (gamma_hat saltato: {e})")
    # ---- SEGMENTAZIONE (tesi sez. 7.1.1): non e' un test a parte, e' il confronto dei DUE
    # pattern lungo la curva. Maffei trova: (i) liquidita' -> beta e R2 CALANO col tenore
    # (Tab. III: R2 0.466@2y, picco 0.710@5y, 0.438@10y); (ii) sorprese -> effetto piu'
    # FORTE sul LUNGO con beta NEGATIVO e t significativi solo in fondo (Tab. V: 2y -1.12
    # t-1.3 non signif. -> 20y -2.45 t-7.2). Lettura: i lunghi stanno agli assicuratori
    # (poco sensibili alla liquidita', molto agli shock d'inflazione), i corti ai trader.
    # NB: si selezionano le tabelle PER NOME. Prima si usava tabs[-1], che per gli US e'
    # gamma~PCA (appesa dopo) e non lambda~InfS: il verdetto US era sulla regressione
    # sbagliata.
    def _tab(name):
        for t_ in tabs:
            if len(t_) and t_["reg"].iloc[0] == name:
                return t_.sort_values("mat")
        return pd.DataFrame()

    t_liq, t_inf = _tab("lambda~PCA"), _tab("lambda~InfS")

    def _short_long(t_, k=3):
        """medie di |beta| e R2 sul tratto corto e su quello lungo della curva."""
        if len(t_) < 4:
            return None
        s_, l_ = t_.head(k), t_.tail(k)
        return (abs(s_.beta).mean(), s_.R2.mean(), abs(l_.beta).mean(), l_.R2.mean(),
                s_.mat.min(), s_.mat.max(), l_.mat.min(), l_.mat.max())

    print("\n  --- SEGMENTAZIONE (tesi 7.1.1): i due pattern lungo la curva ---")
    liq_fade = inf_grow = None
    if len(t_liq) >= 4:
        b_s, r_s, b_l, r_l, m0, m1, m2, m3 = _short_long(t_liq)
        liq_fade = (r_l < r_s) and (b_l < b_s)
        print(f"    liquidita' (PCA): corto {m0:g}-{m1:g}y |b|={b_s:.3f} R2={r_s:.3f}  ->  "
              f"lungo {m2:g}-{m3:g}y |b|={b_l:.3f} R2={r_l:.3f}   "
              f"{'SFUMA sul lungo (come Maffei)' if liq_fade else 'NON sfuma'}")
    if len(t_inf) >= 4:
        b_s, r_s, b_l, r_l, m0, m1, m2, m3 = _short_long(t_inf)
        t_long = t_inf.tail(3)["t"].abs().max()
        neg_long = t_inf.tail(3)["beta"].mean() < 0
        inf_grow = (b_l > b_s) and (t_long >= 1.96)
        print(f"    sorprese (InfS): corto {m0:g}-{m1:g}y |b|={b_s:.3f} R2={r_s:.3f}  ->  "
              f"lungo {m2:g}-{m3:g}y |b|={b_l:.3f} R2={r_l:.3f}  |t|max_lungo={t_long:.1f}   "
              f"{'CRESCE sul lungo' if inf_grow else 'NON cresce'}"
              f"{', beta negativo' if neg_long else ', beta positivo'}")
    if liq_fade is not None and inf_grow is not None:
        if liq_fade and inf_grow:
            print("    >>> SEGMENTAZIONE COME MAFFEI: liquidita' sul corto, sorprese sul lungo.")
            print("        (lettura della tesi: lunghi agli assicuratori, corti ai trader)")
        elif inf_grow and not liq_fade:
            print("    >>> parziale: le sorprese si concentrano sul lungo, ma la liquidita'")
            print("        non sfuma -> segmentazione per clientela non confermata da questo lato")
        elif liq_fade and not inf_grow:
            print("    >>> parziale: la liquidita' sfuma come nella tesi, ma le sorprese non")
            print("        si rafforzano sul lungo -> il canale InfS non e' segmentato qui")
        else:
            print("    >>> NESSUNA segmentazione nel senso della tesi: entrambi i pattern assenti")
    out = pd.concat(tabs) if tabs else pd.DataFrame()
    out.insert(0, "mkt", mkt)
    out.insert(1, "span", tag.split(" ", 1)[1])
    return out


if __name__ == "__main__":
    res = []
    for mkt in ("US", "UK", "IT", "FR", "DE"):
        t = run(mkt)                               # campione pieno, tutti i mercati
        if not t.empty:
            res.append(t)
    res.append(run("US", MAFFEI_SPAN))             # check di replica con la tesi
    print("\n" + "=" * 78)
    print("SINTESI -- la reazione delle SORPRESE (lambda~InfS) per mercato")
    print("=" * 78)
    print("  cerchiamo: segno del beta e su quale tratto della curva si concentra.")
    print("  US (TIPS, Maffei): NEGATIVO sul LUNGO -> assicuratori comprano nell'inflazione.")
    print("  la storia economica e' piu' forte dove il pattern e' NETTO e INTERPRETABILE")
    print("  (segno stabile lungo la curva, t significativi, e una clientela identificabile).")
    all_res = pd.concat(res) if res else pd.DataFrame()
    if not all_res.empty:
        li = all_res[(all_res["reg"] == "lambda~InfS") & (all_res["span"] == "full")]
        for mkt in ("US", "UK", "IT", "FR", "DE"):
            d = li[li["mkt"] == mkt].sort_values("mat")
            if d.empty:
                continue
            short, lng = d.iloc[0], d.iloc[-1]
            seg = "LUNGO" if abs(lng.beta) > abs(short.beta) else "CORTO"
            sig = int((d["t"].abs() >= 1.96).sum())
            sgn = "neg" if d.beta.mean() < 0 else "pos"
            print(f"  {mkt}: segno {sgn} | conc. {seg} | signif. {sig}/{len(d)} tenor | "
                  f"{short.mat:g}y {short.beta:+.1f}(t{short.t:+.1f}) .. "
                  f"{lng.mat:g}y {lng.beta:+.1f}(t{lng.t:+.1f})")
    all_res.to_csv(CACHE / "an_maffei_replica.csv", index=False)
    print("\nsalvato: an_maffei_replica.csv (US, UK, IT, FR, DE)")
