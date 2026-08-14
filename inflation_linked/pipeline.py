"""run - orchestrazione: dalla cache ai pannelli delle basi, piu' regressione e metodo A.

LIBRERIA di orchestrazione: build_market, build_zero_panel, curve_basis, regression.
NON si lancia direttamente: i runner numerati 01-06 la eseguono passo per passo,
ognuno per intero, con le impostazioni come costanti in testa al file.

COSA PRODUCE build_market(mkt) in data/cache/
  basis_{mkt}_interp.parquet    metodo B/C-semplice: TotalYtM - YTM nominale interpolato
                                fra i due adiacenti (la misura dell'ORIGINALE -> regressione)
  basis_{mkt}_nearest.parquet   metodo B FLL: singolo nominale piu' vicino (<=183gg),
                                la versione tradabile
  basis_{mkt}_cexact.parquet    metodo C-esatto: IRR osservato - IRR sintetico sulla curva
                                zero nominale (mismatch zero, effetto cedola cancellato)
  totalytm_{mkt}.parquet, ynom_interp_{mkt}.parquet, mismatch_{mkt}.parquet
  breakeven_{mkt}.parquet       (con --real) TotalYtM - RealYtM
  floorval_{mkt}.parquet        (con --floor) valore del floor in bp (port Black originale)

CURVA ZERO NOMINALE per il C-esatto
  US    : GSW nominale valutata dai parametri NSS pubblicati (qualunque maturita')
  UK    : curva spot nominale BoE (gia' fittata, Anderson-Sleath)
  euro  : NSS fittata QUI dai YTM dei nominali (obiettivo GSW-equivalente, warm start,
          fallback NS con pochi bond), parametri in cache incrementale nss_{mkt}.parquet.
          Diagnostica per data: n_bonds, rmse_bp, modello -- da guardare prima di fidarsi.

SEGMENTO UK: ENTRAMBI gli stili, con le rispettive convenzioni implementate in basis.py.
New-style (1216): real-clean, lag 3 interpolato. Old-style (44/99): NOMINAL-clean (inflazione
maturata dentro il prezzo, rateo in moneta sulla cedola nota), lag 8 senza interpolazione.
Ex-dividend a 7 business day per entrambi (rateo negativo, cedola esclusa). I gilt NON hanno
floor di deflazione: MARKET_FLOOR lo spegne per il UK. Recuperare gli old-style sblocca il
campione UK 2004-2010 (GFC inclusa), dove i new-style quasi non esistono. --uk-new-only per
riprodurre il comportamento precedente. VERIFICA UNA TANTUM sul terminale: che il PX quotato
di un old-style (YAS) sia davvero nominal-clean come da convenzione BoE/DMO.

MODALITA' REGRESSIONE = tutti gli interruttori come l'originale: floor=True, is_total=True,
ytm_convention='local', pub_lag_days=None, exclude_tail=False, metodo 'interp'.
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import holidays as _hol

import bbg
import curves
from basis import (LinkerBond, bond_basis_row, value_floor, regression_check,
                   MAX_MISMATCH_DAYS)
from bbg import NOMINAL_POOL_ALIAS
from config import CACHE, DATA, MARKETS, REGRESSION_TARGET
from inflation import InflationEngine

ZERO_GRID = np.round(np.arange(0.25, 50.26, 0.25), 2)
# floor di deflazione: proprieta' dello STRUMENTO, non scelta di misura.
# BTPei/OATei/OATi/Bundei: par floor a scadenza; TIPS: floor sul principale;
# gilt UK (entrambi gli stili): NESSUN floor.
MARKET_FLOOR = {"IT": True, "FR": True, "FR_CPI": True, "DE": True,
                "US": True, "UK": False}


# ------------------------------------------------------------------ curva zero nominale
def _euro_nss_params(mkt: str, ref_nom: pd.DataFrame, ytm: pd.DataFrame,
                     dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Parametri NSS per data, cache incrementale su nss_{mkt}.parquet."""
    path = CACHE / f"nss_{mkt}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    todo = dates if old is None else dates.difference(old.index)
    if len(todo):
        mats = pd.to_datetime(ref_nom["MATURITY"])
        rows = []
        for dt in todo:
            if dt not in ytm.index:
                continue
            y = ytm.loc[dt].dropna()
            isins = [i for i in y.index if i in mats.index]
            tau = np.array([(mats[i] - dt).days / 365.25 for i in isins])
            keep = (tau > 0.25) & (tau < 50)
            if keep.sum() < 4:
                continue
            y_cc = curves.ytm_to_cc(mkt, y[isins].values[keep])
            rows.append(pd.DataFrame({"date": dt, "tau": tau[keep], "y_cc": y_cc}))
        if rows:
            fit = curves.fit_nss_panel(pd.concat(rows, ignore_index=True))
            old = fit if old is None else pd.concat([old, fit]).sort_index()
            old = old[~old.index.duplicated(keep="last")]
            old.to_parquet(path)
    if old is None:
        raise RuntimeError(f"nessun fit NSS possibile per {mkt}")
    bad = old[old["rmse_bp"] > 12]
    if len(bad):
        print(f"  ATTENZIONE {mkt}: {len(bad)} date con RMSE del fit > 12bp "
              f"(peggiore {old['rmse_bp'].max():.1f}bp il {old['rmse_bp'].idxmax().date()})")
    return old


def build_zero_panel(mkt: str, ref_nom: pd.DataFrame | None = None,
                     ytm: pd.DataFrame | None = None,
                     dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Pannello zero nominale (cc, %, colonne = maturita' in anni) per il C-esatto."""
    if mkt == "US":
        raw = pd.read_parquet(CACHE / "gsw_nominal_raw.parquet")
        return curves.gsw_zero_panel("nominal", ZERO_GRID, df=raw)
    if mkt == "UK":
        return curves.load_panel("boe_nominal_spot_std")
    if mkt == "DE" and getattr(__import__("config"), "DE_NOMINAL_CURVE", "fit") == "bundesbank":
        try:
            return curves.buba_zero_panel(ZERO_GRID)
        except FileNotFoundError:
            print("  DE: parametri Bundesbank assenti in cache -> fallback al fit NSS "
                  "(lanciare 03 con SCARICA_BUNDESBANK=True, o scaricare dal portale)")
    params = _euro_nss_params(mkt, ref_nom, ytm, dates)
    b3 = np.where(params["model"].eq("NS"), 0.0, params["b3"])
    out = {float(t): curves.nss_yield(t, params["b0"].values, params["b1"].values,
                                      params["b2"].values, b3,
                                      params["t1"].values, params["t2"].values)
           for t in ZERO_GRID}
    return pd.DataFrame(out, index=params.index)


# ------------------------------------------------------------------ costruzione mercato
def build_market(mkt: str, methods=("interp", "nearest", "c_exact"),
                 is_total: bool = True, floor: bool | None = None,
                 uk_new_only: bool = False,
                 with_real: bool = False, with_floor_value: bool = False,
                 exclude_tail: bool = False, ytm_convention: str = "local",
                 pub_lag_days: int | None = None, price_field: str = "px_mid",
                 years_history: int = 10, season_weight: float = 1.0,
                 dates: pd.DatetimeIndex | None = None, save: bool = True) -> dict:
    t0 = time.time()
    m = MARKETS[mkt]
    if floor is None:
        floor = MARKET_FLOOR.get(mkt, True)
    nom_mkt = NOMINAL_POOL_ALIAS.get(mkt, mkt)

    ref_l = bbg.load("ref_linker")
    ref_l = ref_l[ref_l["mkt"] == mkt]
    if mkt == "UK":
        n_new = int((ref_l["segment"] == "new").sum())
        n_old = int((ref_l["segment"] == "old").sum())
        if uk_new_only:
            ref_l = ref_l[ref_l["segment"] == "new"]
            print(f"  UK: --uk-new-only -> {n_new} new-style, {n_old} old-style esclusi")
        else:
            print(f"  UK: {n_new} new-style (RC, lag3) + {n_old} old-style "
                  f"(NC, lag8, ex-div 7bd) -- il pre-2005 vive negli old-style")
    if "base_cpi_final" in ref_l.columns:
        bad = ref_l["base_cpi_final"].isna() | (ref_l["base_cpi_final"] <= 0)
        if bad.any():
            print(f"  {mkt}: esclusi {int(bad.sum())} linker senza BASE_CPI valida")
            ref_l = ref_l[~bad]

    ref_n = bbg.load("ref_nominal")
    ref_n = ref_n[ref_n["mkt"] == nom_mkt][["MATURITY", "FIRST_SETTLE_DT", "AMT_OUTSTANDING"]]
    px = bbg.load(f"{price_field}_{mkt}")
    ytm = bbg.load(f"ytm_{nom_mkt}")
    ils = bbg.load(f"ils_{m.ils}")
    cpi = bbg.load(f"cpi_{m.cpi}").iloc[:, 0]

    eng = InflationEngine(m.cpi, cpi, ils, years_history=years_history,
                          season_weight=season_weight, pub_lag_days=pub_lag_days)
    bonds = {i: LinkerBond.from_ref(i, r) for i, r in ref_l.iterrows()}
    settle_cal = m.holidays
    cpn_cal = dict({"IT": _hol.Italy, "FR": _hol.France, "FR_CPI": _hol.France,
                    "DE": _hol.Germany, "UK": _hol.UK, "US": _hol.US}[mkt]())

    if dates is None:
        dates = px.index.intersection(ytm.index)
        dates = dates[dates >= ils.dropna(how="all").index.min()]
    zero = None
    if "c_exact" in methods:
        zero = build_zero_panel(mkt, ref_n, ytm, dates)

    cols = list(bonds)
    panels = {k: pd.DataFrame(np.nan, index=dates, columns=cols)
              for k in ["basis_interp", "basis_nearest", "basis_cexact",
                        "totalytm", "ynom_interp", "mismatch"]}
    if with_real:
        panels["breakeven"] = pd.DataFrame(np.nan, index=dates, columns=cols)
    if with_floor_value:
        panels["floorval"] = pd.DataFrame(np.nan, index=dates, columns=cols)

    n_err, first_errs = 0, []
    for k, asof_ts in enumerate(dates):
        asof = asof_ts.date() if hasattr(asof_ts, "date") else asof_ts
        ytm_row = ytm.loc[asof_ts] if asof_ts in ytm.index else None
        zero_row = None
        if zero is not None:
            zr = zero.loc[:asof_ts]
            zero_row = zr.iloc[-1] if len(zr) else None
        for isin, b in bonds.items():
            p = px.at[asof_ts, isin] if isin in px.columns else np.nan
            if pd.isna(p) or asof >= b.maturity:
                continue
            if exclude_tail and asof > b.maturity - pd.Timedelta(days=365).to_pytimedelta():
                continue
            try:
                row = bond_basis_row(b, asof, eng, float(p), ref_n, ytm_row,
                                     settle_cal, cpn_cal,
                                     zero_row if "c_exact" in methods else None,
                                     floor=floor, ytm_convention=ytm_convention)
                panels["basis_interp"].at[asof_ts, isin] = row["basis_interp"]
                panels["basis_nearest"].at[asof_ts, isin] = row["basis_nearest"]
                panels["totalytm"].at[asof_ts, isin] = row["total_ytm"]
                panels["ynom_interp"].at[asof_ts, isin] = row["ynom_interp"]
                panels["mismatch"].at[asof_ts, isin] = row["mismatch_days"]
                if "basis_c_exact" in row:
                    panels["basis_cexact"].at[asof_ts, isin] = row["basis_c_exact"]
                if with_real:
                    cf_r = b.cashflows(asof, eng, float(p), settle_cal, cpn_cal,
                                       is_total=False, floor=floor)
                    panels["breakeven"].at[asof_ts, isin] = \
                        row["total_ytm"] - b.irr(cf_r, asof)
                if with_floor_value:
                    panels["floorval"].at[asof_ts, isin] = value_floor(b, asof, eng)
            except Exception as e:
                n_err += 1
                if len(first_errs) < 5:
                    first_errs.append(f"{isin} @ {asof}: {e}")
        if (k + 1) % 250 == 0:
            print(f"  {mkt}: {k+1}/{len(dates)} date ({time.time()-t0:.0f}s)")

    if save:
        suf = "" if pub_lag_days is None else f"_lag{pub_lag_days}"
        for name, df in panels.items():
            df.dropna(how="all").to_parquet(CACHE / f"{name}_{mkt}{suf}.parquet")
    done = panels["basis_interp"].notna().sum().sum()
    print(f"  {mkt}: {len(bonds)} linker x {len(dates)} date -> {int(done)} osservazioni, "
          f"{n_err} errori, {time.time()-t0:.0f}s")
    for e in first_errs:
        print(f"    err: {e}")
    return panels


# ------------------------------------------------------------------ metodo A (curve)
CURVE_PILLARS = {"UK": [3.0, 5.0, 10.0, 30.0], "US": [2.0, 5.0, 10.0, 30.0]}


def curve_basis(mkt: str, save: bool = True) -> pd.DataFrame:
    """lambda(tau) = reale + ILS(cc) - nominale, a scadenza costante. Solo UK/US
    (le curve reali pubbliche esistono solo li'; per l'euro il C-esatto e' la via)."""
    pil = CURVE_PILLARS[mkt]
    if mkt == "UK":
        real = curves.interp_maturities(curves.load_panel("boe_real_spot_std"), pil)
        nom = curves.interp_maturities(curves.load_panel("boe_nominal_spot_std"), pil)
        ils = bbg.load("ils_BPSWIT")
    else:
        raw_n = pd.read_parquet(CACHE / "gsw_nominal_raw.parquet")
        raw_t = pd.read_parquet(CACHE / "gsw_tips_raw.parquet")
        nom = curves.gsw_zero_panel("nominal", pil, df=raw_n)
        real = curves.gsw_zero_panel("tips", pil, df=raw_t)
        ils = bbg.load("ils_USSWIT")
    ils = ils[[float(p) for p in pil if float(p) in ils.columns]]
    ils_cc = curves.ann_to_cc(ils)
    idx = real.index.intersection(nom.index).intersection(ils_cc.index)
    lam = (real.loc[idx] + ils_cc.loc[idx] - nom.loc[idx]) * 100.0     # bp
    lam.columns = [f"{mkt}{int(p)}Y" for p in pil]
    if save:
        lam.to_parquet(CACHE / f"curve_basis_{mkt}.parquet")
    print(f"  curve basis {mkt}: {lam.shape[0]} date, pillar {pil}, "
          f"medie {lam.mean().round(1).to_dict()}")
    return lam


# ------------------------------------------------------------------ regressione
def regression(years_history: int = 10, season_weight: float = 1.0) -> None:
    """Italia vs btpei_basis.xlsx, interruttori come l'originale. Finche' non passa
    (tol da config), non si tocca nient'altro. years_history/season_weight DEVONO
    essere quelli usati per costruire btpei_basis.xlsx (vedi instructions_btpei.ipynb)."""
    tgt_path = DATA / "btpei_basis.xlsx"
    if not tgt_path.exists():
        tgt_path = REGRESSION_TARGET
    tgt = pd.read_excel(tgt_path, index_col=0, parse_dates=True)
    isins = [str(c).replace("_Basis", "") for c in tgt.columns]
    price_field = "px_ask" if (CACHE / "px_ask_IT.parquet").exists() else "px_mid"
    if price_field == "px_mid":
        print("ATTENZIONE: px_ask_IT assente -- l'originale usava PX_ASK@CBBT. Col MID la"
              "\nregressione mostrera' uno scarto sistematico (~ meta' bid-ask in yield):"
              "\nper il confronto esatto rilancia  python bbg.py --fetch --trading")
    # NB: years_history e season_weight devono essere QUELLI usati per costruire"
    # btpei_basis.xlsx (verificali in instructions_btpei.ipynb) -- qui i default.
    pan = build_market("IT", methods=("interp",), is_total=True, floor=True,
                       ytm_convention="local", pub_lag_days=None,
                       price_field=price_field, years_history=years_history,
                       season_weight=season_weight,
                       dates=pd.DatetimeIndex(tgt.index), save=False)
    comp = pan["basis_interp"][[c for c in pan["basis_interp"].columns if c in isins]]

    # --- DUMP DIAGNOSTICO: vecchio vs nuovo affiancati, per capire la natura dello scarto ---
    import os
    diag_isins = [os.environ.get("DIAG_ISIN", "IT0005004426"), "IT0003745541"]
    for iso in diag_isins:
        if iso in comp.columns and iso in tgt.columns:
            d = pd.DataFrame({"vecchio": tgt[iso], "nuovo": comp[iso]}).dropna()
            d["diff_bp"] = d["nuovo"] - d["vecchio"]
            d["ratio"] = d["nuovo"] / d["vecchio"]
            outp = CACHE / f"_diag_{iso}.csv"
            d.to_csv(outp)
            print(f"  [diag {iso}] {len(d)} righe -> {outp.name} | "
                  f"vecchio[:3]={d['vecchio'].head(3).round(2).tolist()} "
                  f"nuovo[:3]={d['nuovo'].head(3).round(2).tolist()} "
                  f"ratio_mediano={d['ratio'].median():.3f}")

    rep = regression_check(comp, tgt_path)
    print(rep.head(10).to_string())
