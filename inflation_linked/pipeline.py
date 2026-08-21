"""run - orchestrazione: dalla cache ai pannelli delle basi (bond-level) e al metodo curva.

LIBRERIA di orchestrazione: build_market, build_zero_panel, curve_basis.
NON si lancia direttamente: i runner numerati 01-05 la eseguono passo per passo,
ognuno per intero, con le impostazioni come costanti in testa al file.

I TRE METODI DEL PROGETTO (FLL 2014 JF; Kita-Tortorice JMCB; Barria-Pinter BoE 1034):
  1. nearest      FLL/KT matched-maturity SENZA leg STRIPS (STRIP illiquidi ovunque,
                  verificato su terminale per Gilt e BTP: omissione uniforme e dichiarata).
                  TotalYtM(linker, proiezione ILS) - YTM del singolo nominale piu' vicino
                  per scadenza (mismatch in giorni, cap 183). Convenzione yield ANNUAL:
                  UK/US semi->annuale, euro invariato (l'IRR del linker e' annuale).
  2. c_exact      ISIN vs curva: IRR osservato - IRR sintetico dai flussi del linker
                  scontati sulla curva zero nominale (griglia 0.25y, sconto al regolamento,
                  identita' su curva piatta testata). Mismatch zero per costruzione.
  3. curve_basis  curva vs curva: reale + ILS(cc) - nominale a scadenza costante,
                  UK (BoE) e US (GSW dai parametri). Per l'euro: dopo il fit NSS.

COSA PRODUCE build_market(mkt) in data/cache/
  basis_{mkt}_nearest.parquet   metodo 1
  basis_{mkt}_cexact.parquet    metodo 2
  totalytm_{mkt}.parquet, ynom_nearest_{mkt}.parquet, mismatch_{mkt}.parquet
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
from basis import (settlement, LinkerBond, bond_basis_row, value_floor,
                   MAX_MISMATCH_DAYS)
from bbg import NOMINAL_POOL_ALIAS
from config import CACHE, DATA, MARKETS
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
    """Parametri NSS per data -- fit GSW SUI PREZZI (Gurkaynak-Sack-Wright 2007, JME).
    Si minimizzano gli errori di prezzo pesati per l'inversa della duration modificata:
    la scelta canonica di GSW, che la motivano perche' approssima l'errore di yield
    (rmse_bp riportato = bp di yield equivalente, confrontabile).

    Input: prezzi clean PX_MID dei nominali (px_nom_{mkt}, scaricati solo dove la curva
    la fittiamo noi), dirty = clean + rateo ACT/ACT ICMA, sconto dal REGOLAMENTO
    (coerente col C-esatto), cedole generate all'indietro dalla scadenza con CPN e
    CPN_FREQ dell'anagrafica. Esclusi bond con vita residua <3 mesi o >50 anni (GSW).
    Duration modificata dallo YTM osservato; fallback = maturita' (zero-coupon approx).
    Cache incrementale su nss_{mkt}.parquet."""
    path = CACHE / f"nss_{mkt}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    todo = dates if old is None else dates.difference(old.index)
    if len(todo):
        px_path = CACHE / f"px_nom_{mkt}.parquet"
        if not px_path.exists():
            raise RuntimeError(
                f"px_nom_{mkt}.parquet assente: il fit della curva e' sui PREZZI (GSW). "
                f"Lancia il 02 (fase NOMINALI) per scaricare PX_MID dei nominali {mkt}.")
        px_nom = pd.read_parquet(px_path)
        px_nom.index = pd.to_datetime(px_nom.index)
        m = MARKETS[mkt]
        hol = m.holidays
        n_set = getattr(m, "settle_days", 2)
        mats = pd.to_datetime(ref_nom["MATURITY"])
        cpns = pd.to_numeric(ref_nom["CPN"], errors="coerce")
        freqs = pd.to_numeric(ref_nom.get("CPN_FREQ"), errors="coerce").fillna(2)
        day_bonds = {}
        for dt in todo:
            if dt not in px_nom.index:
                continue
            p_row = px_nom.loc[dt].dropna()
            y_row = ytm.loc[dt].dropna() if dt in ytm.index else pd.Series(dtype=float)
            settle = pd.Timestamp(settlement(dt.date(), n_set, hol))
            cf_t, cf_a, po, dm = [], [], [], []
            for isin, p_clean in p_row.items():
                if isin not in mats.index or pd.isna(cpns.get(isin)):
                    continue
                mat = mats[isin]
                tau_m = (mat - settle).days / 365.25
                if not (0.25 < tau_m < 50):          # GSW: via il brevissimo e l'oltre-50y
                    continue
                fcd = ref_nom.at[isin, "FIRST_CPN_DT"] if "FIRST_CPN_DT" in ref_nom.columns else None
                if pd.notna(fcd) and settle < pd.Timestamp(fcd):
                    continue    # primo periodo cedolare irregolare: schedule/rateo generati
                                # all'indietro sarebbero errati -> escluso dal fit (prassi std)
                freq = int(freqs.get(isin, 2)) or 2
                step = max(1, round(12 / freq))
                dts = [mat]                           # cedole all'indietro dalla scadenza
                while dts[-1] > settle:
                    dts.append(dts[-1] - pd.DateOffset(months=step))
                nxt = sorted(d for d in dts if d > settle)
                if not nxt:
                    continue
                last = nxt[0] - pd.DateOffset(months=step)
                ced = float(cpns[isin]) / freq
                den = (nxt[0] - last).days
                acc = ced * (settle - last).days / den if den > 0 else 0.0
                dirty = float(p_clean) + acc          # ACT/ACT ICMA
                taus = np.array([(d - settle).days / 365.25 for d in nxt])
                amts = np.full(len(nxt), ced)
                amts[-1] += 100.0
                # PESO GSW/BIS-25: 1/duration, dove 'duration' e' la MACAULAY duration
                # (l'elasticita' del prezzo rispetto a (1+y) -- BIS 25, nota 2), calcolata
                # scontando i cashflow allo YTM osservato e pesando i tempi per il valore
                # presente. Denominatore = DIRTY osservato (il prezzo a cui il bond scambia),
                # coerente col prezzo fittato. Macaulay (non modified): niente fattore di
                # compounding, quindi nessuna convenzione arbitraria. Fallback = maturita'
                # (proxy zero-coupon) quando lo YTM non e' disponibile.
                y = float(y_row.get(isin, np.nan))
                if np.isfinite(y):
                    dfs = (1.0 + y / 100.0) ** (-taus)
                    dmac = float(np.sum(taus * amts * dfs)) / float(np.sum(amts * dfs))
                else:
                    dmac = tau_m
                cf_t.append(taus); cf_a.append(amts); po.append(dirty); dm.append(dmac)
            if len(po) >= 4:
                day_bonds[dt] = (cf_t, cf_a, np.array(po), np.array(dm))
        if day_bonds:
            fit = curves.fit_nss_prices_panel(day_bonds)
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
            raw = curves.buba_zero_panel(ZERO_GRID)     # puo' tornare ndarray (ann_to_cc)
            if isinstance(raw, pd.DataFrame):
                return raw
            par = pd.read_parquet(CACHE / "buba_params.parquet")
            return pd.DataFrame(np.asarray(raw), index=pd.to_datetime(par.index),
                                columns=[float(t) for t in ZERO_GRID]).sort_index()
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
def build_market(mkt: str, methods=("nearest", "c_exact"),
                 is_total: bool = True, floor: bool | None = None,
                 uk_new_only: bool = False,
                 with_real: bool = False, with_floor_value: bool = False,
                 exclude_tail: bool = False, ytm_convention: str = "annual",
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
    ref_n = ref_n[ref_n["mkt"] == nom_mkt][["MATURITY", "FIRST_SETTLE_DT",
                                            "AMT_OUTSTANDING", "CPN", "CPN_FREQ",
                                            "FIRST_CPN_DT", "ISSUE_DT"]]
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
        # inizio ILS: la prima data con abbastanza tenor da proiettare (non un dato isolato
        # anteriore). Con >=4 tenor la curva forward e' costruibile; sotto, la proiezione
        # dell'inflazione fallisce sulle cedole lontane (KeyError sulla griglia).
        ils_ok = ils.notna().sum(axis=1) >= 4
        ils_start = ils.index[ils_ok].min() if ils_ok.any() else ils.dropna(how="all").index.min()
        dates = dates[dates >= ils_start]
    zero = None
    if "c_exact" in methods:
        zero = build_zero_panel(nom_mkt, ref_n, ytm, dates)

    cols = list(bonds)
    panels = {k: pd.DataFrame(np.nan, index=dates, columns=cols)
              for k in ["basis_nearest", "basis_cexact",
                        "totalytm", "ynom_nearest", "mismatch"]}
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
                panels["basis_nearest"].at[asof_ts, isin] = row["basis_nearest"]
                panels["totalytm"].at[asof_ts, isin] = row["total_ytm"]
                panels["ynom_nearest"].at[asof_ts, isin] = row["ynom_nearest"]
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
    done = panels["basis_nearest"].notna().sum().sum()
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


