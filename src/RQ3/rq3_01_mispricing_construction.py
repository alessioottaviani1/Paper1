"""
================================================================================
rq3_01_mispricing_construction.py — Costruzione serie mispricing e regimi
================================================================================
RQ3: Slow-moving capital (Duffie 2010)

  1. Carica returns giornalieri → resampla a mensile
  2. Carica livelli di basis (market o portfolio)
     — BTP Italia: media cross-sectional con filtro maturity
     — CDS-Bond: 3 metodi selezionabili (top_n_negative, threshold_only, top_n_all)
     — iTraxx: media delle 4 on-the-run
  3. Costruisce m_{i,t} con trasformazioni corrette (negate_negative_only per CDS)
  4. Carica stress proxy e definisce regimi
  5. Salva dati + plot

Author: Alessio Ottaviani
Institution: EDHEC Business School — PhD Thesis
================================================================================
"""

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_00_config import *

print("=" * 72)
print("RQ3 — FILE 1: MISPRICING CONSTRUCTION & REGIME DEFINITION")
print("=" * 72)
print_config_summary()


# ============================================================================
# STEP 1: CARICA RETURNS GIORNALIERI E RESAMPLA A MENSILE
# ============================================================================

print("\n" + "=" * 72)
print("STEP 1: Caricamento strategy returns")
print("=" * 72)

returns_daily = {}
returns_monthly = {}

for name, path in STRATEGY_INDEX_FILES.items():
    print(f"\n📂 {STRATEGY_LABELS[name]}: {path.name}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    if 'index_return' in df.columns:
        ret = df['index_return'].copy()
    elif 'index_return_ew' in df.columns:
        ret = df['index_return_ew'].copy()
    else:
        raise ValueError(f"Colonna return non trovata in {path.name}: {df.columns.tolist()}")
    
    ret = ret.dropna()
    returns_daily[name] = ret
    
    # Resample a mensile: somma dei daily returns
    ret_monthly = ret.resample('ME').apply(
        lambda x: ((1 + x/100).prod() - 1) * 100)
    # Filtra mesi con meno di 10 giorni di trading (non mesi con return=0!)
    days_per_month = ret.resample('ME').count()
    ret_monthly = ret_monthly[days_per_month >= 10]
    # Burn-in: skip first N months (noisy returns from few active trades)
    if BURN_IN_MONTHS > 0:
        ret_monthly = ret_monthly.iloc[BURN_IN_MONTHS:]
    returns_monthly[name] = ret_monthly
    
    print(f"   Daily:   {len(ret)} obs, {ret.index.min().strftime('%Y-%m-%d')} → {ret.index.max().strftime('%Y-%m-%d')}")
    print(f"   Monthly: {len(ret_monthly)} obs, {ret_monthly.index.min().strftime('%Y-%m')} → {ret_monthly.index.max().strftime('%Y-%m')}")

df_returns_monthly = pd.DataFrame(returns_monthly)
overlap_mask = df_returns_monthly.notna().all(axis=1)
df_returns_overlap = df_returns_monthly[overlap_mask].copy()

print(f"\n{'='*50}")
print(f"📊 PERIODO OVERLAPPING (tutte e 3 attive):")
print(f"   {df_returns_overlap.index.min().strftime('%Y-%m')} → {df_returns_overlap.index.max().strftime('%Y-%m')}")
print(f"   T = {len(df_returns_overlap)} osservazioni mensili")
print(f"{'='*50}")


# ============================================================================
# STEP 2: CARICA LIVELLI DI BASIS
# ============================================================================

print("\n" + "=" * 72)
print(f"STEP 2: Caricamento livelli di basis — approccio: {BASIS_APPROACH.upper()}")
print("=" * 72)

basis_daily = {}

if BASIS_APPROACH == "market":

    # ==================================================================
    # BTP ITALIA — media cross-sectional con filtro maturity
    # ==================================================================
    print(f"\n📂 {STRATEGY_LABELS['btp_italia']}...")
    cfg = RAW_BASIS_FILES["btp_italia"]
    btp_basis_wide = pd.read_parquet(cfg["path"])
    basis_cols = [c for c in btp_basis_wide.columns if c.endswith(cfg["suffix"])]
    
    # Carica maturity dates dal file Excel
    # Il file ha colonne: Date, ISIN1_Basis, ISIN1_DV01, ISIN2_Basis, ...
    # Le maturity si inferiscono dall'ultima data con dato valido per ogni ISIN
    # (il bond smette di quotare quando scade)
    isin_last_date = {}
    for col in basis_cols:
        isin = col.replace(cfg["suffix"], "")
        valid = btp_basis_wide[col].dropna()
        if len(valid) > 0:
            isin_last_date[isin] = valid.index.max()
    
    # Per ogni giorno, escludi ISIN con meno di BTP_MIN_MONTHS_TO_MATURITY mesi
    # a scadenza (approssimata come l'ultima data con dato valido)
    from dateutil.relativedelta import relativedelta
    
    n_included_per_day = []
    btp_filtered_mean = pd.Series(index=btp_basis_wide.index, dtype=float)
    
    for date in btp_basis_wide.index:
        min_maturity_date = date + relativedelta(months=BTP_MIN_MONTHS_TO_MATURITY)
        
        eligible_cols = []
        for col in basis_cols:
            isin = col.replace(cfg["suffix"], "")
            last_dt = isin_last_date.get(isin)
            if last_dt is not None and last_dt >= min_maturity_date:
                val = btp_basis_wide.loc[date, col]
                if pd.notna(val):
                    eligible_cols.append(col)
        
        if eligible_cols:
            btp_filtered_mean[date] = btp_basis_wide.loc[date, eligible_cols].mean()
            n_included_per_day.append(len(eligible_cols))
        else:
            n_included_per_day.append(0)
    
    btp_filtered_mean = btp_filtered_mean.dropna()
    basis_daily["btp_italia"] = btp_filtered_mean
    
    avg_bonds = np.mean(n_included_per_day) if n_included_per_day else 0
    print(f"   {len(basis_cols)} ISIN totali, media {avg_bonds:.1f} eleggibili/giorno "
          f"(filtro: maturity >= {BTP_MIN_MONTHS_TO_MATURITY} mesi)")
    print(f"   {len(btp_filtered_mean)} giorni, basis medio: {btp_filtered_mean.mean():.2f} bps")

    # ==================================================================
    # CDS-BOND BASIS — 3 metodi selezionabili
    # ==================================================================
    print(f"\n📂 {STRATEGY_LABELS['cds_bond_basis']}...")
    print(f"   Metodo: {CDS_BOND_BASIS_METHOD}")
    cfg = RAW_BASIS_FILES["cds_bond_basis"]
    cds_long = pd.read_parquet(cfg["path"])
    cds_long[cfg["date_col"]] = pd.to_datetime(cds_long[cfg["date_col"]])
    
    all_dates = cds_long[cfg["date_col"]].unique()
    all_dates = np.sort(all_dates)
    
    cds_daily = pd.Series(index=pd.DatetimeIndex(all_dates), dtype=float)
    n_bonds_used = []
    
    for date in all_dates:
        day_data = cds_long[cds_long[cfg["date_col"]] == date][cfg["basis_col"]].dropna()
        
        if len(day_data) == 0:
            n_bonds_used.append(0)
            continue
        
        if CDS_BOND_BASIS_METHOD == "top_n_negative":
            # Opzione 4: filtra basis < 0, media delle N più negative
            negative = day_data[day_data < 0].sort_values()
            if len(negative) > 0:
                selected = negative.head(CDS_BOND_TOP_N)
                cds_daily[date] = selected.mean()
                n_bonds_used.append(len(selected))
            else:
                n_bonds_used.append(0)
        
        elif CDS_BOND_BASIS_METHOD == "threshold_only":
            # Opzione 2: filtra basis < soglia, media
            below_thresh = day_data[day_data < CDS_BOND_ENTRY_THRESHOLD]
            if len(below_thresh) > 0:
                cds_daily[date] = below_thresh.mean()
                n_bonds_used.append(len(below_thresh))
            else:
                n_bonds_used.append(0)
        
        elif CDS_BOND_BASIS_METHOD == "top_n_all":
            # Opzione 3: ordina tutte, media delle N più negative
            sorted_basis = day_data.sort_values()
            selected = sorted_basis.head(CDS_BOND_TOP_N)
            cds_daily[date] = selected.mean()
            n_bonds_used.append(len(selected))
        
        else:
            raise ValueError(f"CDS_BOND_BASIS_METHOD non valido: {CDS_BOND_BASIS_METHOD}")
    
    cds_daily = cds_daily.dropna()
    basis_daily["cds_bond_basis"] = cds_daily
    
    # Mediana cross-sectional di TUTTO l'universo (per summary statistics table)
    cds_daily_median = cds_long.groupby(cfg["date_col"])[cfg["basis_col"]].median()
    cds_daily_median.index = pd.to_datetime(cds_daily_median.index)
    cds_daily_median = cds_daily_median.sort_index().dropna()
    basis_daily["cds_bond_basis_median"] = cds_daily_median
    print(f"   [Median full universe: {len(cds_daily_median)} days, "
          f"median={cds_daily_median.median():.2f} bps]")
    
    avg_bonds = np.mean(n_bonds_used) if n_bonds_used else 0
    pct_nan = (1 - len(cds_daily) / len(all_dates)) * 100
    print(f"   {cds_long['ISIN'].nunique()} ISIN totali, media {avg_bonds:.1f} bond usati/giorno")
    print(f"   {len(cds_daily)} giorni con dato ({pct_nan:.1f}% NaN)")
    print(f"   Basis medio: {cds_daily.mean():.2f} bps "
          f"(range: {cds_daily.min():.1f} / {cds_daily.max():.1f})")

    # ==================================================================
    # ITRAXX COMBINED — media delle 4 on-the-run
    # ==================================================================
    print(f"\n📂 {STRATEGY_LABELS['itraxx_combined']}...")
    cfg = RAW_BASIS_FILES["itraxx_combined"]
    itraxx_all_means = []
    
    for idx_name, fpath in cfg["paths"].items():
        if fpath.exists():
            itrx_wide = pd.read_parquet(fpath)
            b_cols = [c for c in itrx_wide.columns if c.endswith(cfg["suffix"])]
            
            series_nums = {}
            for c in b_cols:
                match = re.search(r'Ser(\d+)', c)
                if match:
                    series_nums[c] = int(match.group(1))
            
            b_cols_sorted = sorted(b_cols, key=lambda c: series_nums.get(c, 0), reverse=True)
            basis_matrix = itrx_wide[b_cols_sorted]
            otr_basis = basis_matrix.bfill(axis=1).iloc[:, 0].dropna()
            otr_basis = otr_basis.groupby(otr_basis.index).mean()
            otr_basis.name = idx_name
            itraxx_all_means.append(otr_basis)
            print(f"   {idx_name}: {len(b_cols)} serie, OTR→ {len(otr_basis)} giorni, "
                  f"media={otr_basis.mean():.2f} bps")
        else:
            print(f"   ⚠️ {idx_name}: file non trovato ({fpath})")
    
    if itraxx_all_means:
        itraxx_combined_df = pd.concat(itraxx_all_means, axis=1)
        itraxx_cs_mean = itraxx_combined_df.mean(axis=1).dropna()
        basis_daily["itraxx_combined"] = itraxx_cs_mean
        print(f"   Combined: {len(itraxx_cs_mean)} giorni, media={itraxx_cs_mean.mean():.2f} bps")


elif BASIS_APPROACH == "portfolio":

    for name, path in STRATEGY_TRADES_FILES.items():
        print(f"\n📂 {STRATEGY_LABELS[name]}: {path.name}")
        trades = pd.read_csv(path, parse_dates=['entry_date', 'exit_date'])
        print(f"   Trade totali: {len(trades)}")
        
        all_dates = pd.date_range(
            start=trades['entry_date'].min(),
            end=trades['exit_date'].max(), freq='B'
        )
        
        basis_series = pd.Series(index=all_dates, dtype=float)
        for date in all_dates:
            active = trades[
                (trades['entry_date'] <= date) & (trades['exit_date'] >= date)
            ]
            if len(active) > 0:
                basis_series[date] = active['entry_basis'].mean()
        
        basis_series = basis_series.dropna()
        basis_daily[name] = basis_series
        print(f"   Giorni con trade attivi: {len(basis_series)}")
        print(f"   Basis medio: {basis_series.mean():.2f} bps")

else:
    raise ValueError(f"BASIS_APPROACH non valido: {BASIS_APPROACH}")

print(f"\n{'='*50}")
print(f"   Approccio: {BASIS_APPROACH.upper()}")
for name in STRATEGY_NAMES:
    s = basis_daily[name]
    print(f"   {STRATEGY_LABELS[name]:20s}: {len(s)} giorni, "
          f"{s.index.min().strftime('%Y-%m-%d')} → {s.index.max().strftime('%Y-%m-%d')}")
print(f"{'='*50}")


# ============================================================================
# STEP 3: COSTRUISCI SERIE m_{i,t} (MISPRICING MAGNITUDE)
# ============================================================================

print("\n" + "=" * 72)
print("STEP 3: Costruzione mispricing magnitude m_{i,t}")
print("=" * 72)

mispricing_daily = {}
mispricing_monthly = {}

for name in STRATEGY_NAMES:
    basis = basis_daily[name]
    transform = MISPRICING_TRANSFORM[name]
    
    if transform == "negate_negative_only":
        # CDS-Bond: m = -basis SE basis < 0, altrimenti NaN
        # Così m ≥ 0 sempre, e cresce quando la basis diventa più negativa
        m = basis.copy()
        m[basis >= 0] = np.nan
        m = -m  # ora m ≥ 0
        m = m.dropna()
        print(f"\n   {STRATEGY_LABELS[name]}: m_t = -basis_t (solo basis < 0)")
    elif transform == "negate_clamp_zero":
        # Robustness: m = max(-basis, 0) → nessun NaN, sempre definita
        m = (-basis).clip(lower=0)
        print(f"\n   {STRATEGY_LABELS[name]}: m_t = max(-basis_t, 0) (clamp zero)")
    elif transform == "negate":
        m = -basis
        print(f"\n   {STRATEGY_LABELS[name]}: m_t = -basis_t  (negate)")
    elif transform == "absolute":
        m = basis.abs()
        print(f"\n   {STRATEGY_LABELS[name]}: m_t = |basis_t|  (absolute)")
    else:
        raise ValueError(f"Trasformazione sconosciuta: {transform}")
    
    mispricing_daily[name] = m
    
    # Resample a mensile (media del mese — livello medio di mispricing)
    m_monthly = m.resample('ME').mean()
    m_monthly = m_monthly.dropna()
    # Burn-in: skip first N months (same as returns)
    if BURN_IN_MONTHS > 0:
        m_monthly = m_monthly.iloc[BURN_IN_MONTHS:]
    mispricing_monthly[name] = m_monthly
    
    print(f"   m medio: {m_monthly.mean():.2f} bps, min/max: {m_monthly.min():.2f} / {m_monthly.max():.2f}")
    print(f"   Monthly obs: {len(m_monthly)}")
    # Sanity check
    if transform == "negate_negative_only" and (m_monthly < 0).any():
        print(f"   ⚠️ ERRORE: m_t ha valori negativi! Controllare trasformazione.")
    if transform == "absolute" and (m_monthly < 0).any():
        print(f"   ⚠️ ERRORE: m_t ha valori negativi! Controllare trasformazione.")

df_mispricing_monthly = pd.DataFrame(mispricing_monthly)
df_delta_m = df_mispricing_monthly.diff()

overlap_m = df_mispricing_monthly.dropna().index.intersection(df_returns_overlap.index)
df_mispricing_overlap = df_mispricing_monthly.loc[overlap_m].copy()
df_delta_m_overlap = df_delta_m.loc[overlap_m].copy()

print(f"\n{'='*50}")
print(f"📊 MISPRICING OVERLAP: {len(df_mispricing_overlap)} mesi")
print(f"   Δm overlap: {len(df_delta_m_overlap.dropna())} mesi")
print(f"{'='*50}")

# --- Variante robustness: m = max(-basis, 0) per CDS-Bond ---
# Verifica che i risultati non siano artefatto della sample selection
print(f"\n--- Robustness: variante m=0 (nessun NaN per CDS-Bond) ---")

mispricing_monthly_robust = {}
for name in STRATEGY_NAMES:
    transform_r = MISPRICING_TRANSFORM_ROBUST[name]
    basis = basis_daily[name]
    
    if transform_r == "negate_clamp_zero":
        m = (-basis).clip(lower=0)
    elif transform_r == "absolute":
        m = basis.abs()
    else:
        m = basis.copy()
    
    m_monthly = m.resample('ME').mean().dropna()
    mispricing_monthly_robust[name] = m_monthly

df_mispricing_robust = pd.DataFrame(mispricing_monthly_robust)
overlap_robust = df_mispricing_robust.dropna().index.intersection(df_returns_overlap.index)
df_dm_robust = df_mispricing_robust.loc[overlap_robust].diff()

n_main = len(df_delta_m_overlap.dropna())
n_robust = len(df_dm_robust.dropna())
print(f"   Campione principale (NaN): T={n_main}")
print(f"   Campione robusto (m=0):    T={n_robust}")
print(f"   Differenza: {n_robust - n_main} mesi aggiunti")

# --- Stationarity pre-test (ADF) su Δm ---
print(f"\n--- ADF Stationarity Test su Δm ---")
from statsmodels.tsa.stattools import adfuller

for name in STRATEGY_NAMES:
    dm_series = df_delta_m_overlap[name].dropna()
    if len(dm_series) < 20:
        print(f"   {STRATEGY_LABELS[name]}: troppo pochi dati, skip")
        continue
    adf_result = adfuller(dm_series, autolag='AIC')
    adf_stat, adf_p, adf_lags = adf_result[0], adf_result[1], adf_result[2]
    status = "✅ Stazionario" if adf_p < 0.05 else "⚠️ NON stazionario"
    print(f"   {STRATEGY_LABELS[name]:20s}: ADF={adf_stat:+.3f}, p={adf_p:.4f}, "
          f"lags={adf_lags} → {status}")


# ============================================================================
# STEP 4: CARICA STRESS PROXY (LIVELLI) E DEFINISCI REGIMI
# ============================================================================

print("\n" + "=" * 72)
print("STEP 4: Caricamento stress proxy e definizione regimi")
print("=" * 72)

stress_proxy_daily = {}

# --- iTraxx Main ---
print("\n📂 iTraxx Main 5Y (livelli)...")
_itrx_raw = pd.read_excel(
    TRADABLE_CB_FILE, sheet_name=ITRAXX_MAIN_SHEET,
    skiprows=ITRAXX_MAIN_SKIPROWS, usecols=ITRAXX_MAIN_USECOLS, header=0
)
_itrx_raw.columns = ITRAXX_MAIN_COLNAMES
_itrx_raw["Date"] = pd.to_datetime(_itrx_raw["Date"], errors="coerce")
_itrx_raw = _itrx_raw.dropna(subset=["Date"]).set_index("Date")
stress_proxy_daily["ITRX_MAIN"] = pd.to_numeric(_itrx_raw["ITRX_MAIN"], errors="coerce").dropna()
print(f"   {len(stress_proxy_daily['ITRX_MAIN'])} obs")

# --- iTraxx Crossover ---
print("\n📂 iTraxx Crossover 5Y (livelli)...")
_xover_raw = pd.read_excel(
    TRADABLE_CB_FILE, sheet_name=ITRAXX_XOVER_SHEET,
    skiprows=ITRAXX_XOVER_SKIPROWS, usecols=ITRAXX_XOVER_USECOLS, header=0
)
_xover_raw.columns = ITRAXX_XOVER_COLNAMES
_xover_raw["Date"] = pd.to_datetime(_xover_raw["Date"], errors="coerce")
_xover_raw = _xover_raw.dropna(subset=["Date"]).set_index("Date")
stress_proxy_daily["ITRX_XOVER"] = pd.to_numeric(_xover_raw["ITRX_XOVER"], errors="coerce").dropna()
print(f"   {len(stress_proxy_daily['ITRX_XOVER'])} obs, "
      f"{stress_proxy_daily['ITRX_XOVER'].index.min().strftime('%Y-%m-%d')} → "
      f"{stress_proxy_daily['ITRX_XOVER'].index.max().strftime('%Y-%m-%d')}")

# --- VIX e V2X ---
print("\n📂 VIX e V2X (livelli)...")
_vix_raw = pd.read_excel(
    NONTRADABLE_FILE, sheet_name=VIX_SHEET,
    skiprows=VIX_SKIPROWS, usecols=VIX_USECOLS, header=0
)
_vix_raw.columns = VIX_COLNAMES
_vix_raw["Date"] = pd.to_datetime(_vix_raw["Date"], errors="coerce")
_vix_raw = _vix_raw.dropna(subset=["Date"]).set_index("Date")
stress_proxy_daily["VIX"] = pd.to_numeric(_vix_raw["VIX"], errors="coerce").dropna()
stress_proxy_daily["V2X"] = pd.to_numeric(_vix_raw["V2X"], errors="coerce").dropna()
for pn in ["VIX", "V2X"]:
    s = stress_proxy_daily[pn]
    print(f"   {pn}: {len(s)} obs, media={s.mean():.1f}")

# --- Resample a mensile ---
stress_proxy_monthly = {}
for pn, series in stress_proxy_daily.items():
    stress_proxy_monthly[pn] = series.resample('ME').last().dropna()

# --- DEFINISCI REGIMI ---
print("\n" + "-" * 50)
print("Definizione regimi di stress")
print("-" * 50)

regimes = {}

for proxy_name in ALL_STRESS_PROXIES:
    proxy_monthly = stress_proxy_monthly[proxy_name]
    regimes[proxy_name] = {}
    
    # Manual
    thresholds = MANUAL_THRESHOLDS[proxy_name]
    low_upper = thresholds["low_upper"]
    high_lower = thresholds["high_lower"]
    
    regime_manual = pd.Series("MEDIUM", index=proxy_monthly.index)
    regime_manual[proxy_monthly < low_upper] = "LOW"
    regime_manual[proxy_monthly >= high_lower] = "HIGH"
    regimes[proxy_name]["manual"] = regime_manual
    
    regimes[proxy_name]["manual"] = regime_manual
    
    n_l = (regime_manual == "LOW").sum()
    n_m = (regime_manual == "MEDIUM").sum()
    n_h = (regime_manual == "HIGH").sum()
    print(f"\n   {proxy_name} MANUAL (3-level): LOW={n_l}, MED={n_m}, HIGH={n_h}")
    
    # 2-level regime: NORMAL/HIGH
    if REGIME_2L:
        regime_2l = pd.Series("NORMAL", index=proxy_monthly.index)
        regime_2l[proxy_monthly >= high_lower] = "HIGH"
        regimes[proxy_name]["manual_2l"] = regime_2l
        
        n_n = (regime_2l == "NORMAL").sum()
        n_h2 = (regime_2l == "HIGH").sum()
        print(f"   {proxy_name} MANUAL (2-level): NORMAL={n_n}, HIGH={n_h2}")
    
    # Percentile
    full_sample = proxy_monthly[proxy_monthly.index >= pd.Timestamp(PERCENTILE_START_DATE)]
    pct_low = np.percentile(full_sample.dropna(), PERCENTILE_THRESHOLDS["low_upper"])
    pct_high = np.percentile(full_sample.dropna(), PERCENTILE_THRESHOLDS["high_lower"])
    
    regime_pct = pd.Series("MEDIUM", index=proxy_monthly.index)
    regime_pct[proxy_monthly < pct_low] = "LOW"
    regime_pct[proxy_monthly >= pct_high] = "HIGH"
    regimes[proxy_name]["percentile"] = regime_pct
    
    n_l = (regime_pct == "LOW").sum()
    n_m = (regime_pct == "MEDIUM").sum()
    n_h = (regime_pct == "HIGH").sum()
    print(f"   {proxy_name} PERCENTILE (P{PERCENTILE_THRESHOLDS['low_upper']}={pct_low:.1f}, "
          f"P{PERCENTILE_THRESHOLDS['high_lower']}={pct_high:.1f}): LOW={n_l}, MED={n_m}, HIGH={n_h}")

default_regime = regimes[DEFAULT_STRESS_PROXY][DEFAULT_REGIME_MODE]
print(f"\n📌 REGIME DEFAULT: {DEFAULT_STRESS_PROXY} ({DEFAULT_REGIME_MODE})")


# ============================================================================
# STEP 5: CARICA FATTORI MENSILI PER RESIDUALIZZAZIONE
# ============================================================================

print("\n" + "=" * 72)
print("STEP 5: Caricamento fattori mensili")
print("=" * 72)

factors_monthly = pd.read_parquet(FACTORS_PATH)
print(f"   {len(factors_monthly.columns)} fattori, {len(factors_monthly)} obs")

purge_factors = COMMON_FACTORS_FOR_PURGING if PURGE_FACTOR_SET == "base" else COMMON_FACTORS_EXTENDED
missing = [f for f in purge_factors if f not in factors_monthly.columns]
if missing:
    print(f"   ⚠️ FATTORI MANCANTI: {missing}")
else:
    print(f"   ✅ Fattori purge presenti: {purge_factors}")

# Check PC1 funding factors
missing_pc1 = [f for f in PC1_FUNDING_FACTORS if f not in factors_monthly.columns]
if missing_pc1:
    print(f"   ⚠️ PC1 FUNDING FACTORS MANCANTI: {missing_pc1}")
else:
    print(f"   ✅ PC1 funding factors presenti: {PC1_FUNDING_FACTORS}")

df_purge_factors = factors_monthly[purge_factors].copy()


# ============================================================================
# STEP 6: SALVA DATI PREPARATI
# ============================================================================

print("\n" + "=" * 72)
print("STEP 6: Salvataggio")
print("=" * 72)

df_returns_monthly.to_csv(RQ3_DATA_DIR / "returns_monthly_all.csv")
df_returns_overlap.to_csv(RQ3_DATA_DIR / "returns_monthly_overlap.csv")
df_mispricing_monthly.to_csv(RQ3_DATA_DIR / "mispricing_monthly_all.csv")
df_mispricing_overlap.to_csv(RQ3_DATA_DIR / "mispricing_monthly_overlap.csv")
df_delta_m.to_csv(RQ3_DATA_DIR / "delta_m_monthly_all.csv")
df_delta_m_overlap.to_csv(RQ3_DATA_DIR / "delta_m_monthly_overlap.csv")

df_stress = pd.DataFrame(stress_proxy_monthly)
df_stress.to_csv(RQ3_DATA_DIR / "stress_proxy_monthly.csv")

for proxy_name in ALL_STRESS_PROXIES:
    for mode in ["manual", "percentile"]:
        regimes[proxy_name][mode].to_csv(
            RQ3_DATA_DIR / f"regime_{proxy_name}_{mode}.csv", header=["regime"]
        )
    if REGIME_2L and "manual_2l" in regimes[proxy_name]:
        regimes[proxy_name]["manual_2l"].to_csv(
            RQ3_DATA_DIR / f"regime_{proxy_name}_manual_2l.csv", header=["regime"]
        )

df_purge_factors.to_csv(RQ3_DATA_DIR / "purge_factors_monthly.csv")

master = {
    'returns': df_returns_overlap,
    'mispricing': df_mispricing_overlap,
    'delta_m': df_delta_m_overlap,
    'regime': default_regime,
    'delta_m_robust': df_dm_robust,
}
if REGIME_2L:
    master['regime_2l'] = regimes[DEFAULT_STRESS_PROXY]["manual_2l"]

pd.to_pickle(master, RQ3_DATA_DIR / "rq3_master_data.pkl")

# Salva anche returns mensili completi per pairwise long analysis
pd.to_pickle({
    'returns_all': df_returns_monthly,
    'mispricing_all': df_mispricing_monthly,
    'delta_m_all': df_delta_m,
}, RQ3_DATA_DIR / "rq3_full_data.pkl")

pd.to_pickle(basis_daily, RQ3_DATA_DIR / "basis_daily.pkl")

print(f"💾 Tutti i file salvati in {RQ3_DATA_DIR}")


# ============================================================================
# STEP 7: PLOT
# ============================================================================

print("\n" + "=" * 72)
print("STEP 7: Plot")
print("=" * 72)

plt.style.use(FIGURE_STYLE)

# PLOT A: Livelli mispricing (3 pannelli)
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Mispricing Magnitude $m_{i,t}$ — Livelli Mensili", fontsize=14, fontweight='bold')

for idx, name in enumerate(STRATEGY_NAMES):
    ax = axes[idx]
    m_series = df_mispricing_monthly[name].dropna()
    ax.plot(m_series.index, m_series.values, color=STRATEGY_COLORS[name], linewidth=1.2,
            label=STRATEGY_LABELS[name])
    
    for event_date, event_label in MACRO_EVENTS.items():
        event_dt = pd.Timestamp(event_date)
        if m_series.index.min() <= event_dt <= m_series.index.max():
            ax.axvline(event_dt, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
    
    ax.set_ylabel("$m_{i,t}$ (bps)", fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"A1_mispricing_levels.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 A1_mispricing_levels.{FIGURE_FORMAT}")
plt.close()

# PLOT B: Normalizzati sovrapposti
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Mispricing Magnitude (Normalizzata)", fontsize=13, fontweight='bold')
for name in STRATEGY_NAMES:
    m_series = df_mispricing_overlap[name].dropna()
    m_z = (m_series - m_series.mean()) / m_series.std()
    ax.plot(m_z.index, m_z.values, color=STRATEGY_COLORS[name], linewidth=1.2,
            label=STRATEGY_LABELS[name])
ax.legend(fontsize=10, ncol=3, loc='upper left')
ax.set_ylabel("Z-score $m_{i,t}$")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"A2_mispricing_normalized_overlay.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 A2_mispricing_normalized_overlay.{FIGURE_FORMAT}")
plt.close()

# PLOT C: Stress proxy con soglie
fig, axes = plt.subplots(len(ALL_STRESS_PROXIES), 1,
                         figsize=(14, 3.5 * len(ALL_STRESS_PROXIES)), sharex=True)
fig.suptitle("Stress Proxy — Livelli con Soglie Regime", fontsize=13, fontweight='bold')
for idx, proxy_name in enumerate(ALL_STRESS_PROXIES):
    ax = axes[idx]
    ps = stress_proxy_monthly[proxy_name]
    ax.plot(ps.index, ps.values, color='black', linewidth=0.9, label=proxy_name)
    thresh = MANUAL_THRESHOLDS[proxy_name]
    ax.axhline(thresh["low_upper"], color=REGIME_COLORS["LOW"], linestyle='--', linewidth=0.8)
    ax.axhline(thresh["high_lower"], color=REGIME_COLORS["HIGH"], linestyle='--', linewidth=0.8)
    ax.set_ylabel(proxy_name)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"A4_stress_proxy_levels.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 A4_stress_proxy_levels.{FIGURE_FORMAT}")
plt.close()


# ============================================================================
# STEP 8: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 72)
print("STEP 8: Summary statistics")
print("=" * 72)

summary_rows = []
for name in STRATEGY_NAMES:
    ret = df_returns_overlap[name]
    m = df_mispricing_overlap[name]
    dm = df_delta_m_overlap[name].dropna()
    summary_rows.append({
        'Strategy': STRATEGY_LABELS[name], 'T': len(ret),
        'Return_Mean': ret.mean(), 'Return_Std': ret.std(),
        'Return_Skew': ret.skew(), 'Return_Kurt': ret.kurtosis(),
        'Mispricing_Mean': m.mean(), 'Mispricing_Std': m.std(),
        'Δm_Mean': dm.mean(), 'Δm_Std': dm.std(),
        'Pct_Widening': (dm > 0).mean() * 100,
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(RQ3_TABLES_DIR / "T1_summary_statistics.csv", index=False)

print(f"\n{'Strategy':<20} {'T':>4} {'Ret μ':>8} {'Ret σ':>8} {'m μ':>8} {'m σ':>8} {'%Wid':>6}")
print("-" * 62)
for _, row in df_summary.iterrows():
    print(f"{row['Strategy']:<20} {row['T']:>4.0f} {row['Return_Mean']:>8.2f} "
          f"{row['Return_Std']:>8.2f} {row['Mispricing_Mean']:>8.2f} "
          f"{row['Mispricing_Std']:>8.2f} {row['Pct_Widening']:>5.1f}%")

print(f"\n💾 T1_summary_statistics.csv")


# ============================================================================
# STEP 9: BASIS TIME SERIES PLOT (for paper Section 3)
# ============================================================================
# Market-level basis aggregated across bonds/CDS for each strategy.
# 3 vertical panels, each showing the basis level over time with
# entry thresholds as dashed horizontal lines.
# Output: results/tables/../basis_time_series.pdf (for paper)
# ============================================================================

print("\n" + "=" * 72)
print("STEP 9: Basis Time Series Plot (for paper)")
print("=" * 72)

# --- Output also to results/figures/ for the paper ---
PAPER_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

plot_config = {
    "btp_italia": {
        "label": "BTP Italia Inflation-Linked Basis",
        "color": "#1f77b4",
        "thresholds": [40, -50],  # long entry, short entry
        "threshold_labels": ["Long entry (+40 bps)", "Short entry (−50 bps)"],
        "ylabel": "Basis (bps)",
    },
    "cds_bond_basis": {
        "label": "CDS-Bond Basis",
        "color": "#d62728",
        "thresholds": [-40],  # negative basis entry
        "threshold_labels": ["Entry (−40 bps)"],
        "ylabel": "Basis (bps)",
    },
    "itraxx_combined": {
        "label": "CDS Index Skew (iTraxx Combined)",
        "color": "#2ca02c",
        "thresholds": [10, -10],  # two-sided entry
        "threshold_labels": ["Entry (+10 bps)", "Entry (−10 bps)"],
        "ylabel": "Skew (bps)",
    },
}

for idx, name in enumerate(["btp_italia", "cds_bond_basis", "itraxx_combined"]):
    ax = axes[idx]
    cfg = plot_config[name]

    if name in basis_daily:
        series = basis_daily[name].dropna()
        ax.plot(series.index, series.values, color=cfg["color"],
                linewidth=0.7, alpha=0.85)

        # Zero line
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)

        # Entry thresholds
        threshold_colors = ['#ff7f0e', '#9467bd']  # orange, purple
        for t_idx, (thresh, t_label) in enumerate(
                zip(cfg["thresholds"], cfg["threshold_labels"])):
            ax.axhline(thresh, color=threshold_colors[t_idx % 2],
                       linewidth=1.0, linestyle='--', alpha=0.7, label=t_label)

        ax.set_ylabel(cfg["ylabel"], fontsize=10)
        ax.set_title(cfg["label"], fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.2)

        # Date range annotation
        ax.text(0.01, 0.95,
                f"{series.index.min().strftime('%b %Y')} — {series.index.max().strftime('%b %Y')}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                fontstyle='italic', color='grey')
    else:
        ax.text(0.5, 0.5, f"No data for {name}", transform=ax.transAxes,
                ha='center', fontsize=12, color='red')

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.tight_layout()

# Save for paper (PDF) and for RQ3 figures (original format)
fig.savefig(PAPER_FIGURES_DIR / "basis_time_series.pdf", bbox_inches='tight')
fig.savefig(RQ3_FIGURES_DIR / f"A0_basis_time_series.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()

print(f"📊 basis_time_series.pdf → results/figures/")
print(f"📊 A0_basis_time_series.{FIGURE_FORMAT} → {RQ3_FIGURES_DIR.name}/")

print("\n" + "=" * 72)
print("✅ FILE 1 COMPLETATO")
print("=" * 72)
print(f"\n   Prossimo step → rq3_02_correlation_cowidening.py")