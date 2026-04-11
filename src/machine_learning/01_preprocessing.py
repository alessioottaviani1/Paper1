"""
================================================================================
01_aen_preprocessing.py - Preprocessing for Adaptive Elastic-Net
================================================================================
Steps (per strategy):
1. Load factors and strategy returns
2. Align on common dates (inner-join)
3. FLAG highly correlated pairs for manual inspection
4. Apply manual exclusions (FACTORS_TO_EXCLUDE)
5. [If SIS_BIC] Sure Independence Screening — keep top d_n factors by
   marginal |corr(Xⱼ, y)| (paper Section 5)
6. Center predictors and response (paper p. 1733)
7. Standardize predictors to L2-norm = 1 (paper p. 1735)
8. Save clean (y, X) datasets

References:
    Zou & Zhang (2009), Section 5: SIS + AEN for high-dimensional settings.
    Fan & Lv (2008), "Sure Independence Screening for Ultrahigh Dimensional
        Feature Space", JRSS-B.

Author:      Alessio Ottaviani
Institution: EDHEC Business School – PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aen_config)

FACTORS_PATH          = aen_config.FACTORS_PATH
STRATEGIES            = aen_config.STRATEGIES
FACTORS_END_DATE      = aen_config.FACTORS_END_DATE
CORRELATION_THRESHOLD = aen_config.CORRELATION_THRESHOLD
FACTORS_TO_EXCLUDE    = aen_config.FACTORS_TO_EXCLUDE
AEN_TUNING_CRITERION  = aen_config.AEN_TUNING_CRITERION
SIS_D_N_RULE          = aen_config.SIS_D_N_RULE
get_aen_output_dir    = aen_config.get_aen_output_dir
get_strategy_aen_dir  = aen_config.get_strategy_aen_dir


def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


def load_strategy_returns(strategy_path: Path) -> pd.Series:
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    if 'index_return' not in daily_df.columns:
        raise ValueError(f"Column 'index_return' not found in {strategy_path}")
    daily_returns = daily_df['index_return'].dropna()
    monthly_returns = daily_returns.resample('ME').apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    monthly_returns = monthly_returns.dropna()
    monthly_returns.name = 'Strategy_Return'
    return monthly_returns


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def find_correlated_pairs(X, threshold):
    corr_matrix = X.corr()
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr_matrix.iloc[i, j]
            if abs(rho) > threshold:
                pairs.append({'factor_1': cols[i], 'factor_2': cols[j],
                              'correlation': round(rho, 6)})
    df_pairs = pd.DataFrame(pairs)
    if len(df_pairs) > 0:
        df_pairs = df_pairs.sort_values('correlation', key=abs, ascending=False)
        df_pairs = df_pairs.reset_index(drop=True)
    return df_pairs


def build_correlation_report(X, y, threshold):
    pairs = find_correlated_pairs(X, threshold)
    if len(pairs) == 0:
        return pairs
    all_factors = set(pairs['factor_1'].tolist() + pairs['factor_2'].tolist())
    corr_with_y = {f: round(abs(X[f].corr(y)), 6) for f in all_factors}
    pairs['|corr(f1, y)|'] = pairs['factor_1'].map(corr_with_y)
    pairs['|corr(f2, y)|'] = pairs['factor_2'].map(corr_with_y)
    pairs['suggestion_drop'] = pairs.apply(
        lambda row: row['factor_1'] if row['|corr(f1, y)|'] < row['|corr(f2, y)|']
        else row['factor_2'], axis=1)
    return pairs


def print_correlation_report(report, strategy_name):
    if len(report) == 0:
        print(f"\n   ✅ No pairs with |ρ| > threshold for {strategy_name}")
        return
    print(f"\n   {'─' * 74}")
    print(f"   CORRELATED PAIRS for {strategy_name}  (|ρ| > {CORRELATION_THRESHOLD})")
    print(f"   {'─' * 74}")
    print(f"   {'#':<4} {'Factor 1':<20} {'Factor 2':<20} {'ρ':>8}"
          f"  {'|r(f1,y)|':>10} {'|r(f2,y)|':>10}")
    print(f"   {'─' * 74}")
    for idx, row in report.iterrows():
        print(f"   {idx+1:<4} {row['factor_1']:<20} {row['factor_2']:<20}"
              f" {row['correlation']:>8.4f}"
              f"  {row['|corr(f1, y)|']:>10.4f}"
              f"  {row['|corr(f2, y)|']:>10.4f}")
    print(f"   {'─' * 74}")
    suggested = report['suggestion_drop'].unique().tolist()
    print(f"   Suggestion: drop {suggested}")
    print(f"   ⚠️  Populate FACTORS_TO_EXCLUDE in 00_aen_config.py manually.")


# ============================================================================
# SIS — Sure Independence Screening
# ============================================================================

def compute_sis_d_n(T, rule):
    """
    Compute SIS threshold d_n.

    Paper Section 5: d_n = ⌊T / log(T)⌋
    Or manual override with an integer.
    """
    if rule == "paper":
        d_n = int(np.floor(T / np.log(T)))
        return d_n
    elif isinstance(rule, int):
        return rule
    else:
        raise ValueError(f"SIS_D_N_RULE must be 'paper' or int, got: {rule}")


def apply_sis(X, y, d_n):
    """
    Sure Independence Screening: keep the d_n factors with highest
    marginal |corr(Xⱼ, y)|.

    Reference: Fan & Lv (2008), Zou & Zhang (2009) Section 5.

    Args:
        X:   factor DataFrame (T × p)
        y:   response Series (T,)
        d_n: number of factors to retain

    Returns:
        (X_screened, sis_report)
        X_screened: DataFrame (T × d_n)
        sis_report: DataFrame with all factors ranked by |corr|
    """
    # Marginal correlations
    marginal_corr = X.corrwith(y).abs()
    marginal_corr = marginal_corr.sort_values(ascending=False)

    # Build report
    sis_report = pd.DataFrame({
        'factor': marginal_corr.index,
        'abs_corr_with_y': marginal_corr.values,
        'rank': range(1, len(marginal_corr) + 1),
        'retained': [i < d_n for i in range(len(marginal_corr))]
    })

    # Keep top d_n
    retained_factors = marginal_corr.head(d_n).index.tolist()
    X_screened = X[retained_factors]

    return X_screened, sis_report


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("AEN PREPROCESSING PIPELINE")
    aen_config.print_config_summary()

    use_sis = (AEN_TUNING_CRITERION == "SIS_BIC")

    # ── STEP 1: Load factors ───────────────────────────────────────────────
    print_header("STEP 1: Load Factors", "-")
    if not FACTORS_PATH.exists():
        print(f"❌ ERROR: Factors file not found: {FACTORS_PATH}")
        return
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]
    print(f"   ✅ Loaded: {len(all_factors.columns)} factors, {len(all_factors)} months")
    print(f"   📅 {all_factors.index.min().strftime('%Y-%m-%d')}"
          f" → {all_factors.index.max().strftime('%Y-%m-%d')}")

    # ── STEP 2: Load and align ─────────────────────────────────────────────
    print_header("STEP 2: Load Strategy Returns & Align", "-")
    strategy_datasets = {}
    strategy_info = {}

    for strategy_name, strategy_path in STRATEGIES.items():
        print(f"\n   📂 {strategy_name}:")
        if not strategy_path.exists():
            print(f"      ❌ File not found: {strategy_path}")
            continue
        try:
            returns = load_strategy_returns(strategy_path)
            returns = returns[returns.index <= factors_end]
            common_dates = returns.index.intersection(all_factors.index)
            y_raw = returns.loc[common_dates]
            X_raw = all_factors.loc[common_dates]

            nan_counts = X_raw.isna().sum()
            factors_with_nan = nan_counts[nan_counts > 0].index.tolist()
            if factors_with_nan:
                print(f"      ⚠️  Dropping {len(factors_with_nan)} factors with NaN:")
                for f in factors_with_nan[:10]:
                    print(f"         - {f} ({int(nan_counts[f])} NaN)")
                if len(factors_with_nan) > 10:
                    print(f"         ... and {len(factors_with_nan) - 10} more")
                X_raw = X_raw.drop(columns=factors_with_nan)

            mask = ~(X_raw.isna().any(axis=1) | y_raw.isna())
            X_raw, y_raw = X_raw[mask], y_raw[mask]
            T, p = len(y_raw), len(X_raw.columns)
            print(f"      ✅ Aligned: T = {T}, p = {p}")
            print(f"      📅 {y_raw.index.min().strftime('%Y-%m-%d')}"
                  f" → {y_raw.index.max().strftime('%Y-%m-%d')}")
            print(f"      T/p ratio: {T/p:.2f}")

            strategy_datasets[strategy_name] = {
                'y_raw': y_raw.copy(), 'X_raw': X_raw.copy()
            }
            strategy_info[strategy_name] = {
                'start': y_raw.index.min().strftime('%Y-%m-%d'),
                'end': y_raw.index.max().strftime('%Y-%m-%d'),
                'T': T, 'p_initial': p
            }
        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback; traceback.print_exc()

    if not strategy_datasets:
        print("\n❌ No strategies loaded.")
        return


    all_correlation_reports = {}
    # ── STEP 4: Manual exclusions ──────────────────────────────────────────
    print_header("STEP 4: Apply Manual Factor Exclusions", "-")
    if not FACTORS_TO_EXCLUDE:
        print("\n   ⚠️  FACTORS_TO_EXCLUDE is empty. Proceeding without.")
    else:
        print(f"\n   Excluding: {FACTORS_TO_EXCLUDE}")

    for sn, data in strategy_datasets.items():
        cols_to_drop = [c for c in FACTORS_TO_EXCLUDE if c in data['X_raw'].columns]
        if cols_to_drop:
            data['X_after_exclusion'] = data['X_raw'].drop(columns=cols_to_drop)
            print(f"   {sn}: dropped {len(cols_to_drop)} → p = {len(data['X_after_exclusion'].columns)}")
        else:
            data['X_after_exclusion'] = data['X_raw'].copy()

    # ── STEP 3 (moved): Correlation report AFTER exclusions ────────────────
    print_header("STEP 3: Correlation Analysis (after exclusions)", "-")
    print(f"\n   Threshold: |ρ| > {CORRELATION_THRESHOLD}")
    for sn, data in strategy_datasets.items():
        report = build_correlation_report(
            data['X_after_exclusion'], data['y_raw'], CORRELATION_THRESHOLD
        )
        all_correlation_reports[sn] = report
        print_correlation_report(report, sn)

    # ── STEP 5 (conditional): SIS ──────────────────────────────────────────
    if use_sis:
        print_header("STEP 5: Sure Independence Screening (SIS)", "-")
        print(f"   Active because AEN_TUNING_CRITERION = 'SIS_BIC'")
        print(f"   SIS_D_N_RULE = {SIS_D_N_RULE}")

        for sn, data in strategy_datasets.items():
            X = data['X_after_exclusion']
            y = data['y_raw']
            T = len(y)
            p = len(X.columns)

            d_n = compute_sis_d_n(T, SIS_D_N_RULE)
            d_n = min(d_n, p)  # can't retain more than available

            print(f"\n   {sn}: T = {T}, p = {p}")
            print(f"      d_n = ⌊{T}/log({T})⌋ = ⌊{T}/{np.log(T):.2f}⌋ = {d_n}")

            X_screened, sis_report = apply_sis(X, y, d_n)

            print(f"      Retained {len(X_screened.columns)} / {p} factors")
            print(f"      Top 5 by |corr(Xⱼ, y)|:")
            for _, row in sis_report.head(5).iterrows():
                marker = "✅" if row['retained'] else "  "
                print(f"         {marker} {row['factor']:<25} |ρ| = {row['abs_corr_with_y']:.4f}")

            # Cutoff
            if d_n < p:
                cutoff_corr = sis_report.iloc[d_n - 1]['abs_corr_with_y']
                print(f"      Cutoff |corr|: {cutoff_corr:.4f}")

            data['X_after_sis'] = X_screened

            # Save SIS report
            strategy_dir = get_strategy_aen_dir(sn)
            sis_report.to_csv(strategy_dir / "sis_report.csv", index=False)
            print(f"      💾 sis_report.csv")

            strategy_info[sn]['p_after_sis'] = len(X_screened.columns)
            strategy_info[sn]['sis_d_n'] = d_n
    else:
        print_header("STEP 5: SIS — SKIPPED (not SIS_BIC)", "-")
        for sn, data in strategy_datasets.items():
            data['X_after_sis'] = data['X_after_exclusion']

    # ── STEP 6: Center & Standardize (L2-norm = 1) ────────────────────────
    print_header("STEP 6: Center & Standardize (L2-norm = 1)", "-")

    for sn, data in strategy_datasets.items():
        X = data['X_after_sis']
        y = data['y_raw']
        T, p = len(y), len(X.columns)

        y_mean = y.mean()
        y_centered = y - y_mean

        X_mean = X.mean()
        X_centered = X - X_mean

        # Drop zero-variance
        zero_std = X_centered.std()
        zero_cols = zero_std[zero_std == 0].index.tolist()
        if zero_cols:
            print(f"\n   ⚠️  {sn}: dropping {len(zero_cols)} zero-variance factors")
            X_centered = X_centered.drop(columns=zero_cols)
            X_mean = X_mean.drop(zero_cols)
            p = len(X_centered.columns)

        # L2-norm standardization
        X_l2_norms = np.sqrt((X_centered ** 2).sum())
        X_standardized = X_centered / X_l2_norms

        norms_check = np.sqrt((X_standardized ** 2).sum())
        assert np.allclose(norms_check, 1.0, atol=1e-10), \
            f"L2-norm check failed for {sn}"

        print(f"\n   {sn}: T = {T}, p = {p}")
        print(f"      y_mean = {y_mean:.4f}  (subtracted)")
        print(f"      ||Xⱼ||₂ = 1 for all j  ✅")

        data['y'] = y_centered
        data['X'] = X_standardized
        data['y_mean'] = float(y_mean)
        data['X_mean'] = X_mean.to_dict()
        data['X_l2_norms'] = X_l2_norms.to_dict()
        data['T'] = T
        data['p'] = p
        data['factor_names'] = X_standardized.columns.tolist()
        strategy_info[sn]['p_final'] = p

    # ── STEP 7: Save ──────────────────────────────────────────────────────
    print_header("STEP 7: Save Outputs", "-")
    aen_output_dir = get_aen_output_dir()

    for sn, data in strategy_datasets.items():
        strategy_dir = get_strategy_aen_dir(sn)
        data['y'].to_frame(name='y').to_parquet(strategy_dir / "y_centered.parquet")
        data['X'].to_parquet(strategy_dir / "X_standardized.parquet")

        std_params = {
            'y_mean': data['y_mean'],
            'X_mean': data['X_mean'],
            'X_l2_norms': data['X_l2_norms'],
            'factor_names': data['factor_names'],
            'T': data['T'], 'p': data['p'],
            'standardization': 'L2-norm = 1 (paper p. 1735)',
            'tuning_criterion': AEN_TUNING_CRITERION,
            'sis_applied': use_sis
        }
        with open(strategy_dir / "standardization_params.json", 'w') as f:
            json.dump(std_params, f, indent=2, default=str)

        print(f"\n   {sn}:")
        print(f"      💾 y_centered.parquet  (T = {data['T']})")
        print(f"      💾 X_standardized.parquet  (T × p = {data['T']} × {data['p']})")
        print(f"      💾 standardization_params.json")

    for sn, report in all_correlation_reports.items():
        if len(report) > 0:
            strategy_dir = get_strategy_aen_dir(sn)
            report.to_csv(strategy_dir / "correlation_report.csv", index=False)

    report_global = {
        'config': {
            'tuning_criterion': AEN_TUNING_CRITERION,
            'correlation_threshold': CORRELATION_THRESHOLD,
            'factors_to_exclude': FACTORS_TO_EXCLUDE,
            'sis_applied': use_sis,
            'sis_d_n_rule': str(SIS_D_N_RULE) if use_sis else None,
        },
        'strategies': strategy_info
    }
    with open(aen_output_dir / "aen_preprocessing_report.json", 'w') as f:
        json.dump(report_global, f, indent=2, default=str)
    print(f"\n   💾 {aen_output_dir / 'aen_preprocessing_report.json'}")

    # ── SUMMARY ────────────────────────────────────────────────────────────
    print_header("SUMMARY")
    p_col = 'p_after_sis' if use_sis else 'p_final'
    print(f"\n   Criterion: {AEN_TUNING_CRITERION}")
    print(f"   {'Strategy':<20} {'Period':<28} {'T':>5} {'p':>5} {'T/p':>6}")
    print(f"   {'─' * 66}")
    for sn, data in strategy_datasets.items():
        info = strategy_info[sn]
        period = f"{info['start']} → {info['end']}"
        T, p = data['T'], data['p']
        print(f"   {sn:<20} {period:<28} {T:>5} {p:>5} {T/p:>6.2f}")

    print(f"\n{'=' * 80}")
    print(f"✅ AEN PREPROCESSING COMPLETE (criterion: {AEN_TUNING_CRITERION})")
    print(f"{'=' * 80}")
    print(f"\n   🎯 Next: python src/aen/02_aen_estimation.py")


if __name__ == "__main__":
    main()