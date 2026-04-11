"""
Script 00: Import All Risk Factors
===================================
Importa tutti i fattori da diversi Excel, resample a mensile, merge.

OUTPUT: all_factors_monthly.parquet
"""

import pandas as pd
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "external" / "factors"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IMPORT FACTORS
# ============================================================================

print("=" * 80)
print("IMPORTING RISK FACTORS")
print("=" * 80)

all_dfs = []

# ----------------------------------------------------------------------------
# FILE 1: Nontradable Risk Factors - CREDIT
# ----------------------------------------------------------------------------

print("\n📂 Nontradable_risk_factors.xlsx - CREDIT")

df = pd.read_excel(
    DATA_DIR / "Nontradable_risk_factors.xlsx",
    sheet_name="CREDIT",
    skiprows=15  # first value at row 16
)

# Keep:
# A (Date_US), D (CREDIT_US),
# G (Date_EU), J (CREDIT_EU),
# M (CRED_SPR_US), O (CRED_SPR_EU)
# 0-indexed: A=0, D=3, G=6, J=9, M=12, O=14
df = df.iloc[:, [0, 3, 6, 9, 12, 14]]

# --- US side (Date in A, value in D) ---
df_us = df.iloc[:, [0, 1]].copy()
df_us.columns = ["Date", "CREDIT_US"]
df_us.set_index("Date", inplace=True)
df_us.index.name = "Date"

credit_us_daily = pd.to_numeric(df_us["CREDIT_US"], errors="coerce")
credit_us_monthly = credit_us_daily.resample("ME").last()

# --- EU block (Date in G, values in J, M, O) ---
df_eu_block = df.iloc[:, [2, 3, 4, 5]].copy()
df_eu_block.columns = ["Date", "CREDIT_EU", "CRED_SPR_US", "CRED_SPR_EU"]
df_eu_block.set_index("Date", inplace=True)
df_eu_block.index.name = "Date"

credit_eu_daily = pd.to_numeric(df_eu_block["CREDIT_EU"], errors="coerce")
cred_spr_us_daily = pd.to_numeric(df_eu_block["CRED_SPR_US"], errors="coerce")
cred_spr_eu_daily = pd.to_numeric(df_eu_block["CRED_SPR_EU"], errors="coerce")

credit_eu_monthly = credit_eu_daily.resample("ME").last()
cred_spr_us_monthly = cred_spr_us_daily.resample("ME").last()
cred_spr_eu_monthly = cred_spr_eu_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "CREDIT_US": credit_us_monthly,
    "CREDIT_EU": credit_eu_monthly,
    "CRED_SPR_US": cred_spr_us_monthly,
    "CRED_SPR_EU": cred_spr_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: CREDIT_US={df_monthly['CREDIT_US'].isna().sum()}, CREDIT_EU={df_monthly['CREDIT_EU'].isna().sum()}, CRED_SPR_US={df_monthly['CRED_SPR_US'].isna().sum()}, CRED_SPR_EU={df_monthly['CRED_SPR_EU'].isna().sum()}")

credit_us_valid = credit_us_monthly.dropna()
credit_eu_valid = credit_eu_monthly.dropna()
cred_spr_us_valid = cred_spr_us_monthly.dropna()
cred_spr_eu_valid = cred_spr_eu_monthly.dropna()

print(f"\n   CREDIT_US (last 2 valid):")
print(f"      {credit_us_valid.index[-2].strftime('%Y-%m-%d')}: {credit_us_valid.iloc[-2]:.4f}")
print(f"      {credit_us_valid.index[-1].strftime('%Y-%m-%d')}: {credit_us_valid.iloc[-1]:.4f}")

print(f"   CREDIT_EU (last 2 valid):")
print(f"      {credit_eu_valid.index[-2].strftime('%Y-%m-%d')}: {credit_eu_valid.iloc[-2]:.4f}")
print(f"      {credit_eu_valid.index[-1].strftime('%Y-%m-%d')}: {credit_eu_valid.iloc[-1]:.4f}")

print(f"   CRED_SPR_US (last 2 valid):")
print(f"      {cred_spr_us_valid.index[-2].strftime('%Y-%m-%d')}: {cred_spr_us_valid.iloc[-2]:.4f}")
print(f"      {cred_spr_us_valid.index[-1].strftime('%Y-%m-%d')}: {cred_spr_us_valid.iloc[-1]:.4f}")

print(f"   CRED_SPR_EU (last 2 valid):")
print(f"      {cred_spr_eu_valid.index[-2].strftime('%Y-%m-%d')}: {cred_spr_eu_valid.iloc[-2]:.4f}")
print(f"      {cred_spr_eu_valid.index[-1].strftime('%Y-%m-%d')}: {cred_spr_eu_valid.iloc[-1]:.4f}")

all_dfs.append(df_monthly)



# ----------------------------------------------------------------------------
# FILE 2: Nontradable Risk Factors - EPU
# ----------------------------------------------------------------------------

print("\n📂 Nontradable_risk_factors.xlsx - EPU")

df = pd.read_excel(DATA_DIR / "Nontradable_risk_factors.xlsx", 
                   sheet_name="EPU", skiprows=10)
df = df.iloc[:, [0, 3, 4]]
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

epu_us_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
epu_eu_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')

epu_us_monthly = epu_us_daily.resample('ME').last()
epu_eu_monthly = epu_eu_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'EPU_US': epu_us_monthly,
    'EPU_EU': epu_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: EPU_US={df_monthly['EPU_US'].isna().sum()}, EPU_EU={df_monthly['EPU_EU'].isna().sum()}")
print(f"   Last 2: {epu_us_monthly.index[-2].strftime('%Y-%m-%d')}, {epu_us_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   EPU_US: {epu_us_monthly.iloc[-2]:.4f}, {epu_us_monthly.iloc[-1]:.4f}")
print(f"   EPU_EU: {epu_eu_monthly.iloc[-2]:.4f}, {epu_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 3: Nontradable Risk Factors - UNC
# ----------------------------------------------------------------------------

print("\n📂 Nontradable_risk_factors.xlsx - UNC")

df = pd.read_excel(DATA_DIR / "Nontradable_risk_factors.xlsx", 
                   sheet_name="UNC", skiprows=11)
df = df.iloc[:, [0, 4, 5, 6]]  # A, E, F, G
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

uf_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
um_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')
ur_daily = pd.to_numeric(df.iloc[:, 2], errors='coerce')

uf_monthly = uf_daily.resample('ME').last()
um_monthly = um_daily.resample('ME').last()
ur_monthly = ur_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'ΔUF': uf_monthly,
    'ΔUM': um_monthly,
    'ΔUR': ur_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: ΔUF={df_monthly['ΔUF'].isna().sum()}, ΔUM={df_monthly['ΔUM'].isna().sum()}, ΔUR={df_monthly['ΔUR'].isna().sum()}")
print(f"   Last 2: {uf_monthly.index[-2].strftime('%Y-%m-%d')}, {uf_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   ΔUF: {uf_monthly.iloc[-2]:.4f}, {uf_monthly.iloc[-1]:.4f}")
print(f"   ΔUM: {um_monthly.iloc[-2]:.4f}, {um_monthly.iloc[-1]:.4f}")
print(f"   ΔUR: {ur_monthly.iloc[-2]:.4f}, {ur_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 4: Nontradable Risk Factors - LIQNT
# ----------------------------------------------------------------------------

print("\n📂 Nontradable_risk_factors.xlsx - LIQNT")

df = pd.read_excel(DATA_DIR / "Nontradable_risk_factors.xlsx", 
                   sheet_name="LIQNT", skiprows=79)
df = df.iloc[:, [0, 2, 3]]  # A, C, D
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

liqnt_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
liq_v_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')

liqnt_monthly = liqnt_daily.resample('ME').last()
liq_v_monthly = liq_v_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'LIQNT': liqnt_monthly,
    'LIQ_V': liq_v_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: LIQNT={df_monthly['LIQNT'].isna().sum()}, LIQ_V={df_monthly['LIQ_V'].isna().sum()}")
print(f"   Last 2: {liqnt_monthly.index[-2].strftime('%Y-%m-%d')}, {liqnt_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   LIQNT: {liqnt_monthly.iloc[-2]:.4f}, {liqnt_monthly.iloc[-1]:.4f}")
print(f"   LIQ_V: {liq_v_monthly.iloc[-2]:.4f}, {liq_v_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 5: Nontradable Risk Factors - YSP
# ----------------------------------------------------------------------------

print("\n📂 Nontradable_risk_factors.xlsx - YSP")

df = pd.read_excel(DATA_DIR / "Nontradable_risk_factors.xlsx", 
                   sheet_name="YSP", skiprows=13)
df = df.iloc[:, [0, 1, 2]]  # A, B, C
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

ysp_us_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
ysp_eu_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')

ysp_us_monthly = ysp_us_daily.resample('ME').last()
ysp_eu_monthly = ysp_eu_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'YSP_US': ysp_us_monthly,
    'YSP_EU': ysp_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: YSP_US={df_monthly['YSP_US'].isna().sum()}, YSP_EU={df_monthly['YSP_EU'].isna().sum()}")
print(f"   Last 2: {ysp_us_monthly.index[-2].strftime('%Y-%m-%d')}, {ysp_us_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   YSP_US: {ysp_us_monthly.iloc[-2]:.4f}, {ysp_us_monthly.iloc[-1]:.4f}")
print(f"   YSP_EU: {ysp_eu_monthly.iloc[-2]:.4f}, {ysp_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 6: Nontradable Risk Factors - HKM_IC
# ----------------------------------------------------------------------------

print("\n📂 Nontradable_risk_factors.xlsx - HKM_IC")

df = pd.read_excel(DATA_DIR / "Nontradable_risk_factors.xlsx", 
                   sheet_name="HKM_IC", skiprows=10)
df = df.iloc[:, [0, 1]]  # A, B
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

hkm_ic_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
hkm_ic_monthly = hkm_ic_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'HKM_IC': hkm_ic_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: HKM_IC={df_monthly['HKM_IC'].isna().sum()}")
print(f"   Last 2: {hkm_ic_monthly.index[-2].strftime('%Y-%m-%d')}, {hkm_ic_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   HKM_IC: {hkm_ic_monthly.iloc[-2]:.4f}, {hkm_ic_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 7: Nontradable Risk Factors - VIX
# ----------------------------------------------------------------------------

print("\n📂 Nontradable_risk_factors.xlsx - VIX")

df = pd.read_excel(DATA_DIR / "Nontradable_risk_factors.xlsx", 
                   sheet_name="VIX", skiprows=14)
df = df.iloc[:, [6, 9, 10]]  # G, J, K
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

dvix_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
dv2x_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')

dvix_monthly = dvix_daily.resample('ME').last()
dv2x_monthly = dv2x_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'ΔVIX': dvix_monthly,
    'ΔV2X': dv2x_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: ΔVIX={df_monthly['ΔVIX'].isna().sum()}, ΔV2X={df_monthly['ΔV2X'].isna().sum()}")
print(f"   Last 2: {dvix_monthly.index[-2].strftime('%Y-%m-%d')}, {dvix_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   ΔVIX: {dvix_monthly.iloc[-2]:.4f}, {dvix_monthly.iloc[-1]:.4f}")
print(f"   ΔV2X: {dv2x_monthly.iloc[-2]:.4f}, {dv2x_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 8: Nontradable Risk Factors - EP_SVIX
# ----------------------------------------------------------------------------

print("\n📂 Nontradable_risk_factors.xlsx - EP_SVIX")

df = pd.read_excel(DATA_DIR / "Nontradable_risk_factors.xlsx",
                   sheet_name="EP_SVIX", skiprows=18)

df = df.iloc[:, [8, 9, 10]]  # I, J, K
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

ep_svix_1m_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
ep_svix_3m_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')

ep_svix_1m_monthly = ep_svix_1m_daily.resample('ME').last()
ep_svix_3m_monthly = ep_svix_3m_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'EP_SVIX_1M': ep_svix_1m_monthly,
    'EP_SVIX_3M': ep_svix_3m_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: EP_SVIX_1M={df_monthly['EP_SVIX_1M'].isna().sum()}, EP_SVIX_3M={df_monthly['EP_SVIX_3M'].isna().sum()}")
print(f"   Last 2: {ep_svix_1m_monthly.index[-2].strftime('%Y-%m-%d')}, {ep_svix_1m_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   EP_SVIX_1M: {ep_svix_1m_monthly.iloc[-2]:.4f}, {ep_svix_1m_monthly.iloc[-1]:.4f}")
print(f"   EP_SVIX_3M: {ep_svix_3m_monthly.iloc[-2]:.4f}, {ep_svix_3m_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 9: Tradable Corporate Bond Factors - TERM
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - TERM")

df = pd.read_excel(DATA_DIR / "Tradable_corporate_bond_factors.xlsx", 
                   sheet_name="TERM", skiprows=14)
df = df.iloc[:, [5, 9, 10]]  # F, J, K
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

term_us_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
term_eu_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')

term_us_monthly = term_us_daily.resample('ME').last()
term_eu_monthly = term_eu_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'TERM_US': term_us_monthly,
    'TERM_EU': term_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: TERM_US={df_monthly['TERM_US'].isna().sum()}, TERM_EU={df_monthly['TERM_EU'].isna().sum()}")

# Dropna per stampare solo valori validi
term_us_valid = term_us_monthly.dropna()
term_eu_valid = term_eu_monthly.dropna()

print(f"\n   TERM_US (last 2 valid):")
print(f"      {term_us_valid.index[-2].strftime('%Y-%m-%d')}: {term_us_valid.iloc[-2]:.4f}")
print(f"      {term_us_valid.index[-1].strftime('%Y-%m-%d')}: {term_us_valid.iloc[-1]:.4f}")

print(f"   TERM_EU (last 2 valid):")
print(f"      {term_eu_valid.index[-2].strftime('%Y-%m-%d')}: {term_eu_valid.iloc[-2]:.4f}")
print(f"      {term_eu_valid.index[-1].strftime('%Y-%m-%d')}: {term_eu_valid.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 10: Tradable Corporate Bond Factors - SWAPTION
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - SWAPTION")

df = pd.read_excel(DATA_DIR / "Tradable_corporate_bond_factors.xlsx", 
                   sheet_name="SWAPTION", skiprows=10)
df = df.iloc[:, [6, 12, 13]]  # G, M, N
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

atm_iv_itrx_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
atm_iv_cdx_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')

atm_iv_itrx_monthly = atm_iv_itrx_daily.resample('ME').last()
atm_iv_cdx_monthly = atm_iv_cdx_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'ATM_IV_ITRX': atm_iv_itrx_monthly,
    'ATM_IV_CDX': atm_iv_cdx_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: ATM_IV_ITRX={df_monthly['ATM_IV_ITRX'].isna().sum()}, ATM_IV_CDX={df_monthly['ATM_IV_CDX'].isna().sum()}")
print(f"   Last 2: {atm_iv_itrx_monthly.index[-2].strftime('%Y-%m-%d')}, {atm_iv_itrx_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   ATM_IV_ITRX: {atm_iv_itrx_monthly.iloc[-2]:.4f}, {atm_iv_itrx_monthly.iloc[-1]:.4f}")
print(f"   ATM_IV_CDX: {atm_iv_cdx_monthly.iloc[-2]:.4f}, {atm_iv_cdx_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 11: Tradable Corporate Bond Factors - DEF
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - DEF")

df = pd.read_excel(DATA_DIR / "Tradable_corporate_bond_factors.xlsx", 
                   sheet_name="DEF", skiprows=16)
df = df.iloc[:, [6, 11, 12]]  # G, L, M
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

def_us_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
def_eu_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')

def_us_monthly = def_us_daily.resample('ME').last()
def_eu_monthly = def_eu_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'DEF_US': def_us_monthly,
    'DEF_EU': def_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: DEF_US={df_monthly['DEF_US'].isna().sum()}, DEF_EU={df_monthly['DEF_EU'].isna().sum()}")
print(f"   Last 2: {def_us_monthly.index[-2].strftime('%Y-%m-%d')}, {def_us_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   DEF_US: {def_us_monthly.iloc[-2]:.4f}, {def_us_monthly.iloc[-1]:.4f}")
print(f"   DEF_EU: {def_eu_monthly.iloc[-2]:.4f}, {def_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 12: Tradable Corporate Bond Factors - SLOPE
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - SLOPE")

df = pd.read_excel(DATA_DIR / "Tradable_corporate_bond_factors.xlsx", 
                   sheet_name="SLOPE", skiprows=18)

# G, L, M, N, O
# 0-indexed: G=6, L=11, M=12, N=13, O=14
df = df.iloc[:, [6, 11, 12, 13, 14]]
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

dslope_us_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
dslope_eu_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')
d10y_yield_eu_daily = pd.to_numeric(df.iloc[:, 2], errors='coerce')
d10y_yield_us_daily = pd.to_numeric(df.iloc[:, 3], errors='coerce')

dslope_us_monthly = dslope_us_daily.resample('ME').last()
dslope_eu_monthly = dslope_eu_daily.resample('ME').last()
d10y_yield_eu_monthly = d10y_yield_eu_daily.resample('ME').last()
d10y_yield_us_monthly = d10y_yield_us_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'ΔSLOPE_US': dslope_us_monthly,
    'ΔSLOPE_EU': dslope_eu_monthly,
    'Δ10Y_YIELD_EU': d10y_yield_eu_monthly,
    'Δ10Y_YIELD_US': d10y_yield_us_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: ΔSLOPE_US={df_monthly['ΔSLOPE_US'].isna().sum()}, ΔSLOPE_EU={df_monthly['ΔSLOPE_EU'].isna().sum()}, Δ10Y_YIELD_EU={df_monthly['Δ10Y_YIELD_EU'].isna().sum()}, Δ10Y_YIELD_US={df_monthly['Δ10Y_YIELD_US'].isna().sum()}")
print(f"   Last 2: {dslope_us_monthly.index[-2].strftime('%Y-%m-%d')}, {dslope_us_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   ΔSLOPE_US: {dslope_us_monthly.iloc[-2]:.4f}, {dslope_us_monthly.iloc[-1]:.4f}")
print(f"   ΔSLOPE_EU: {dslope_eu_monthly.iloc[-2]:.4f}, {dslope_eu_monthly.iloc[-1]:.4f}")
print(f"   Δ10Y_YIELD_EU: {d10y_yield_eu_monthly.iloc[-2]:.4f}, {d10y_yield_eu_monthly.iloc[-1]:.4f}")
print(f"   Δ10Y_YIELD_US: {d10y_yield_us_monthly.iloc[-2]:.4f}, {d10y_yield_us_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)


# ----------------------------------------------------------------------------
# FILE 13: Tradable Corporate Bond Factors - TED
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - TED")

from statsmodels.tsa.ar_model import AutoReg

df = pd.read_excel(DATA_DIR / "Tradable_corporate_bond_factors.xlsx", 
                   sheet_name="TED", skiprows=18)
df = df.iloc[:, [6, 8, 9, 10]]  # G, I, J, K
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

libor_repo_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')
ted_us_daily = pd.to_numeric(df.iloc[:, 1], errors='coerce')
ted_eu_daily = pd.to_numeric(df.iloc[:, 2], errors='coerce')

libor_repo_monthly = libor_repo_daily.resample('ME').last()
libor_repo_monthly.index.freq = 'ME'  # ← FIX

ted_us_monthly = ted_us_daily.resample('ME').last()
ted_us_monthly.index.freq = 'ME'  # ← FIX

ted_eu_monthly = ted_eu_daily.resample('ME').last()
ted_eu_monthly.index.freq = 'ME'  # ← FIX

# AR(2) innovations per ciascuno
libor_repo_clean = libor_repo_monthly.dropna()
libor_repo_clean.name = 'LIBOR_REPO'
model1 = AutoReg(libor_repo_clean, lags=2, trend='c')
libor_repo_innov = model1.fit().resid

ted_us_clean = ted_us_monthly.dropna()
ted_us_clean.name = 'TED_US'
model2 = AutoReg(ted_us_clean, lags=2, trend='c')
ted_us_innov = model2.fit().resid

ted_eu_clean = ted_eu_monthly.dropna()
ted_eu_clean.name = 'TED_EU'
model3 = AutoReg(ted_eu_clean, lags=2, trend='c')
ted_eu_innov = model3.fit().resid

df_monthly = pd.DataFrame({
    'LIBOR_REPO_SHOCK': libor_repo_innov,
    'TED_SHOCK_US': ted_us_innov,
    'TED_SHOCK_EU': ted_eu_innov
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: LIBOR_REPO={df_monthly['LIBOR_REPO_SHOCK'].isna().sum()}, TED_US={df_monthly['TED_SHOCK_US'].isna().sum()}, TED_EU={df_monthly['TED_SHOCK_EU'].isna().sum()}")

print(f"\n   LIBOR_REPO_SHOCK:")
print(f"      {libor_repo_innov.index[-2].strftime('%Y-%m-%d')}: {libor_repo_innov.iloc[-2]:.4f}")
print(f"      {libor_repo_innov.index[-1].strftime('%Y-%m-%d')}: {libor_repo_innov.iloc[-1]:.4f}")

print(f"   TED_SHOCK_US:")
print(f"      {ted_us_innov.index[-2].strftime('%Y-%m-%d')}: {ted_us_innov.iloc[-2]:.4f}")
print(f"      {ted_us_innov.index[-1].strftime('%Y-%m-%d')}: {ted_us_innov.iloc[-1]:.4f}")

print(f"   TED_SHOCK_EU:")
print(f"      {ted_eu_innov.index[-2].strftime('%Y-%m-%d')}: {ted_eu_innov.iloc[-2]:.4f}")
print(f"      {ted_eu_innov.index[-1].strftime('%Y-%m-%d')}: {ted_eu_innov.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 14: Tradable Corporate Bond Factors - FAIL
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - FAIL")

df = pd.read_excel(DATA_DIR / "Tradable_corporate_bond_factors.xlsx", 
                   sheet_name="FAIL", skiprows=14)  
df = df.iloc[:, [8, 11]]  # I, L
df.set_index(df.columns[0], inplace=True)
df.index.name = 'Date'

dfails_pct_daily = pd.to_numeric(df.iloc[:, 0], errors='coerce')

dfails_pct_monthly = dfails_pct_daily.resample('ME').last()

df_monthly = pd.DataFrame({
    'ΔFAILS_PCT_TSY': dfails_pct_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: ΔFAILS_PCT_TSY={df_monthly['ΔFAILS_PCT_TSY'].isna().sum()}")
print(f"   ΔFAILS_PCT_TSY:")
print(f"      {dfails_pct_monthly.index[-2].strftime('%Y-%m-%d')}: {dfails_pct_monthly.iloc[-2]:.4f}")
print(f"      {dfails_pct_monthly.index[-1].strftime('%Y-%m-%d')}: {dfails_pct_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 15: Tradable Corporate Bond Factors - CDS (Prime Broker CDS spreads)
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - CDS")

df = pd.read_excel(
    DATA_DIR / "Tradable_corporate_bond_factors.xlsx",
    sheet_name="CDS",
    skiprows=14  # data start at row 15
)

# Columns: I (dates), J, K, L, M  -> 0-indexed: 8, 9, 10, 11, 12
df = df.iloc[:, [8, 9, 10, 11, 12]]
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

pb_cds_5y_us_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
pb_cds_1y_us_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")
pb_cds_5y_eu_daily = pd.to_numeric(df.iloc[:, 2], errors="coerce")
pb_cds_1y_eu_daily = pd.to_numeric(df.iloc[:, 3], errors="coerce")

pb_cds_5y_us_monthly = pb_cds_5y_us_daily.resample("ME").last()
pb_cds_1y_us_monthly = pb_cds_1y_us_daily.resample("ME").last()
pb_cds_5y_eu_monthly = pb_cds_5y_eu_daily.resample("ME").last()
pb_cds_1y_eu_monthly = pb_cds_1y_eu_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "PB_US_CDS_5Y": pb_cds_5y_us_monthly,
    "PB_US_CDS_1Y": pb_cds_1y_us_monthly,
    "PB_EU_CDS_5Y": pb_cds_5y_eu_monthly,
    "PB_EU_CDS_1Y": pb_cds_1y_eu_monthly,
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(
    "   Missing: "
    f"PB_US_CDS_5Y={df_monthly['PB_US_CDS_5Y'].isna().sum()}, "
    f"PB_US_CDS_1Y={df_monthly['PB_US_CDS_1Y'].isna().sum()}, "
    f"PB_EU_CDS_5Y={df_monthly['PB_EU_CDS_5Y'].isna().sum()}, "
    f"PB_EU_CDS_1Y={df_monthly['PB_EU_CDS_1Y'].isna().sum()}"
)
print(f"   Last 2: {pb_cds_5y_us_monthly.index[-2].strftime('%Y-%m-%d')}, {pb_cds_5y_us_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   PB_US_CDS_5Y: {pb_cds_5y_us_monthly.iloc[-2]:.4f}, {pb_cds_5y_us_monthly.iloc[-1]:.4f}")
print(f"   PB_US_CDS_1Y: {pb_cds_1y_us_monthly.iloc[-2]:.4f}, {pb_cds_1y_us_monthly.iloc[-1]:.4f}")
print(f"   PB_EU_CDS_5Y: {pb_cds_5y_eu_monthly.iloc[-2]:.4f}, {pb_cds_5y_eu_monthly.iloc[-1]:.4f}")
print(f"   PB_EU_CDS_1Y: {pb_cds_1y_eu_monthly.iloc[-2]:.4f}, {pb_cds_1y_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 16: Tradable Stock Factors - BAB
# ----------------------------------------------------------------------------

print("\n📂 Tradable_stock_factors.xlsx - BAB")

df = pd.read_excel(
    DATA_DIR / "Tradable_stock_factors.xlsx",
    sheet_name="BAB",
    skiprows=14  # data start at row 15
)

# Columns: A (dates), B (BAB_EU), C (BAB_US) -> 0-indexed: 0, 1, 2
df = df.iloc[:, [0, 1, 2]]
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

bab_eu_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
bab_us_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")

bab_eu_monthly = bab_eu_daily.resample("ME").last()
bab_us_monthly = bab_us_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "BAB_EU": bab_eu_monthly,
    "BAB_US": bab_us_monthly,
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: BAB_EU={df_monthly['BAB_EU'].isna().sum()}, BAB_US={df_monthly['BAB_US'].isna().sum()}")
print(f"   Last 2: {bab_us_monthly.index[-2].strftime('%Y-%m-%d')}, {bab_us_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   BAB_EU: {bab_eu_monthly.iloc[-2]:.4f}, {bab_eu_monthly.iloc[-1]:.4f}")
print(f"   BAB_US: {bab_us_monthly.iloc[-2]:.4f}, {bab_us_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 17: Tradable Stock Factors - EU (Fama-French)
# ----------------------------------------------------------------------------

print("\n📂 Tradable_stock_factors.xlsx - EU")

df = pd.read_excel(
    DATA_DIR / "Tradable_stock_factors.xlsx",
    sheet_name="EU",
    skiprows=12
)

df = df.iloc[:, [9, 10, 11, 12, 13, 14, 15]]  # J, K, L, M, N, O, P
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

mkt_eu_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
smb_eu_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")
hml_eu_daily = pd.to_numeric(df.iloc[:, 2], errors="coerce")
rmw_eu_daily = pd.to_numeric(df.iloc[:, 3], errors="coerce")
cma_eu_daily = pd.to_numeric(df.iloc[:, 4], errors="coerce")
umd_eu_daily = pd.to_numeric(df.iloc[:, 5], errors="coerce")

mkt_eu_monthly = mkt_eu_daily.resample("ME").last()
smb_eu_monthly = smb_eu_daily.resample("ME").last()
hml_eu_monthly = hml_eu_daily.resample("ME").last()
rmw_eu_monthly = rmw_eu_daily.resample("ME").last()
cma_eu_monthly = cma_eu_daily.resample("ME").last()
umd_eu_monthly = umd_eu_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "MKT_EU": mkt_eu_monthly,
    "SMB_EU": smb_eu_monthly,
    "HML_EU": hml_eu_monthly,
    "RMW_EU": rmw_eu_monthly,
    "CMA_EU": cma_eu_monthly,
    "UMD_EU": umd_eu_monthly,
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: MKT={df_monthly['MKT_EU'].isna().sum()}, SMB={df_monthly['SMB_EU'].isna().sum()}, HML={df_monthly['HML_EU'].isna().sum()}, RMW={df_monthly['RMW_EU'].isna().sum()}, CMA={df_monthly['CMA_EU'].isna().sum()}, UMD={df_monthly['UMD_EU'].isna().sum()}")
print(f"\n   Last 2 values ({mkt_eu_monthly.index[-2].strftime('%Y-%m-%d')}, {mkt_eu_monthly.index[-1].strftime('%Y-%m-%d')}):")
print(f"   MKT_EU: {mkt_eu_monthly.iloc[-2]:.4f}, {mkt_eu_monthly.iloc[-1]:.4f}")
print(f"   SMB_EU: {smb_eu_monthly.iloc[-2]:.4f}, {smb_eu_monthly.iloc[-1]:.4f}")
print(f"   HML_EU: {hml_eu_monthly.iloc[-2]:.4f}, {hml_eu_monthly.iloc[-1]:.4f}")
print(f"   RMW_EU: {rmw_eu_monthly.iloc[-2]:.4f}, {rmw_eu_monthly.iloc[-1]:.4f}")
print(f"   CMA_EU: {cma_eu_monthly.iloc[-2]:.4f}, {cma_eu_monthly.iloc[-1]:.4f}")
print(f"   UMD_EU: {umd_eu_monthly.iloc[-2]:.4f}, {umd_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 18: Other Factors - HPW (NOISE)
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - HPW")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="HPW",
    skiprows=9  
)

# Columns: G
# (dates), L (HPW_NOISE) -> 0-indexed: 6, 11
df = df.iloc[:, [9, 11]]  # J, L
df.set_index(df.columns[0], inplace=True)  # G → index
df.index.name = "Date"

hpw_noise_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")

hpw_noise_monthly = hpw_noise_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "HPW_NOISE": hpw_noise_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: HPW_NOISE={df_monthly['HPW_NOISE'].isna().sum()}")
print(f"   Last 2: {hpw_noise_monthly.index[-2].strftime('%Y-%m-%d')}, {hpw_noise_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   HPW_NOISE: {hpw_noise_monthly.iloc[-2]:.4f}, {hpw_noise_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 19: Other Factors - EBP
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - EBP")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="EBP",
    skiprows=9  # data start at row 10
)

# Columns: A (dates), B (EBP) -> 0-indexed: 0, 1
df = df.iloc[:, [0, 1]]  # A, B
df.set_index(df.columns[0], inplace=True)  # A → index
df.index.name = "Date"

ebp_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")

ebp_monthly = ebp_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "EBP": ebp_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: EBP={df_monthly['EBP'].isna().sum()}")
print(f"   Last 2: {ebp_monthly.index[-2].strftime('%Y-%m-%d')}, {ebp_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   EBP: {ebp_monthly.iloc[-2]:.4f}, {ebp_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 20: Other Factors - REPO
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - REPO")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="REPO"
)

# Columns: F (dates), H (GC-REPO_T-BILL) -> 0-indexed: 5, 7
df = df.iloc[:, [5, 7]]  # F, H
df.set_index(df.columns[0], inplace=True)  # F → index
df.index.name = "Date"

gc_repo_tbill_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")

gc_repo_tbill_monthly = gc_repo_tbill_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "GC-REPO_T-BILL": gc_repo_tbill_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: GC-REPO_T-BILL={df_monthly['GC-REPO_T-BILL'].isna().sum()}")
print(f"   Last 2: {gc_repo_tbill_monthly.index[-2].strftime('%Y-%m-%d')}, {gc_repo_tbill_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   GC-REPO_T-BILL: {gc_repo_tbill_monthly.iloc[-2]:.4f}, {gc_repo_tbill_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 21: Other Factors - Amihur (ILLIQ + SILLIQ)
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - Amihur")

from statsmodels.tsa.ar_model import AutoReg

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="Amihur",
    skiprows=13  # data start at row 14
)

# Columns: A (dates), F (ILLIQ), G (input for SILLIQ) -> 0-indexed: 0, 5, 6
df = df.iloc[:, [3, 5, 6]]  # D, F, G
df.set_index(df.columns[0], inplace=True)  # A → index
df.index.name = "Date"

illiq_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")      # F
illiq_input_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # G

illiq_monthly = illiq_daily.resample("ME").last()
illiq_monthly.index.freq = "ME"  # ← FIX

illiq_input_monthly = illiq_input_daily.resample("ME").last()
illiq_input_monthly.index.freq = "ME"  # ← FIX

# AR(3) innovation on the monthly-average input series (col G)
illiq_input_clean = illiq_input_monthly.dropna()
illiq_input_clean.name = "ILLIQ_INPUT"
model = AutoReg(illiq_input_clean, lags=3, trend="c")
silliq_innov = model.fit().resid

df_monthly = pd.DataFrame({
    "ILLIQ": illiq_monthly,
    "SILLIQ": silliq_innov
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: ILLIQ={df_monthly['ILLIQ'].isna().sum()}, SILLIQ={df_monthly['SILLIQ'].isna().sum()}")

print(f"\n   ILLIQ:")
print(f"      {illiq_monthly.index[-2].strftime('%Y-%m-%d')}: {illiq_monthly.iloc[-2]:.4f}")
print(f"      {illiq_monthly.index[-1].strftime('%Y-%m-%d')}: {illiq_monthly.iloc[-1]:.4f}")

print(f"   SILLIQ:")
print(f"      {silliq_innov.index[-2].strftime('%Y-%m-%d')}: {silliq_innov.iloc[-2]:.4f}")
print(f"      {silliq_innov.index[-1].strftime('%Y-%m-%d')}: {silliq_innov.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 22: Other Factors - GVAL_GMOM
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - GVAL_GMOM")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="GVAL_GMOM",
    skiprows=18  # data start at row 19
)

# Columns: F (dates), H (GVAL_EU), I (GMOM_EU) -> 0-indexed: 5, 7, 8
df = df.iloc[:, [5, 7, 8]]  # F, H, I
df.set_index(df.columns[0], inplace=True)  # F → index
df.index.name = "Date"

gval_eu_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
gmom_eu_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")

gval_eu_monthly = gval_eu_daily.resample("ME").last()
gmom_eu_monthly = gmom_eu_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "GVAL_EU": gval_eu_monthly,
    "GMOM_EU": gmom_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: GVAL_EU={df_monthly['GVAL_EU'].isna().sum()}, GMOM_EU={df_monthly['GMOM_EU'].isna().sum()}")
print(f"   Last 2: {gval_eu_monthly.index[-2].strftime('%Y-%m-%d')}, {gval_eu_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   GVAL_EU: {gval_eu_monthly.iloc[-2]:.4f}, {gval_eu_monthly.iloc[-1]:.4f}")
print(f"   GMOM_EU: {gmom_eu_monthly.iloc[-2]:.4f}, {gmom_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 23: Other Factors - MOVE
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - MOVE")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="MOVE",
    skiprows=18  # data start at row 19
)

# Columns: D (dates), E (MOVE) -> 0-indexed: 3, 4
df = df.iloc[:, [3, 4]]  # D, E
df.set_index(df.columns[0], inplace=True)  # D → index
df.index.name = "Date"

move_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")

move_monthly = move_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "MOVE": move_monthly
    
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: MOVE={df_monthly['MOVE'].isna().sum()}")
print(f"   Last 2: {move_monthly.index[-2].strftime('%Y-%m-%d')}, {move_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   MOVE: {move_monthly.iloc[-2]:.4f}, {move_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 24: Tradable Corporate Bond Factors - ITRX (SNR-MAIN, MAIN_5/3_FLATTENER, XOVER/MAIN)
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - ITRX")

df = pd.read_excel(
    DATA_DIR / "Tradable_corporate_bond_factors.xlsx",
    sheet_name="ITRX",
    skiprows=15  # first value from row 16
)

# Columns: M (dates), Q (SNR-MAIN), R (MAIN_5/3_FLATTENER), S (XOVER/MAIN)
# 0-indexed: 12, 16, 17, 18
df = df.iloc[:, [12, 16, 17, 18]]  # M, Q, R, S
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

snr_main_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
main_5_3_flattener_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")
xover_main_daily = pd.to_numeric(df.iloc[:, 2], errors="coerce")

snr_main_monthly = snr_main_daily.resample("ME").last()
main_5_3_flattener_monthly = main_5_3_flattener_daily.resample("ME").last()
xover_main_monthly = xover_main_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "SNR-MAIN": snr_main_monthly,
    "MAIN_5/3_FLATTENER": main_5_3_flattener_monthly,
    "XOVER/MAIN": xover_main_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(
    "   Missing: "
    f"SNR-MAIN={df_monthly['SNR-MAIN'].isna().sum()}, "
    f"MAIN_5/3_FLATTENER={df_monthly['MAIN_5/3_FLATTENER'].isna().sum()}, "
    f"XOVER/MAIN={df_monthly['XOVER/MAIN'].isna().sum()}"
)
print(f"   Last 2: {snr_main_monthly.index[-2].strftime('%Y-%m-%d')}, {snr_main_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   SNR-MAIN: {snr_main_monthly.iloc[-2]:.4f}, {snr_main_monthly.iloc[-1]:.4f}")
print(f"   MAIN_5/3_FLATTENER: {main_5_3_flattener_monthly.iloc[-2]:.4f}, {main_5_3_flattener_monthly.iloc[-1]:.4f}")
print(f"   XOVER/MAIN: {xover_main_monthly.iloc[-2]:.4f}, {xover_main_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 25: Tradable Corporate Bond Factors - ITRX_MAIN / CDX_IG
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - ITRX_MAIN / CDX_IG")

df = pd.read_excel(
    DATA_DIR / "Tradable_corporate_bond_factors.xlsx",
    sheet_name="CDS_INDEX",
    skiprows=18  # first value from row 19
)

# Columns: E (dates), H (ITRX_MAIN), I (CDX_IG) -> 0-indexed: 4, 7, 8
df = df.iloc[:, [4, 7, 8]]  # E, H, I
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

itrx_main_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
cdx_ig_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")

itrx_main_monthly = itrx_main_daily.resample("ME").last()
cdx_ig_monthly = cdx_ig_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "ITRX_MAIN": itrx_main_monthly,
    "CDX_IG": cdx_ig_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: ITRX_MAIN={df_monthly['ITRX_MAIN'].isna().sum()}, CDX_IG={df_monthly['CDX_IG'].isna().sum()}")
print(f"   Last 2: {itrx_main_monthly.index[-2].strftime('%Y-%m-%d')}, {itrx_main_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   ITRX_MAIN: {itrx_main_monthly.iloc[-2]:.4f}, {itrx_main_monthly.iloc[-1]:.4f}")
print(f"   CDX_IG: {cdx_ig_monthly.iloc[-2]:.4f}, {cdx_ig_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 26: Other Factors - BFCI (Bloomberg Euro Area Financial Conditions Index)
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - BFCI")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="BFCI",
    skiprows=11
)

# Columns: D (dates), F (BFCI_EU) -> 0-indexed: 3, 5
df = df.iloc[:, [3, 5]]  # D, F
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

bfci_eu_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")

bfci_eu_monthly = bfci_eu_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "BFCI_EU": bfci_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: BFCI_EU={df_monthly['BFCI_EU'].isna().sum()}")
print(f"   Last 2: {bfci_eu_monthly.index[-2].strftime('%Y-%m-%d')}, {bfci_eu_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   BFCI_EU: {bfci_eu_monthly.iloc[-2]:.4f}, {bfci_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 27: Other Factors - SS (Swap Spreads)
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - SS")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="SS"
)

# Columns: F (dates), K (SS10Y), L (SS5Y), M (SS2Y) -> 0-indexed: 5, 10, 11, 12
df = df.iloc[:, [5, 10, 11, 12]]  # F, K, L, M
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

ss10y_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
ss5y_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")
ss2y_daily = pd.to_numeric(df.iloc[:, 2], errors="coerce")

ss10y_monthly = ss10y_daily.resample("ME").last()
ss5y_monthly = ss5y_daily.resample("ME").last()
ss2y_monthly = ss2y_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "SS10Y": ss10y_monthly,
    "SS5Y": ss5y_monthly,
    "SS2Y": ss2y_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(
    "   Missing: "
    f"SS10Y={df_monthly['SS10Y'].isna().sum()}, "
    f"SS5Y={df_monthly['SS5Y'].isna().sum()}, "
    f"SS2Y={df_monthly['SS2Y'].isna().sum()}"
)
print(f"   Last 2: {ss10y_monthly.index[-2].strftime('%Y-%m-%d')}, {ss10y_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   SS10Y: {ss10y_monthly.iloc[-2]:.4f}, {ss10y_monthly.iloc[-1]:.4f}")
print(f"   SS5Y: {ss5y_monthly.iloc[-2]:.4f}, {ss5y_monthly.iloc[-1]:.4f}")
print(f"   SS2Y: {ss2y_monthly.iloc[-2]:.4f}, {ss2y_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 28: Tradable Corporate Bond Factors - LIBOR_OIS
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - LIBOR_OIS")

df = pd.read_excel(
    DATA_DIR / "Tradable_corporate_bond_factors.xlsx",
    sheet_name="LIBOR_OIS",
    skiprows=12  # data start at row 13
)

# Columns: O (dates), R (EURIBOR_OIS), S (LIBOR_OIS) -> 0-indexed: 14, 17, 18
df = df.iloc[:, [14, 17, 18]]  # O, R, S
df.set_index(df.columns[0], inplace=True)  # O → index
df.index.name = "Date"

euribor_ois_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
libor_ois_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")

euribor_ois_monthly = euribor_ois_daily.resample("ME").last()
libor_ois_monthly = libor_ois_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "EURIBOR_OIS": euribor_ois_monthly,
    "LIBOR_OIS": libor_ois_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: EURIBOR_OIS={df_monthly['EURIBOR_OIS'].isna().sum()}, LIBOR_OIS={df_monthly['LIBOR_OIS'].isna().sum()}")

# Dropna per stampare solo valori validi
euribor_ois_valid = euribor_ois_monthly.dropna()
libor_ois_valid = libor_ois_monthly.dropna()

print(f"\n   EURIBOR_OIS (last 2 valid):")
print(f"      {euribor_ois_valid.index[-2].strftime('%Y-%m-%d')}: {euribor_ois_valid.iloc[-2]:.4f}")
print(f"      {euribor_ois_valid.index[-1].strftime('%Y-%m-%d')}: {euribor_ois_valid.iloc[-1]:.4f}")

print(f"   LIBOR_OIS (last 2 valid):")
print(f"      {libor_ois_valid.index[-2].strftime('%Y-%m-%d')}: {libor_ois_valid.iloc[-2]:.4f}")
print(f"      {libor_ois_valid.index[-1].strftime('%Y-%m-%d')}: {libor_ois_valid.iloc[-1]:.4f}")

all_dfs.append(df_monthly)


# ----------------------------------------------------------------------------
# FILE 29: Other Factors - Others (IV_BUND, IV_TSY, BTP_BUND, 5Y5Y_INFL, EURUSD_3M_IV)
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - Others")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="Others",
    skiprows = 12
)

# Columns: A (dates), H (IV_BUND), I (IV_TSY), O (BTP_BUND), P (5Y5Y_INFL), Q (EURUSD_3M_IV)
# 0-indexed: 0, 7, 8, 14, 15, 16
df = df.iloc[:, [0, 7, 8, 14, 15, 16]]  # A, H, I, O, P, Q
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

iv_bund_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
iv_tsy_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")
btp_bund_daily = pd.to_numeric(df.iloc[:, 2], errors="coerce")
infl_5y5y_daily = pd.to_numeric(df.iloc[:, 3], errors="coerce")
eurusd_3m_iv_daily = pd.to_numeric(df.iloc[:, 4], errors="coerce")

iv_bund_monthly = iv_bund_daily.resample("ME").last()
iv_tsy_monthly = iv_tsy_daily.resample("ME").last()
btp_bund_monthly = btp_bund_daily.resample("ME").last()
infl_5y5y_monthly = infl_5y5y_daily.resample("ME").last()
eurusd_3m_iv_monthly = eurusd_3m_iv_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "IV_BUND": iv_bund_monthly,
    "IV_TSY": iv_tsy_monthly,
    "BTP_BUND": btp_bund_monthly,
    "5Y5Y_INFL": infl_5y5y_monthly,
    "EURUSD_3M_IV": eurusd_3m_iv_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(
    "   Missing: "
    f"IV_BUND={df_monthly['IV_BUND'].isna().sum()}, "
    f"IV_TSY={df_monthly['IV_TSY'].isna().sum()}, "
    f"BTP_BUND={df_monthly['BTP_BUND'].isna().sum()}, "
    f"5Y5Y_INFL={df_monthly['5Y5Y_INFL'].isna().sum()}, "
    f"EURUSD_3M_IV={df_monthly['EURUSD_3M_IV'].isna().sum()}"
)
print(f"   Last 2: {iv_bund_monthly.index[-2].strftime('%Y-%m-%d')}, {iv_bund_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   IV_BUND: {iv_bund_monthly.iloc[-2]:.4f}, {iv_bund_monthly.iloc[-1]:.4f}")
print(f"   IV_TSY: {iv_tsy_monthly.iloc[-2]:.4f}, {iv_tsy_monthly.iloc[-1]:.4f}")
print(f"   BTP_BUND: {btp_bund_monthly.iloc[-2]:.4f}, {btp_bund_monthly.iloc[-1]:.4f}")
print(f"   5Y5Y_INFL: {infl_5y5y_monthly.iloc[-2]:.4f}, {infl_5y5y_monthly.iloc[-1]:.4f}")
print(f"   EURUSD_3M_IV: {eurusd_3m_iv_monthly.iloc[-2]:.4f}, {eurusd_3m_iv_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 30: Tradable Stock Factors - Fung_Hsieh
# ----------------------------------------------------------------------------

print("\n📂 Tradable_stock_factors.xlsx - Fung_Hsieh")

df = pd.read_excel(
    DATA_DIR / "Tradable_stock_factors.xlsx",
    sheet_name="Fung_Hsieh",
    skiprows=15  # starts from row 16
)

# Columns: A (Date=YYYYMM), B (PTFSBD), C (PTFSFX), D (PTFSCOM),
#          E (PTFSIR), F (PTFSSTK)
# 0-indexed: 0,1,2,3,4,5
df = df.iloc[:, [0, 1, 2, 3, 4, 5]]

# A → index (same as all other blocks)
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

# Convert YYYYMM -> datetime (necessary for resample)
df.index = (
    df.index.astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.strip()
)
df.index = pd.to_datetime(df.index + "01", format="%Y%m%d")

# Extract columns (same style)
ptfsbd_daily  = pd.to_numeric(df.iloc[:, 0], errors="coerce")
ptfsfx_daily  = pd.to_numeric(df.iloc[:, 1], errors="coerce")
ptfscom_daily = pd.to_numeric(df.iloc[:, 2], errors="coerce")
ptfsir_daily  = pd.to_numeric(df.iloc[:, 3], errors="coerce")
ptfsstk_daily = pd.to_numeric(df.iloc[:, 4], errors="coerce")

# Resample monthly (exactly like all other blocks)
ptfsbd_monthly  = ptfsbd_daily.resample("ME").last()
ptfsfx_monthly  = ptfsfx_daily.resample("ME").last()
ptfscom_monthly = ptfscom_daily.resample("ME").last()
ptfsir_monthly  = ptfsir_daily.resample("ME").last()
ptfsstk_monthly = ptfsstk_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "PTFSBD":  ptfsbd_monthly,
    "PTFSFX":  ptfsfx_monthly,
    "PTFSCOM": ptfscom_monthly,
    "PTFSIR":  ptfsir_monthly,
    "PTFSSTK": ptfsstk_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(
    "   Missing: "
    f"PTFSBD={df_monthly['PTFSBD'].isna().sum()}, "
    f"PTFSFX={df_monthly['PTFSFX'].isna().sum()}, "
    f"PTFSCOM={df_monthly['PTFSCOM'].isna().sum()}, "
    f"PTFSIR={df_monthly['PTFSIR'].isna().sum()}, "
    f"PTFSSTK={df_monthly['PTFSSTK'].isna().sum()}"
)

print(f"   Last 2: {ptfsbd_monthly.index[-2].strftime('%Y-%m-%d')}, {ptfsbd_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   PTFSBD: {ptfsbd_monthly.iloc[-2]:.4f}, {ptfsbd_monthly.iloc[-1]:.4f}")
print(f"   PTFSFX: {ptfsfx_monthly.iloc[-2]:.4f}, {ptfsfx_monthly.iloc[-1]:.4f}")
print(f"   PTFSCOM: {ptfscom_monthly.iloc[-2]:.4f}, {ptfscom_monthly.iloc[-1]:.4f}")
print(f"   PTFSIR: {ptfsir_monthly.iloc[-2]:.4f}, {ptfsir_monthly.iloc[-1]:.4f}")
print(f"   PTFSSTK: {ptfsstk_monthly.iloc[-2]:.4f}, {ptfsstk_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 31: Tradable Corporate Bond Factors - R2_EU / R5_EU / R10_EU
# ----------------------------------------------------------------------------

print("\n📂 Tradable_corporate_bond_factors.xlsx - TREAS (R2_EU / R5_EU / R10_EU)")

df = pd.read_excel(
    DATA_DIR / "Tradable_corporate_bond_factors.xlsx",
    sheet_name="TREAS",
    skiprows=7  # first value from row 8
)

# Columns: A (dates), B (R2_EU), C (R5_EU), D (R10_EU) -> 0-indexed: 0, 1, 2, 3
df = df.iloc[:, [0, 1, 2, 3]]  # A, B, C, D
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

r2_eu_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
r5_eu_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")
r10_eu_daily = pd.to_numeric(df.iloc[:, 2], errors="coerce")

r2_eu_monthly = r2_eu_daily.resample("ME").last()
r5_eu_monthly = r5_eu_daily.resample("ME").last()
r10_eu_monthly = r10_eu_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "R2_EU": r2_eu_monthly,
    "R5_EU": r5_eu_monthly,
    "R10_EU": r10_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: R2_EU={df_monthly['R2_EU'].isna().sum()}, R5_EU={df_monthly['R5_EU'].isna().sum()}, R10_EU={df_monthly['R10_EU'].isna().sum()}")
print(f"   Last 2: {r2_eu_monthly.index[-2].strftime('%Y-%m-%d')}, {r2_eu_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   R2_EU: {r2_eu_monthly.iloc[-2]:.4f}, {r2_eu_monthly.iloc[-1]:.4f}")
print(f"   R5_EU: {r5_eu_monthly.iloc[-2]:.4f}, {r5_eu_monthly.iloc[-1]:.4f}")
print(f"   R10_EU: {r10_eu_monthly.iloc[-2]:.4f}, {r10_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)


# ----------------------------------------------------------------------------
# FILE 32: Tradable Stock Factors - RI_RB (RI_EU / RB_EU)
# ----------------------------------------------------------------------------

print("\n📂 Tradable_stock_factors.xlsx - RI_RB (RI_EU / RB_EU)")

df = pd.read_excel(
    DATA_DIR / "Tradable_stock_factors.xlsx",
    sheet_name="RI_RB",
    skiprows=13  # first value from row 14
)

# Columns: A (dates), B (RI_EU), C (RB_EU) -> 0-indexed: 0, 1, 2
df = df.iloc[:, [0, 1, 2]]  # A, B, C
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

ri_eu_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")
rb_eu_daily = pd.to_numeric(df.iloc[:, 1], errors="coerce")

ri_eu_monthly = ri_eu_daily.resample("ME").last()
rb_eu_monthly = rb_eu_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "RI_EU": ri_eu_monthly,
    "RB_EU": rb_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: RI_EU={df_monthly['RI_EU'].isna().sum()}, RB_EU={df_monthly['RB_EU'].isna().sum()}")
print(f"   Last 2: {ri_eu_monthly.index[-2].strftime('%Y-%m-%d')}, {ri_eu_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   RI_EU: {ri_eu_monthly.iloc[-2]:.4f}, {ri_eu_monthly.iloc[-1]:.4f}")
print(f"   RB_EU: {rb_eu_monthly.iloc[-2]:.4f}, {rb_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)


# ----------------------------------------------------------------------------
# FILE 33: Tradable Stock Factors - RS (RS_EU)
# ----------------------------------------------------------------------------

print("\n📂 Tradable_stock_factors.xlsx - RS (RS_EU)")

df = pd.read_excel(
    DATA_DIR / "Tradable_stock_factors.xlsx",
    sheet_name="RS",
    skiprows=7  # first value from row 8
)

# Columns: A (dates), B (RS_EU) -> 0-indexed: 0, 1
df = df.iloc[:, [0, 1]]  # A, B
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

rs_eu_daily = pd.to_numeric(df.iloc[:, 0], errors="coerce")

rs_eu_monthly = rs_eu_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "RS_EU": rs_eu_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: RS_EU={df_monthly['RS_EU'].isna().sum()}")
print(f"   Last 2: {rs_eu_monthly.index[-2].strftime('%Y-%m-%d')}, {rs_eu_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   RS_EU: {rs_eu_monthly.iloc[-2]:.4f}, {rs_eu_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ----------------------------------------------------------------------------
# FILE 33: Other Factors - Active_FI
# ----------------------------------------------------------------------------

print("\n📂 Other_factors.xlsx - Active_FI")

df = pd.read_excel(
    DATA_DIR / "Other_factors.xlsx",
    sheet_name="Active_FI",
    skiprows=7  # first value from row 8
)

# Columns: A (dates), B (GLOBAL_TERM), C (GLOBAL_AGG), D (INFL_LINK),
#          E (CORP_CREDIT), F (EMERG_DEBT), G (EMERG_FX)
# 0-indexed: A=0, B=1, C=2, D=3, E=4, F=5, G=6
df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]]  # A..G
df.set_index(df.columns[0], inplace=True)
df.index.name = "Date"

global_term_daily  = pd.to_numeric(df.iloc[:, 0], errors="coerce")
global_agg_daily   = pd.to_numeric(df.iloc[:, 1], errors="coerce")
infl_link_daily    = pd.to_numeric(df.iloc[:, 2], errors="coerce")
corp_credit_daily  = pd.to_numeric(df.iloc[:, 3], errors="coerce")
emerg_debt_daily   = pd.to_numeric(df.iloc[:, 4], errors="coerce")
emerg_fx_daily     = pd.to_numeric(df.iloc[:, 5], errors="coerce")

global_term_monthly = global_term_daily.resample("ME").last()
global_agg_monthly  = global_agg_daily.resample("ME").last()
infl_link_monthly   = infl_link_daily.resample("ME").last()
corp_credit_monthly = corp_credit_daily.resample("ME").last()
emerg_debt_monthly  = emerg_debt_daily.resample("ME").last()
emerg_fx_monthly    = emerg_fx_daily.resample("ME").last()

df_monthly = pd.DataFrame({
    "GLOBAL_TERM": global_term_monthly,
    "GLOBAL_AGG": global_agg_monthly,
    "INFL_LINK": infl_link_monthly,
    "CORP_CREDIT": corp_credit_monthly,
    "EMERG_DEBT": emerg_debt_monthly,
    "EMERG_FX": emerg_fx_monthly
})

print(f"✅ {len(df_monthly.columns)} factors, {len(df_monthly)} months")
print(f"   Missing: GLOBAL_TERM={df_monthly['GLOBAL_TERM'].isna().sum()}, GLOBAL_AGG={df_monthly['GLOBAL_AGG'].isna().sum()}, INFL_LINK={df_monthly['INFL_LINK'].isna().sum()}, CORP_CREDIT={df_monthly['CORP_CREDIT'].isna().sum()}, EMERG_DEBT={df_monthly['EMERG_DEBT'].isna().sum()}, EMERG_FX={df_monthly['EMERG_FX'].isna().sum()}")
print(f"   Last 2: {global_term_monthly.index[-2].strftime('%Y-%m-%d')}, {global_term_monthly.index[-1].strftime('%Y-%m-%d')}")
print(f"   GLOBAL_TERM: {global_term_monthly.iloc[-2]:.4f}, {global_term_monthly.iloc[-1]:.4f}")
print(f"   GLOBAL_AGG: {global_agg_monthly.iloc[-2]:.4f}, {global_agg_monthly.iloc[-1]:.4f}")
print(f"   INFL_LINK: {infl_link_monthly.iloc[-2]:.4f}, {infl_link_monthly.iloc[-1]:.4f}")
print(f"   CORP_CREDIT: {corp_credit_monthly.iloc[-2]:.4f}, {corp_credit_monthly.iloc[-1]:.4f}")
print(f"   EMERG_DEBT: {emerg_debt_monthly.iloc[-2]:.4f}, {emerg_debt_monthly.iloc[-1]:.4f}")
print(f"   EMERG_FX: {emerg_fx_monthly.iloc[-2]:.4f}, {emerg_fx_monthly.iloc[-1]:.4f}")

all_dfs.append(df_monthly)

# ============================================================================
# MERGE ALL
# ============================================================================

print("\n" + "=" * 80)
print("MERGING")
print("=" * 80)

all_factors = pd.concat(all_dfs, axis=1, join='outer')
# ============================================================================
# PRINT FACTOR NAMES (ONLY)
# ============================================================================

print("\n" + "=" * 80)
print("FACTOR LIST (NAMES ONLY)")
print("=" * 80)

factor_names = sorted(all_factors.columns.tolist())

for i, name in enumerate(factor_names, 1):
    print(f"{i:02d}. {name}")

print("=" * 80)
print(f"TOTAL FACTORS: {len(factor_names)}")
print("=" * 80)
print(f"✅ {len(all_factors.columns)} factors, {len(all_factors)} months")
print(f"📅 {all_factors.index.min()} to {all_factors.index.max()}")

# ============================================================================
# SAVE
# ============================================================================

all_factors.to_parquet(OUTPUT_DIR / "all_factors_monthly.parquet")
print(f"\n💾 Saved: all_factors_monthly.parquet")
print("=" * 80)