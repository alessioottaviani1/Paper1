"""
================================================================================
Script 3b: Simulazione Trading Strategy CDS-Bond Basis — Signal-Weighted
================================================================================
Versione con pesatura condizionale ispirata a Rebonato & Ronzani (2021),
"Is Convexity Efficiently Priced?", EDHEC Working Paper.

DIFFERENZA RISPETTO A 02c (equal-weight):
  In 02c, l'indice è la media equal-weighted dei return dei trade aperti.
  Qui, ogni trade è pesato da |basis_entry - theta_t|, dove theta_t è la
  media expanding (o rolling) della basis aggregata fino al giorno t.

COSA RIMANE IDENTICO A 02c:
  - Tutta la simulazione trading (soglie, maturity rules, eligibilità ticker, ecc.)
  - Il calcolo del P&L giornaliero per ogni trade
  - La logica di apertura/chiusura trade
  - MultiIndex optimizations

COSA CAMBIA:
  - Step 7: l'indice usa pesi proporzionali al segnale invece di equal-weight
  - Nuovo parametro: THETA_WINDOW_TYPE e THETA_MIN_MONTHS
  - Output salvati in cartella separata (cds_bond_basis_rebonato/)

Riferimento: Rebonato & Ronzani (2021), Section 3.2.2, Eq. 14-16

Author: Alessio Ottaviani
Date: December 2025
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI STRATEGIA - IDENTICI A 02c (MODIFICA QUESTI VALORI)
# ============================================================================

# === ENTRY THRESHOLDS ===
ENTRY_THRESHOLD = -40            # bps - entra se basis < questo (più negativa)

# === EXIT THRESHOLDS ===
EXIT_THRESHOLD = 0               # bps - target per chiudere (verso 0)

# === OUTLIER FILTERS ===
MAX_BASIS_ENTRY = 200            # bps - ignora basis > questo (errore dato)
MIN_BASIS_ENTRY = -500           # bps - ignora basis < questo (errore dato)

# === MATURITY RULES ===
MIN_MONTHS_TO_MATURITY_ENTRY = 6 # mesi - non entrare se < 6 mesi a scadenza

# === HOLDING RULES ===
MIN_OPEN_TRADES = 1              # Dopo primo trade, mantieni sempre almeno N trade

# === MULTIPLE ENTRIES ===
ALLOW_MULTIPLE_ENTRIES = True    # True/False - permetti più trade sullo stesso ISIN
REENTRY_BASIS_WIDENING = 30      # bps - quanto deve allargarsi la basis per rientrare
MAX_CONCURRENT_PER_ISIN = 3      # Max trade attivi sullo stesso ISIN (None = no cap)

# === DATA QUALITY FILTERS ===
COOLDOWN_DAYS_AFTER_EXIT = 20    # Trading days minimi tra exit e re-entry sullo stesso ISIN (None = no cooldown)
MAX_LIFETIME_TRADES_PER_ISIN = 5  # Max trade totali sullo stesso ISIN nell'intero campione (None = no cap)
MIN_TRADE_DURATION_DAYS = 3      # Trade con durata < N trading days esclusi ex-post dall'indice (None = no filter)

# === FREQUENCY ===
ENTRY_CHECK_FREQ = "daily"       # "daily", "weekly", "monthly"
INDEX_FREQ = "daily"             # "daily", "weekly", "monthly"

# =====================================================================
# OUTPUT SELECTION 
# =====================================================================
OUTPUT_RETURN_MODE = "SW"   # "EW" oppure "SW"

# ============================================================================
# TRANSACTION FEES — 3 livelli basati su iTraxx Main 5Y
# ============================================================================
# La fee per ogni trade viene determinata dal livello di iTraxx Main alla data
# di entrata (fee entrata) e alla data di uscita (fee uscita, solo se non MATURITY).
# Le soglie vanno calibrate sui terzili storici della serie iTraxx Main.
#
# Fee entrata = entry_dv01 * fee_bps_entry / 100
# Fee uscita  = exit_dv01  * fee_bps_exit  / 100  (solo se exit_reason != MATURITY)
# Fee totale spalmata su len(trade_dates)-1 giorni di trading effettivi.

# --- Livelli di fee (modifica questi valori) ---
FEE_LOW_BPS  = 4.0    # bps - iTraxx Main sotto soglia bassa
FEE_MID_BPS  = 7.0    # bps - iTraxx Main tra le due soglie
FEE_HIGH_BPS = 10.0   # bps - iTraxx Main sopra soglia alta

# --- Soglie iTraxx Main in bps (modifica questi valori) ---
ITRAXX_MAIN_LOW_THRESHOLD  = 60.0   # sotto questa -> FEE_LOW_BPS
ITRAXX_MAIN_HIGH_THRESHOLD = 100.0  # sopra questa -> FEE_HIGH_BPS
                                     # tra le due   -> FEE_MID_BPS

# --- Path file iTraxx Main ---
ITRAXX_MAIN_FILE = Path(r"C:\Users\aless\Desktop\THESIS\data\external\Factors\Tradable_corporate_bond_factors.xlsx")

# ============================================================================
# PARAMETRI REBONATO - NUOVI
# ============================================================================

THETA_WINDOW_TYPE = "expanding"  # "expanding" (Eq. 15) o "rolling" (Eq. 16)
THETA_ROLLING_WINDOW = 252       # giorni - usato solo se THETA_WINDOW_TYPE = "rolling"
THETA_MIN_MONTHS = 6             # mesi - warm-up period prima del signal-weighting

# === SIGNAL SCALING ===
NORMALIZE_SIGNAL = True          # True: weight = signal / mean(signal_history)
                                 # Porta il peso medio a ~1 (nozionale "normale")
                                 # False: weight = signal grezzo (|basis - theta|)

SIGNAL_FLOOR = 0.0               # 0 = disattivato. Peso minimo dopo normalizzazione.
                                 # Es. 0.25 = ogni trade pesa almeno 25% del medio
SIGNAL_CAP = 0.0                 # 0 = disattivato. Peso massimo dopo normalizzazione.
                                 # Es. 4.0 = nessun trade pesa più di 4x il medio

# ============================================================================
# PATHS - Output separato da 02c
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "cds_bond_basis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")

def print_metric(label, value, unit="", decimals=0):
    if isinstance(value, (int, np.integer)):
        print(f"   {label}: {value:,}{unit}")
    elif isinstance(value, (float, np.floating)):
        if decimals == 0:
            print(f"   {label}: {value:,.0f}{unit}")
        else:
            print(f"   {label}: {value:,.{decimals}f}{unit}")
    else:
        print(f"   {label}: {value}{unit}")

def months_to_maturity(date, maturity_date):
    if date > maturity_date:
        return 0
    months = (maturity_date.year - date.year) * 12 + (maturity_date.month - date.month)
    return months

# ============================================================================
# STEP 1: CARICA DATI
# ============================================================================

print_header("CDS-BOND BASIS TRADING STRATEGY [REBONATO SIGNAL-WEIGHTED]")
print_header("STEP 1: Caricamento dati processati")

bonds_df = pd.read_parquet(PROCESSED_DATA_DIR / "cds_bond_basis_long.parquet")

print(f"✅ Bond data loaded: {len(bonds_df):,} rows")
print_metric("ISIN unici", bonds_df['ISIN'].nunique())
print_metric("Ticker unici", bonds_df['Ticker'].nunique())
print_metric("Date range", f"{bonds_df['date'].min().strftime('%Y-%m-%d')} → {bonds_df['date'].max().strftime('%Y-%m-%d')}")

# Carica Main composition
series_df = pd.read_csv(PROCESSED_DATA_DIR / "itraxx_main_series.csv")
series_df['start_date'] = pd.to_datetime(series_df['start_date'])
series_df['maturity_date'] = pd.to_datetime(series_df['maturity_date'])
series_df['tickers'] = series_df['tickers'].str.split('|')
series_df['tickers_set'] = series_df['tickers'].apply(set)

print(f"\n✅ Main composition loaded: {len(series_df)} series")

# ============================================================================
# STEP 2: PREPARA DATI
# ============================================================================

print_header("STEP 2: Preparazione dati")

bonds_df = bonds_df.sort_values('date', kind='mergesort').reset_index(drop=True)

dup_count = bonds_df.duplicated(['date', 'ISIN']).sum()
if dup_count > 0:
    print(f"⚠️  Duplicati (date, ISIN) trovati: {dup_count:,} — rimozione keep='first'...")
    bonds_df = bonds_df.drop_duplicates(['date', 'ISIN'], keep='first')
    print(f"   ✅ Dopo dedup: {len(bonds_df):,} righe")
else:
    print(f"✅ Nessun duplicato (date, ISIN)")

# MultiIndex per lookup O(1)
bonds_mi = bonds_df.set_index(['date', 'ISIN']).sort_index()

all_dates = bonds_df['date'].unique()
all_dates = pd.to_datetime(all_dates)
all_dates = pd.DatetimeIndex(sorted(all_dates))

print_metric("Giorni totali", len(all_dates))

if all_dates.isna().any():
    all_dates = all_dates[~all_dates.isna()]
    print_metric("Giorni dopo cleanup", len(all_dates))

# --- Carica iTraxx Main per fee dinamiche ---
print(f"\n📂 Caricamento iTraxx Main per fee dinamiche...")
_itrx_raw = pd.read_excel(
    ITRAXX_MAIN_FILE,
    sheet_name="CDS_INDEX",
    skiprows=18
)
_itrx_raw = _itrx_raw.iloc[:, [4, 7]]
_itrx_raw.columns = ["Date", "ITRX_MAIN"]
_itrx_raw["Date"] = pd.to_datetime(_itrx_raw["Date"], errors="coerce")
_itrx_raw = _itrx_raw.dropna(subset=["Date"]).set_index("Date")
itrx_main_daily = pd.to_numeric(_itrx_raw["ITRX_MAIN"], errors="coerce").dropna()

_q33 = itrx_main_daily.quantile(0.33)
_q67 = itrx_main_daily.quantile(0.67)
print(f"✅ iTraxx Main caricato: {len(itrx_main_daily)} osservazioni")
print(f"   Range: {itrx_main_daily.min():.1f} - {itrx_main_daily.max():.1f} bps")
print(f"   Terzile 33%: {_q33:.1f} bps  |  Terzile 67%: {_q67:.1f} bps")
print(f"   Soglie configurate: LOW={ITRAXX_MAIN_LOW_THRESHOLD:.1f}  HIGH={ITRAXX_MAIN_HIGH_THRESHOLD:.1f}")

def get_fee_bps(date):
    """
    Restituisce il livello di fee in bps in base al livello di iTraxx Main
    alla data specificata. Usa forward-fill se la data esatta non e' disponibile.
    """
    available = itrx_main_daily[itrx_main_daily.index <= date]
    if len(available) == 0:
        return FEE_MID_BPS
    level = available.iloc[-1]
    if level < ITRAXX_MAIN_LOW_THRESHOLD:
        return FEE_LOW_BPS
    elif level >= ITRAXX_MAIN_HIGH_THRESHOLD:
        return FEE_HIGH_BPS
    else:
        return FEE_MID_BPS

# ============================================================================
# STEP 3: GENERA DATE DI CHECK ENTRY + CALCOLA THETA GIORNALIERO
# ============================================================================

print_header("STEP 3: Generazione date di controllo + calcolo theta")

if ENTRY_CHECK_FREQ == "daily":
    check_dates = all_dates
elif ENTRY_CHECK_FREQ == "weekly":
    check_dates = all_dates[all_dates.weekday == 4]
elif ENTRY_CHECK_FREQ == "monthly":
    check_dates = pd.DatetimeIndex(pd.Series(all_dates).groupby([all_dates.year, all_dates.month]).last().values)
else:
    raise ValueError(f"ENTRY_CHECK_FREQ non valida: {ENTRY_CHECK_FREQ}")

check_dates_set = set(check_dates)

print_metric("Frequenza entry check", ENTRY_CHECK_FREQ)
print_metric("Date di check", f"{len(check_dates)} su {len(all_dates)} giorni")

# --- CALCOLO THETA PER-ISIN GIORNALIERO (Rebonato Eq. 15-16) ---
# Ogni ISIN ha il proprio theta_t = expanding (o rolling) mean della propria
# basis storica. Questo è più appropriato per il CDS-Bond basis dove ogni bond
# ha una storia indipendente, a differenza del mercato swap dove Rebonato ha
# un unico livello di curvatura per l'intero mercato.

print(f"\n📐 Calcolo theta per-ISIN ({THETA_WINDOW_TYPE} window)...")

# Pivot: da long format a wide (date x ISIN) per la basis
basis_wide = bonds_df.pivot_table(index='date', columns='ISIN', values='Basis', aggfunc='first')
basis_wide = basis_wide.reindex(all_dates)

# Calcola theta per ogni ISIN: expanding o rolling mean della propria serie
if THETA_WINDOW_TYPE == "expanding":
    theta_isin_daily = basis_wide.expanding(min_periods=1).mean()
elif THETA_WINDOW_TYPE == "rolling":
    theta_isin_daily = basis_wide.rolling(window=THETA_ROLLING_WINDOW, min_periods=1).mean()
else:
    raise ValueError(f"THETA_WINDOW_TYPE non valido: {THETA_WINDOW_TYPE}")

# Calcola la prima data con dati validi per ogni ISIN (per warm-up per-ISIN)
isin_first_date = basis_wide.apply(lambda col: col.dropna().index.min())
# Data di maturità del warm-up per ogni ISIN
isin_theta_start = isin_first_date + pd.DateOffset(months=THETA_MIN_MONTHS)

n_isins = len(basis_wide.columns)
theta_means = theta_isin_daily.iloc[-1].dropna()

print(f"✅ Theta calcolato per {n_isins} ISIN")
print(f"📊 Theta medio (ultimo giorno, cross-ISIN): {theta_means.mean():.2f} bps")
print(f"📊 Theta mediano (ultimo giorno, cross-ISIN): {theta_means.median():.2f} bps")
print(f"📊 ISIN con theta maturo (>{THETA_MIN_MONTHS} mesi storia): {(isin_theta_start <= all_dates[-1]).sum()}")

# ============================================================================
# STEP 4: FUNZIONE PER VERIFICARE ELIGIBILITÀ TICKER
# ============================================================================

print_header("STEP 4: Setup funzione eligibilità ticker")

def is_ticker_eligible(ticker, trade_date, bond_maturity, series_df):
    active_series = series_df[
        (series_df['start_date'] <= trade_date) & 
        (series_df['maturity_date'] >= trade_date)
    ]
    if len(active_series) == 0:
        return False, None
    
    series_with_ticker = active_series[
        active_series['tickers_set'].apply(lambda s: ticker in s)
    ]
    if len(series_with_ticker) == 0:
        return False, None
    
    last_series_maturity = series_with_ticker['maturity_date'].max()
    if bond_maturity >= last_series_maturity:
        return False, last_series_maturity
    
    return True, last_series_maturity

print("✅ Funzione eligibilità configurata")

test_ticker = bonds_df['Ticker'].iloc[0]
test_date = pd.Timestamp('2010-05-15')
test_maturity = pd.Timestamp('2013-06-20')
eligible, last_mat = is_ticker_eligible(test_ticker, test_date, test_maturity, series_df)
print(f"\n   Test: {test_ticker} @ {test_date.strftime('%Y-%m-%d')}")
print(f"   Eligible: {eligible}, Last series maturity: {last_mat}")

# ============================================================================
# STEP 5: SIMULAZIONE TRADING (IDENTICA A 02c, con aggiunta signal)
# ============================================================================

print_header("STEP 5: Simulazione trading")

trades_log = []
trade_id_counter = 1
first_trade_opened = False
open_trades = {}

# --- Data quality tracking ---
isin_last_exit_date = {}    # ISIN -> last exit date (for cooldown)
isin_lifetime_count = {}    # ISIN -> total trades opened (for max lifetime)

print("🔄 Inizio simulazione...\n")

for i, date in enumerate(all_dates):
    
    if i % 500 == 0:
        print(f"   Processing: {date.strftime('%Y-%m-%d')} ({i}/{len(all_dates)})")
    
    # === STEP 5.1: AGGIORNA P&L E CHECK EXIT PER TRADE APERTI ===
    
    trades_to_close = []
    
    for trade_id, trade in open_trades.items():
        isin = trade['isin']
        ticker = trade['ticker']
        
        try:
            row = bonds_mi.loc[(date, isin)]
        except KeyError:
            row = None
        
        if row is None:
            if date >= trade['maturity_date']:
                trade['exit_date'] = date
                trade['exit_basis'] = 0
                trade['exit_dv01'] = 0.0   # bond scaduto
                trade['exit_reason'] = 'MATURITY'
                trades_to_close.append(trade_id)
                continue
            else:
                continue
        
        current_basis = row['Basis']
        current_dv01 = row['DV01']
        dv01_observed = pd.notna(current_dv01)

        if pd.isna(current_basis):
            if date >= trade['maturity_date']:
                trade['exit_date'] = date
                trade['exit_basis'] = 0
                trade['exit_dv01'] = 0.0   # bond scaduto
                trade['exit_reason'] = 'MATURITY'
                trades_to_close.append(trade_id)
                continue
            else:
                continue

        if pd.isna(current_dv01):
            current_dv01 = trade.get('prev_dv01', trade['entry_dv01'])
        
        prev_basis = trade.get('prev_basis', trade['entry_basis'])
        days_elapsed = (date - trade['prev_date']).days if 'prev_date' in trade else 1
        if days_elapsed <= 0:
            trade['prev_basis'] = current_basis
            trade['prev_date'] = date
            continue
        
        sign = -1
        pnl_computed = False
        if pd.notna(prev_basis) and pd.notna(current_basis) and pd.notna(current_dv01):
            capital_gain_bps = sign * (prev_basis - current_basis) * current_dv01
            carry_bps = sign * prev_basis * (days_elapsed / 365.0)
            daily_pnl_bps = capital_gain_bps + carry_bps
            daily_pnl_pct = daily_pnl_bps / 100.0
            
            trade['cumulative_pnl'] = trade.get('cumulative_pnl', 0) + daily_pnl_pct
            trade['cumulative_capital_gain'] = trade.get('cumulative_capital_gain', 0) + (capital_gain_bps / 100.0)
            trade['cumulative_carry'] = trade.get('cumulative_carry', 0) + (carry_bps / 100.0)
            pnl_computed = True

        trade['prev_basis'] = current_basis
        trade['prev_date'] = date

        if pnl_computed and dv01_observed:
            trade['prev_dv01'] = current_dv01

        # Check exit
        months_left = months_to_maturity(date, trade['maturity_date'])
        
        if months_left == 0 or date >= trade['maturity_date']:
            trade['exit_date'] = date
            trade['exit_basis'] = 0
            trade['exit_dv01'] = 0.0   # bond scaduto
            trade['exit_reason'] = 'MATURITY'
            trades_to_close.append(trade_id)
            continue
        
        can_close = len(open_trades) > MIN_OPEN_TRADES or not first_trade_opened
        
        if can_close:
            if current_basis > EXIT_THRESHOLD:
                trade['exit_date'] = date
                trade['exit_basis'] = current_basis
                trade['exit_dv01'] = current_dv01
                trade['exit_reason'] = 'TARGET_HIT'
                trades_to_close.append(trade_id)
    
    for trade_id in trades_to_close:
        trade = open_trades.pop(trade_id)
        trades_log.append(trade.copy())
        # Track last exit date for cooldown
        isin_last_exit_date[trade['isin']] = date
    
    # === STEP 5.2: CHECK NUOVI ENTRY ===
    
    if date in check_dates_set:
        
        try:
            today_bonds = bonds_mi.loc[date].reset_index()
        except KeyError:
            today_bonds = pd.DataFrame(columns=bonds_df.columns)
        
        candidates = []
        
        for bond in today_bonds.itertuples(index=False):
            isin = bond.ISIN
            ticker = bond.Ticker
            basis = bond.Basis
            dv01 = bond.DV01
            maturity = bond.Maturity
            
            if pd.isna(basis) or pd.isna(dv01) or pd.isna(maturity):
                continue
            if basis > MAX_BASIS_ENTRY or basis < MIN_BASIS_ENTRY:
                continue
            if basis >= 0:
                continue
            
            months_left = months_to_maturity(date, maturity)
            if months_left < MIN_MONTHS_TO_MATURITY_ENTRY:
                continue
            
            eligible, last_series_mat = is_ticker_eligible(ticker, date, maturity, series_df)
            if not eligible:
                continue
            
            # Check multiple entries
            if not ALLOW_MULTIPLE_ENTRIES:
                already_open = any(t['isin'] == isin for t in open_trades.values())
                if already_open:
                    continue
            else:
                if REENTRY_BASIS_WIDENING > 0:
                    last_entry_basis = None
                    for t in open_trades.values():
                        if t['isin'] == isin:
                            if last_entry_basis is None:
                                last_entry_basis = t['entry_basis']
                            else:
                                if t['entry_basis'] < last_entry_basis:
                                    last_entry_basis = t['entry_basis']
                    
                    if last_entry_basis is not None:
                        if basis > last_entry_basis - REENTRY_BASIS_WIDENING:
                            continue
            
            # Cap max trade contemporanei sullo stesso ISIN
            if MAX_CONCURRENT_PER_ISIN is not None:
                n_open_same = sum(1 for t in open_trades.values() if t['isin'] == isin)
                if n_open_same >= MAX_CONCURRENT_PER_ISIN:
                    continue
            
            # Cooldown post-uscita: non rientrare su un ISIN troppo presto (trading days)
            if COOLDOWN_DAYS_AFTER_EXIT is not None and isin in isin_last_exit_date:
                trading_days_since_exit = ((all_dates > isin_last_exit_date[isin]) & (all_dates <= date)).sum()
                if trading_days_since_exit < COOLDOWN_DAYS_AFTER_EXIT:
                    continue
            
            # Max lifetime trades per ISIN
            if MAX_LIFETIME_TRADES_PER_ISIN is not None:
                if isin_lifetime_count.get(isin, 0) >= MAX_LIFETIME_TRADES_PER_ISIN:
                    continue
            
            # Check soglia entry
            if basis < ENTRY_THRESHOLD:
                # --- REBONATO: calcola il segnale all'entry (theta per-ISIN) ---
                theta_t = np.nan
                if isin in theta_isin_daily.columns and date in theta_isin_daily.index:
                    theta_t = theta_isin_daily.loc[date, isin]
                
                if pd.notna(theta_t):
                    signal = abs(basis - theta_t)
                else:
                    signal = 1.0  # fallback se theta non disponibile
                
                # Warm-up per-ISIN: se questo ISIN ha meno di THETA_MIN_MONTHS di storia
                isin_mature = False
                if isin in isin_theta_start.index:
                    isin_start = isin_theta_start[isin]
                    if pd.notna(isin_start) and date >= isin_start:
                        isin_mature = True
                
                if not isin_mature:
                    signal = 1.0
                
                candidates.append({
                    'isin': isin,
                    'ticker': ticker,
                    'basis': basis,
                    'dv01': dv01,
                    'maturity': maturity,
                    'abs_basis': abs(basis),
                    'last_series_maturity': last_series_mat,
                    'signal': signal,
                    'theta_t': theta_t if pd.notna(theta_t) else np.nan
                })
        
        if candidates:
            candidates.sort(key=lambda x: x['abs_basis'], reverse=True)
            best = candidates[0]
            
            new_trade = {
                'trade_id': trade_id_counter,
                'isin': best['isin'],
                'ticker': best['ticker'],
                'entry_date': date,
                'entry_basis': best['basis'],
                'entry_dv01': best['dv01'],
                'maturity_date': best['maturity'],
                'last_series_maturity': best['last_series_maturity'],
                'exit_date': None,
                'exit_basis': None,
                'exit_reason': None,
                'cumulative_pnl': 0,
                'cumulative_capital_gain': 0,
                'cumulative_carry': 0,
                'prev_basis': best['basis'],
                'prev_date': date,
                # Rebonato signal fields
                'signal': best['signal'],
                'theta_at_entry': best['theta_t']
            }
            
            open_trades[trade_id_counter] = new_trade
            isin_lifetime_count[best['isin']] = isin_lifetime_count.get(best['isin'], 0) + 1
            trade_id_counter += 1
            first_trade_opened = True

# === CHIUDI TRADE ANCORA APERTI ALLA FINE DEL SAMPLE ===

print(f"\n🔍 Trade ancora aperti alla fine: {len(open_trades)}")

last_date = all_dates[-1]

for trade_id in list(open_trades.keys()):
    trade = open_trades[trade_id]
    isin = trade['isin']
    
    if last_date >= trade['maturity_date']:
        bond_data_last = bonds_df[(bonds_df['ISIN'] == isin) & (bonds_df['date'] <= trade['maturity_date'])]
        
        if len(bond_data_last) > 0:
            bond_data_last = bond_data_last.sort_values('date').iloc[-1]
            last_dv01 = bond_data_last['DV01']
            last_basis = bond_data_last['Basis']
            last_data_date = bond_data_last['date']
            
            if pd.notna(last_basis) and pd.notna(last_dv01):
                days_to_maturity = (trade['maturity_date'] - last_data_date).days
                sign = -1
                capital_gain_bps = sign * (last_basis - 0) * last_dv01
                carry_bps = sign * last_basis * (days_to_maturity / 365.0)
                final_pnl_bps = capital_gain_bps + carry_bps
                final_pnl_pct = final_pnl_bps / 100.0
                
                trade['cumulative_pnl'] = trade.get('cumulative_pnl', 0) + final_pnl_pct
                trade['cumulative_capital_gain'] = trade.get('cumulative_capital_gain', 0) + (capital_gain_bps / 100.0)
                trade['cumulative_carry'] = trade.get('cumulative_carry', 0) + (carry_bps / 100.0)
        
        trade['exit_date'] = trade['maturity_date']
        trade['exit_basis'] = 0
        trade['exit_dv01'] = 0.0   # bond scaduto
        trade['exit_reason'] = 'MATURITY'
    else:
        trade['exit_date'] = last_date
        try:
            row = bonds_mi.loc[(last_date, isin)]
            exit_basis = row['Basis']
            if pd.isna(exit_basis):
                exit_basis = trade.get('prev_basis', 0)
            exit_dv01 = row['DV01']
            if pd.isna(exit_dv01):
                exit_dv01 = trade.get('prev_dv01', trade['entry_dv01'])
        except KeyError:
            exit_basis = trade.get('prev_basis', 0)
            exit_dv01 = trade.get('prev_dv01', trade['entry_dv01'])
        
        trade['exit_basis'] = exit_basis
        trade['exit_dv01'] = exit_dv01
        trade['exit_reason'] = 'END_OF_SAMPLE'
    
    open_trades.pop(trade_id)
    trades_log.append(trade.copy())

print(f"✅ Simulazione completata!")
print_metric("Trade totali eseguiti", len(trades_log))

# ============================================================================
# STEP 6: CREA DATAFRAME TRADES
# ============================================================================

print_header("STEP 6: Creazione log dei trade")

if len(trades_log) == 0:
    print("❌ ERRORE: Nessun trade eseguito con questi parametri!")
    exit()

trades_df = pd.DataFrame(trades_log)
trades_df['duration_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
trades_df['duration_trading_days'] = trades_df.apply(
    lambda r: ((all_dates >= r['entry_date']) & (all_dates <= r['exit_date'])).sum(), axis=1)
trades_df['pnl_pct'] = trades_df['cumulative_pnl']

# Assicura che exit_dv01 esista (fallback a entry_dv01 se mancante)
if 'exit_dv01' not in trades_df.columns:
    trades_df['exit_dv01'] = trades_df['entry_dv01']
trades_df['exit_dv01'] = trades_df['exit_dv01'].fillna(trades_df['entry_dv01'])

print_metric("Trade log creato", f"{len(trades_df)} trade")

print(f"\n📈 Statistiche Trade:")
print_metric("Durata media", f"{trades_df['duration_days'].mean():.0f} giorni")
print_metric("P&L medio", f"{trades_df['pnl_pct'].mean():.2f}%")

print(f"\n📊 Exit reasons:")
for reason, count in trades_df['exit_reason'].value_counts().items():
    print_metric(f"   {reason}", f"{count} ({count/len(trades_df)*100:.1f}%)")

print(f"\n📊 Top 5 Ticker per numero trade:")
top_tickers = trades_df['ticker'].value_counts().head(5)
for ticker, count in top_tickers.items():
    print(f"   {ticker}: {count} trade")

# --- REBONATO: Stampa statistiche segnale ---
print(f"\n📐 REBONATO SIGNAL STATISTICS (theta per-ISIN):")
print(f"   Signal medio: {trades_df['signal'].mean():.2f}")
print(f"   Signal mediano: {trades_df['signal'].median():.2f}")
print(f"   Signal min: {trades_df['signal'].min():.2f}")
print(f"   Signal max: {trades_df['signal'].max():.2f}")
print(f"   Theta medio all'entry: {trades_df['theta_at_entry'].mean():.2f} bps")
warm_up_trades = trades_df[trades_df['signal'] == 1.0]
signal_trades = trades_df[trades_df['signal'] != 1.0]
print(f"   Trade in warm-up (peso=1): {len(warm_up_trades)}")
print(f"   Trade signal-weighted: {len(signal_trades)}")

# Debug: primi 10 trade
print(f"\n🔍 DEBUG: Primi 10 Trade (con Rebonato signal):")
print(trades_df[['isin', 'ticker', 'entry_date', 'exit_date', 'entry_basis', 'exit_basis', 
                 'duration_days', 'cumulative_pnl', 'exit_reason',
                 'signal', 'theta_at_entry']].head(10).to_string(index=False))

# Salva trades log
trades_path = RESULTS_DIR / "trades_log.csv"
trades_df.to_csv(trades_path, index=False)
print(f"\n💾 Salvato: {trades_path.name}")

# ============================================================================
# STEP 7: COSTRUZIONE INDICE - SIGNAL-WEIGHTED (REBONATO)
# ============================================================================

print_header("STEP 7: Costruzione indice dei return [REBONATO SIGNAL-WEIGHTED]")

# --- Ex-post filter: rimuovi trade troppo brevi (probabile rumore dati) ---
if MIN_TRADE_DURATION_DAYS is not None:
    n_before = len(trades_df)
    short_trade_ids = trades_df[trades_df['duration_trading_days'] < MIN_TRADE_DURATION_DAYS]['trade_id'].tolist()
    trades_df_index = trades_df[trades_df['duration_trading_days'] >= MIN_TRADE_DURATION_DAYS].copy()
    n_removed = n_before - len(trades_df_index)
    print(f"   ⚠️ Filtro MIN_TRADE_DURATION_DAYS={MIN_TRADE_DURATION_DAYS} (trading days): "
          f"rimossi {n_removed}/{n_before} trade (durata < {MIN_TRADE_DURATION_DAYS} trading days)")
else:
    trades_df_index = trades_df.copy()
    short_trade_ids = []

# --- Costruisci pesi con normalizzazione e clip opzionali ---
# Ordina trade per entry_date per calcolare media expanding dei segnali
trades_sorted = trades_df_index.sort_values('entry_date').copy()

# Calcola il peso per ogni trade
trade_weights = {}
signal_history = []  # accumula segnali per normalizzazione expanding

for _, trade in trades_sorted.iterrows():
    raw_signal = trade['signal']
    weight = raw_signal
    
    # Step 1: Normalizzazione (se attiva)
    if NORMALIZE_SIGNAL and raw_signal != 1.0:  # 1.0 = warm-up, non normalizzare
        if len(signal_history) > 0:
            avg_signal = np.mean(signal_history)
            if avg_signal > 0:
                weight = raw_signal / avg_signal
            else:
                weight = 1.0
        else:
            weight = 1.0  # primo trade signal-weighted: peso = 1
        signal_history.append(raw_signal)
    
    # Step 2: Clip (se attivi, applicati dopo normalizzazione)
    if SIGNAL_FLOOR > 0 and weight != 1.0:  # non clippare warm-up
        weight = max(weight, SIGNAL_FLOOR)
    if SIGNAL_CAP > 0 and weight != 1.0:
        weight = min(weight, SIGNAL_CAP)
    
    trade_weights[f"trade_{int(trade['trade_id'])}"] = weight

weight_values = list(trade_weights.values())
print(f"📐 Pesi assegnati a {len(trade_weights)} trade")
print(f"   Peso min: {min(weight_values):.4f}")
print(f"   Peso max: {max(weight_values):.4f}")
print(f"   Peso medio: {np.mean(weight_values):.4f}")
print(f"   Peso mediano: {np.median(weight_values):.4f}")
if NORMALIZE_SIGNAL:
    print(f"   [Normalizzazione ATTIVA: peso medio ≈ 1]")
if SIGNAL_FLOOR > 0 or SIGNAL_CAP > 0:
    print(f"   [Clip: floor={SIGNAL_FLOOR}, cap={SIGNAL_CAP}]")

# Ricostruisci P&L giornaliero per ogni trade (identico a 02c)
daily_returns = pd.DataFrame(index=all_dates)

for _, trade in trades_df_index.iterrows():
    trade_id = trade['trade_id']
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    isin = trade['isin']
    
    trade_dates = all_dates[(all_dates >= entry_date) & (all_dates <= exit_date)]
    if len(trade_dates) == 0:
        continue
    
    try:
        trade_bond_data = bonds_mi.xs(isin, level='ISIN')
    except KeyError:
        continue

    # -------------------------------------------------------------------------
    # CALCOLO FEE TOTALE DEL TRADE
    # Fee entrata = entry_dv01 * fee_bps(entry_date) / 100  (sempre)
    # Fee uscita  = exit_dv01  * fee_bps(exit_date)  / 100  (solo se non MATURITY)
    # La fee totale viene spalmata SOLO sui giorni con return valido (non NaN),
    # cosi' i giorni con basis mancante restano NaN e non distorcono la media EW,
    # ma la fee totale pagata e' comunque esattamente quella attesa.
    # -------------------------------------------------------------------------
    fee_bps_entry = get_fee_bps(entry_date)
    fee_entry_pct = trade['entry_dv01'] * fee_bps_entry / 100.0
    if trade['exit_reason'] != 'MATURITY':
        fee_bps_exit = get_fee_bps(exit_date)
        fee_exit_pct = trade['exit_dv01'] * fee_bps_exit / 100.0
    else:
        fee_bps_exit = 0.0
        fee_exit_pct = 0.0
    fee_total_pct = fee_entry_pct + fee_exit_pct

    trade_returns = []
    prev_basis = trade['entry_basis']
    prev_date = entry_date
    prev_dv01 = trade['entry_dv01']

    if trade_id == 1:
        print(f"\n🔍 DEBUG Trade #1 (primi 5 giorni):")
        print(f"   ISIN: {isin}")
        print(f"   Entry date: {entry_date}")
        print(f"   Entry basis: {prev_basis:.2f}")
        print(f"   Signal (peso): {trade['signal']:.2f}")

    # --- Passata 1: calcola return grezzi (senza fee), NaN dove mancano dati ---
    for idx, date in enumerate(trade_dates[1:]):

        if date in trade_bond_data.index:
            row = trade_bond_data.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.sort_index().iloc[0]
            current_basis = row['Basis']
            current_dv01 = row['DV01']
        else:
            current_basis = np.nan
            current_dv01 = np.nan

        if pd.isna(current_basis):
            if date >= trade['maturity_date']:
                current_basis = 0
                if pd.isna(current_dv01):
                    current_dv01 = prev_dv01
            else:
                trade_returns.append(np.nan)
                continue

        if pd.isna(current_dv01):
            current_dv01 = prev_dv01

        days_elapsed = (date - prev_date).days

        sign = -1

        if pd.notna(prev_basis) and pd.notna(current_basis) and pd.notna(current_dv01):
            capital_gain_bps = sign * (prev_basis - current_basis) * current_dv01
            carry_bps = sign * prev_basis * (days_elapsed / 365.0)
            daily_ret_bps = capital_gain_bps + carry_bps
            daily_ret_pct = daily_ret_bps / 100.0

            if trade_id == 1 and idx < 5:
                print(f"   Day {idx+1} ({date.strftime('%Y-%m-%d')}): prev={prev_basis:.2f}, curr={current_basis:.2f}, dv01={current_dv01:.4f}")
                print(f"      CG={capital_gain_bps:.4f} bps, Carry={carry_bps:.4f} bps, Total={daily_ret_pct:.6f}%")
        else:
            daily_ret_pct = np.nan
            if trade_id == 1 and idx < 5:
                print(f"   Day {idx+1} ({date.strftime('%Y-%m-%d')}): SKIPPED (missing data)")

        trade_returns.append(daily_ret_pct)
        if pd.notna(daily_ret_pct):
            prev_basis = current_basis
            prev_date = date
            prev_dv01 = current_dv01

    # --- Passata 2: spalma fee_total sui soli giorni validi ---
    n_valid = sum(1 for r in trade_returns if pd.notna(r))
    if n_valid > 0 and fee_total_pct > 0:
        daily_fee_pct = fee_total_pct / n_valid
        trade_returns = [r - daily_fee_pct if pd.notna(r) else np.nan
                         for r in trade_returns]

    daily_returns.loc[trade_dates[1:], f"trade_{trade_id}"] = trade_returns

# --- REBONATO: Calcolo indice signal-weighted ---
trade_cols = [col for col in daily_returns.columns if col.startswith('trade_')]

print(f"\n📐 Calcolo indice signal-weighted su {len(trade_cols)} trade...")

weights_matrix = pd.DataFrame(index=all_dates, columns=trade_cols, dtype=float)
for col in trade_cols:
    weight = trade_weights[col]
    active_mask = daily_returns[col].notna()
    weights_matrix.loc[active_mask, col] = weight

weighted_returns = daily_returns[trade_cols] * weights_matrix
numerator = weighted_returns.sum(axis=1, skipna=True)
denominator = weights_matrix.sum(axis=1, skipna=True)
denominator = denominator.replace(0, np.nan)

# === Signal-weighted (SW) ===
daily_returns['index_return_sw'] = numerator / denominator

# === Equal-weight (EW) ===
daily_returns['index_return_ew'] = daily_returns[trade_cols].mean(axis=1, skipna=True)

# Filtra
first_trade_entry = trades_df['entry_date'].min()
first_return_date = first_trade_entry + pd.Timedelta(days=1)
daily_returns = daily_returns[daily_returns.index >= first_return_date]

# =====================================================================
# SELECT OUTPUT SERIES (index_return = EW oppure SW)
# =====================================================================
mode = OUTPUT_RETURN_MODE.upper().strip()

if mode == "EW":
    print("\n" + "="*80)
    print("🚨🚨🚨 USING EQUAL-WEIGHT: index_return = index_return_ew 🚨🚨🚨")
    print("="*80 + "\n")
    daily_returns['index_return'] = daily_returns['index_return_ew']

elif mode == "SW":
    print("\n" + "="*80)
    print("🚨🚨🚨 USING SIGNAL-WEIGHTED: index_return = index_return_sw 🚨🚨🚨")
    print("="*80 + "\n")
    daily_returns['index_return'] = daily_returns['index_return_sw']

else:
    raise ValueError("OUTPUT_RETURN_MODE must be 'EW' or 'SW'")

print_metric("Indice costruito", f"{len(daily_returns)} giorni")
print(f"📅 Dal {daily_returns.index.min()} al {daily_returns.index.max()}")
print(f"\n✅ Primo trade entry: {first_trade_entry.strftime('%Y-%m-%d')}")
print(f"✅ Primo return index: {first_return_date.strftime('%Y-%m-%d')}")

valid_sw = daily_returns['index_return_sw'].dropna()
valid_ew = daily_returns['index_return_ew'].dropna()
valid_sel = daily_returns['index_return'].dropna()

print(f"\n📊 Return medio giornaliero (signal-weighted SW): {valid_sw.mean():.4f}%")
print(f"📊 Return medio giornaliero (equal-weight  EW):   {valid_ew.mean():.4f}%")
print(f"📊 Return medio giornaliero (SELECTED -> PCA):    {valid_sel.mean():.4f}%")

# DEBUG: Prime 20 date
print("\n" + "=" * 80)
print("🔍 DEBUG: Prime 20 Date dell'Indice (Selected vs SW vs EW)")
print("=" * 80)
print(daily_returns[['index_return', 'index_return_sw', 'index_return_ew']].head(20).to_string())

# ============================================================================
# STEP 8: CALCOLA CUMULATIVE RETURN
# ============================================================================

print_header("STEP 8: Calcolo equity curve")

# === SELECTED ===
daily_returns['index_value'] = 100 * (1 + daily_returns['index_return'] / 100).cumprod()
daily_returns['cumulative_return'] = (daily_returns['index_value'] / 100 - 1) * 100

daily_returns['index_value_sw'] = 100 * (1 + daily_returns['index_return_sw'] / 100).cumprod()
daily_returns['cumulative_return_sw'] = (daily_returns['index_value_sw'] / 100 - 1) * 100

daily_returns['index_value_ew'] = 100 * (1 + daily_returns['index_return_ew'] / 100).cumprod()
daily_returns['cumulative_return_ew'] = (daily_returns['index_value_ew'] / 100 - 1) * 100

print(f"✅ Equity curve calcolata")
print(f"📌 SELECTED MODE (PCA): {OUTPUT_RETURN_MODE.upper()}")
print(f"📈 Return totale (SELECTED): {daily_returns['cumulative_return'].iloc[-1]:.2f}%")
print(f"📈 Return totale (SW):       {daily_returns['cumulative_return_sw'].iloc[-1]:.2f}%")
print(f"📈 Return totale (EW):       {daily_returns['cumulative_return_ew'].iloc[-1]:.2f}%")
print(f"📈 Index finale (SELECTED):  {daily_returns['index_value'].iloc[-1]:.2f}")
print(f"📈 Index finale (SW):        {daily_returns['index_value_sw'].iloc[-1]:.2f}")
print(f"📈 Index finale (EW):        {daily_returns['index_value_ew'].iloc[-1]:.2f}")

# ============================================================================
# STEP 9: RESAMPLE A FREQUENZA DESIDERATA
# ============================================================================

print_header("STEP 9: Resample a frequenza indice")

if INDEX_FREQ == "daily":
    index_final = daily_returns[[
        'index_return', 'cumulative_return', 'index_value',            # SELECTED
        'index_return_sw', 'cumulative_return_sw', 'index_value_sw',   # SW
        'index_return_ew', 'cumulative_return_ew', 'index_value_ew'    # EW
    ]].copy()

elif INDEX_FREQ == "weekly":
    index_final = daily_returns[['index_return', 'index_return_sw', 'index_return_ew']].resample('W-FRI').sum()

    index_final['cumulative_return'] = index_final['index_return'].cumsum()
    index_final['index_value'] = 100 * (1 + index_final['cumulative_return'] / 100)

    index_final['cumulative_return_sw'] = index_final['index_return_sw'].cumsum()
    index_final['index_value_sw'] = 100 * (1 + index_final['cumulative_return_sw'] / 100)

    index_final['cumulative_return_ew'] = index_final['index_return_ew'].cumsum()
    index_final['index_value_ew'] = 100 * (1 + index_final['cumulative_return_ew'] / 100)
elif INDEX_FREQ == "monthly":
    index_final = daily_returns[['index_return', 'index_return_sw', 'index_return_ew']].resample('M').sum()

    index_final['cumulative_return'] = index_final['index_return'].cumsum()
    index_final['index_value'] = 100 * (1 + index_final['cumulative_return'] / 100)

    index_final['cumulative_return_sw'] = index_final['index_return_sw'].cumsum()
    index_final['index_value_sw'] = 100 * (1 + index_final['cumulative_return_sw'] / 100)

    index_final['cumulative_return_ew'] = index_final['index_return_ew'].cumsum()
    index_final['index_value_ew'] = 100 * (1 + index_final['cumulative_return_ew'] / 100)
else:
    raise ValueError(f"INDEX_FREQ non valida: {INDEX_FREQ}")

print_metric("Indice resample a", INDEX_FREQ)
print_metric("Osservazioni", len(index_final))

# ============================================================================
# STEP 10: SALVA RISULTATI
# ============================================================================

print_header("STEP 10: Salvataggio risultati")

index_path = RESULTS_DIR / f"index_{INDEX_FREQ}.csv"
index_final.to_csv(index_path)
print(f"💾 Salvato: {index_path.name}")

daily_path = RESULTS_DIR / "index_daily.csv"
daily_returns.to_csv(daily_path)
print(f"💾 Salvato: {daily_path.name}")

# Salva parametri
params_dict = {
    'FEE_LOW_BPS': FEE_LOW_BPS,
    'FEE_MID_BPS': FEE_MID_BPS,
    'FEE_HIGH_BPS': FEE_HIGH_BPS,
    'ITRAXX_MAIN_LOW_THRESHOLD': ITRAXX_MAIN_LOW_THRESHOLD,
    'ITRAXX_MAIN_HIGH_THRESHOLD': ITRAXX_MAIN_HIGH_THRESHOLD,
    'THETA_WINDOW_TYPE': THETA_WINDOW_TYPE,
    'THETA_ROLLING_WINDOW': THETA_ROLLING_WINDOW,
    'THETA_MIN_MONTHS': THETA_MIN_MONTHS,
    'NORMALIZE_SIGNAL': NORMALIZE_SIGNAL,
    'SIGNAL_FLOOR': SIGNAL_FLOOR,
    'SIGNAL_CAP': SIGNAL_CAP,
    'ENTRY_THRESHOLD': ENTRY_THRESHOLD,
    'EXIT_THRESHOLD': EXIT_THRESHOLD,
    'MAX_BASIS_ENTRY': MAX_BASIS_ENTRY,
    'MIN_BASIS_ENTRY': MIN_BASIS_ENTRY,
    'MIN_MONTHS_TO_MATURITY_ENTRY': MIN_MONTHS_TO_MATURITY_ENTRY,
    'MIN_OPEN_TRADES': MIN_OPEN_TRADES,
    'ALLOW_MULTIPLE_ENTRIES': ALLOW_MULTIPLE_ENTRIES,
    'REENTRY_BASIS_WIDENING': REENTRY_BASIS_WIDENING,
    'ENTRY_CHECK_FREQ': ENTRY_CHECK_FREQ,
    'INDEX_FREQ': INDEX_FREQ,
    'OUTPUT_RETURN_MODE': OUTPUT_RETURN_MODE,
}
params_df_out = pd.DataFrame(list(params_dict.items()), columns=['Parameter', 'Value'])
params_path = RESULTS_DIR / "parameters.csv"
params_df_out.to_csv(params_path, index=False)
print(f"💾 Salvato: {params_path.name}")

# ============================================================================
# STEP 11: STATISTICHE FINALI + CONFRONTO
# ============================================================================

print_header("STATISTICHE FINALI — CONFRONTO SIGNAL-WEIGHTED vs EQUAL-WEIGHT")

print(f"\n📊 PARAMETRI USATI:")
print_metric("Entry threshold", f"< {ENTRY_THRESHOLD} bps")
print_metric("Exit threshold", f"> {EXIT_THRESHOLD} bps")
print_metric("Min months to maturity", MIN_MONTHS_TO_MATURITY_ENTRY)
print_metric("Min open trades", MIN_OPEN_TRADES)
print_metric("Multiple entries", "Yes" if ALLOW_MULTIPLE_ENTRIES else "No")
if ALLOW_MULTIPLE_ENTRIES:
    print_metric("Reentry widening", f"{REENTRY_BASIS_WIDENING} bps")
print(f"\n💸 TRANSACTION FEES (dinamiche su iTraxx Main):")
print_metric("Fee LOW",  f"{FEE_LOW_BPS} bps  (iTraxx Main < {ITRAXX_MAIN_LOW_THRESHOLD} bps)")
print_metric("Fee MID",  f"{FEE_MID_BPS} bps  (iTraxx Main {ITRAXX_MAIN_LOW_THRESHOLD}-{ITRAXX_MAIN_HIGH_THRESHOLD} bps)")
print_metric("Fee HIGH", f"{FEE_HIGH_BPS} bps  (iTraxx Main > {ITRAXX_MAIN_HIGH_THRESHOLD} bps)")
print(f"\n📐 REBONATO PARAMETERS:")
print_metric("Theta window", THETA_WINDOW_TYPE)
if THETA_WINDOW_TYPE == "rolling":
    print_metric("Rolling window", f"{THETA_ROLLING_WINDOW} days")
print_metric("Theta min months", THETA_MIN_MONTHS)
print_metric("Normalize signal", "Yes" if NORMALIZE_SIGNAL else "No")
if SIGNAL_FLOOR > 0 or SIGNAL_CAP > 0:
    print_metric("Signal floor", SIGNAL_FLOOR)
    print_metric("Signal cap", SIGNAL_CAP)

def compute_metrics(returns_series, freq_label="daily"):
    valid = returns_series.dropna()
    if len(valid) == 0:
        return {}
    
    ann_factor = {'daily': 252, 'weekly': 52, 'monthly': 12}.get(freq_label, 252)
    
    avg_ret = valid.mean() * ann_factor
    vol = valid.std() * np.sqrt(ann_factor)
    sharpe = avg_ret / vol if vol > 0 else 0
    
    cum = valid.cumsum()
    running_max = cum.expanding().max()
    dd = cum - running_max
    max_dd = dd.min()
    
    return {
        'Avg Return (% p.a.)': avg_ret,
        'Volatility (% p.a.)': vol,
        'Sharpe (ann.)': sharpe,
        'Max Drawdown (%)': max_dd,
        'Skewness': valid.skew(),
        'Kurtosis': valid.kurtosis(),
        'N obs': len(valid)
    }

metrics_sw = compute_metrics(index_final['index_return_sw'], INDEX_FREQ)
metrics_ew = compute_metrics(index_final['index_return_ew'], INDEX_FREQ)

print(f"\n📌 SELECTED MODE: {OUTPUT_RETURN_MODE.upper()}  -> index_return")

print(f"\n{'Metric':<25} {'Signal-Weighted':>18} {'Equal-Weight':>18}")
print("=" * 65)
for key in metrics_sw:
    sw_val = metrics_sw[key]
    ew_val = metrics_ew[key]
    if key == 'N obs':
        print(f"{key:<25} {int(sw_val):>18} {int(ew_val):>18}")
    else:
        print(f"{key:<25} {sw_val:>18.4f} {ew_val:>18.4f}")

print(f"\n📈 TRADE STATISTICS:")
print_metric("Trade totali", len(trades_df))
print_metric("Durata media", f"{trades_df['duration_days'].mean():.0f} giorni")
print_metric("P&L medio per trade", f"{trades_df['pnl_pct'].mean():.2f}%")

print_header("✅ SIMULAZIONE REBONATO COMPLETATA CON SUCCESSO!")

print(f"\n📁 File salvati in: {RESULTS_DIR}/")
print(f"   • trades_log.csv")
print(f"   • index_{INDEX_FREQ}.csv (contiene sia signal-weighted che equal-weight)")
print(f"   • index_daily.csv")
print(f"   • parameters.csv")

print("\n🎯 Prossimo step:")
print("   • Confronta con 02c (equal-weight originale)")
print("   • Calcola MPPM (04_mppm_analysis.py) su questo indice")
print("   • Calcola Moreira-Muir (05_moreira_muir.py) su questo indice")
print()