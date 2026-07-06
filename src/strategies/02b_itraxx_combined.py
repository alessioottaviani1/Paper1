"""
Script 2c: Simulazione Trading Strategy iTraxx Combined — Signal-Weighted
==========================================================================
Versione con pesatura condizionale ispirata a Rebonato & Ronzani (2021),
"Is Convexity Efficiently Priced?", EDHEC Working Paper.

DIFFERENZA RISPETTO A 02b (equal-weight):
  In 02b, l'indice è la media equal-weighted dei return dei trade aperti.
  Qui, ogni trade è pesato da |basis_entry - theta_t|, dove theta_t è la
  media expanding (o rolling) della basis aggregata fino al giorno t.

COSA RIMANE IDENTICO A 02b:
  - Tutta la simulazione trading (soglie entry/exit per indice, on-the-run, ecc.)
  - Il calcolo del P&L giornaliero per ogni trade
  - La logica di apertura/chiusura trade

COSA CAMBIA:
  - Step 7: l'indice usa pesi proporzionali al segnale invece di equal-weight
  - Nuovo parametro: THETA_WINDOW_TYPE e THETA_MIN_MONTHS
  - Output salvati in cartella separata (itraxx_combined_rebonato/)

Riferimento: Rebonato & Ronzani (2021), Section 3.2.2, Eq. 14-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SELEZIONA INDICI DA COMBINARE - MODIFICA QUESTO
# ============================================================================

INDICES_TO_COMBINE = ['Main', 'SnrFin', 'SubFin', 'Xover']  # Tutti

# ============================================================================
# PARAMETRI STRATEGIA - SPECIFICI PER OGNI INDICE (IDENTICI A 02b)
# ============================================================================

INDEX_PARAMS = {
    'Main': {
        'entry_long': 5,
        'entry_short': -10,
        'exit_long': -5,
        'exit_short': 10,
        'max_basis': 50,
        'min_basis': -130,
    },
    'SnrFin': {
        'entry_long': 8,
        'entry_short': -15,
        'exit_long': -5,
        'exit_short': 5,
        'max_basis': 60,
        'min_basis': -150,
    },
    'SubFin': {
        'entry_long': 10,
        'entry_short': -20,
        'exit_long': -10,
        'exit_short': 10,
        'max_basis': 80,
        'min_basis': -180,
    },
    'Xover': {
        'entry_long': 15,
        'entry_short': -25,
        'exit_long': -15,
        'exit_short': 10,
        'max_basis': 100,
        'min_basis': -200,
    }
}

# === PARAMETRI COMUNI ===
MAX_SERIES_RANK = 1               # 1=solo on-the-run, 2=on-the-run+previous, ecc.
MIN_OPEN_TRADES = 1               # Dopo primo trade, mantieni sempre almeno N trade

# === MULTIPLE ENTRIES ===
ALLOW_MULTIPLE_ENTRIES = True    # True/False - permetti più trade sullo stesso indice+serie
REENTRY_BASIS_WIDENING = 10        # bps - quanto deve allargarsi la basis per rientrare
MAX_CONCURRENT_PER_INDEX_SERIE = None  # Max trade attivi su stessa combo indice+serie (None = no cap)

# === DATA QUALITY FILTER ===
MIN_TRADE_DURATION_DAYS = 3      # Trade con durata < N trading days esclusi ex-post dall'indice (None = no filter)

# === FREQUENCY ===
ENTRY_CHECK_FREQ = "daily"        # "daily", "weekly", "monthly"
INDEX_FREQ = "daily"              # "daily", "weekly", "monthly"

# =====================================================================
# OUTPUT SELECTION 
# =====================================================================
OUTPUT_RETURN_MODE = "SW"   # "EW" oppure "SW"

# ============================================================================
# TRANSACTION FEES — 3 livelli basati su iTraxx Main 5Y, separati per indice
# ============================================================================
# La fee modella il costo di round-trip della strategia iTraxx skew (indice vs
# basket di single names). La fee totale e' la SOMMA delle due mezze bid-offer
# (half-spread indice + half-spread single name), perche' le due esecuzioni
# avvengono su mercati separati.
#
# Gerarchia di liquidita' (dalla letteratura):
#   Main    : 125 nomi IG, mercato piu' liquido
#             Collin-Dufresne, Junge, Trolle (2020, JF): index ~0.5-1 bps,
#             single name IG ~5-10 bps -> totale LOW~3, MID~8, HIGH~15
#   SnrFin  : 25 banche senior IG, piu' concentrato del Main
#             bid-offer index ~0.5-2 bps (Bazzana et al. 2023), single name
#             financials IG leggermente piu' larghi -> totale LOW~4, MID~10, HIGH~18
#   SubFin  : stessi 25 nomi ma debito subordinato; investitori risk-averse con
#             leverage effects (Katsourides 2022, Business Perspectives);
#             bid-offer molto piu' dinamico -> totale LOW~6, MID~14, HIGH~25
#   Xover   : 75 nomi sub-IG; CDX HY ~3 bps half-spread (Collin-Dufresne et al.
#             2020), single name HY ~15-30 bps -> totale LOW~8, MID~18, HIGH~35

# --- Fee Main (modifica questi valori) ---
FEE_MAIN_LOW_BPS  = 0.5
FEE_MAIN_MID_BPS  = 0.75
FEE_MAIN_HIGH_BPS = 1.0

# --- Fee SnrFin (modifica questi valori) ---
FEE_SNRFIN_LOW_BPS  = 0.5
FEE_SNRFIN_MID_BPS  = 0.75
FEE_SNRFIN_HIGH_BPS = 1

# --- Fee SubFin (modifica questi valori) ---
FEE_SUBFIN_LOW_BPS  = 1.0
FEE_SUBFIN_MID_BPS  = 2.0
FEE_SUBFIN_HIGH_BPS = 3.0

# --- Fee Xover (modifica questi valori) ---
FEE_XOVER_LOW_BPS  = 3.0
FEE_XOVER_MID_BPS  = 5.0
FEE_XOVER_HIGH_BPS = 7.0

# --- Mapping indice -> (LOW, MID, HIGH) ---
FEE_BY_INDEX = {
    'Main':   (FEE_MAIN_LOW_BPS,   FEE_MAIN_MID_BPS,   FEE_MAIN_HIGH_BPS),
    'SnrFin': (FEE_SNRFIN_LOW_BPS, FEE_SNRFIN_MID_BPS, FEE_SNRFIN_HIGH_BPS),
    'SubFin': (FEE_SUBFIN_LOW_BPS, FEE_SUBFIN_MID_BPS, FEE_SUBFIN_HIGH_BPS),
    'Xover':  (FEE_XOVER_LOW_BPS,  FEE_XOVER_MID_BPS,  FEE_XOVER_HIGH_BPS),
}

# --- Soglie iTraxx Main in bps (modifica questi valori) ---
# Suggerimento: calibrare sui terzili storici della serie iTraxx Main
ITRAXX_MAIN_LOW_THRESHOLD  = 60.0   # sotto questa -> fee LOW
ITRAXX_MAIN_HIGH_THRESHOLD = 100.0  # sopra questa -> fee HIGH
                                     # tra le due   -> fee MID

# --- Path file iTraxx Main ---
ITRAXX_MAIN_FILE = Path(r"C:\Users\aless\Desktop\THESIS\data\external\Factors\Tradable_corporate_bond_factors.xlsx")


# ============================================================================
# PARAMETRI REBONATO - NUOVI
# ============================================================================

# === THETA ESTIMATION (Rebonato Section 3.2.2, Eq. 15-16) ===
THETA_WINDOW_TYPE = "expanding"  # "expanding" (Eq. 15) o "rolling" (Eq. 16)
                                 # Rebonato riporta risultati principali con expanding
                                 # (scelta "more conservative"), e testa rolling come
                                 # sensitivity analysis (Table 4)

THETA_ROLLING_WINDOW = 252       # giorni - usato solo se THETA_WINDOW_TYPE = "rolling"

THETA_MIN_MONTHS = 6             # mesi - periodo minimo di dati prima di iniziare
                                 # la pesatura signal-weighted. Prima di questo,
                                 # i trade pesano tutti 1 (equal-weight warm-up).

# === SIGNAL SCALING ===
NORMALIZE_SIGNAL = True          # True: weight = signal / mean(signal_history)
                                 # Porta il peso medio a ~1 (nozionale "normale")
                                 # False: weight = signal grezzo (|basis - theta|)

SIGNAL_FLOOR = 0.0               # 0 = disattivato. Peso minimo dopo normalizzazione.
                                 # Es. 0.25 = ogni trade pesa almeno 25% del medio
SIGNAL_CAP = 0.0                 # 0 = disattivato. Peso massimo dopo normalizzazione.
                                 # Es. 4.0 = nessun trade pesa più di 4x il medio

# ============================================================================
# PATHS - Output separato da 02b
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "itraxx_combined"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: CARICA DATI PER TUTTI GLI INDICI
# ============================================================================

print("=" * 80)
print("STEP 1: Caricamento dati processati - Indici Selezionati")
print("       [REBONATO SIGNAL-WEIGHTED VERSION]")
print("=" * 80)

print(f"\n📊 Indici selezionati: {', '.join(INDICES_TO_COMBINE)}")

valid_indices = ['Main', 'SnrFin', 'SubFin', 'Xover']
for idx in INDICES_TO_COMBINE:
    if idx not in valid_indices:
        raise ValueError(f"Indice '{idx}' non valido. Usa: {valid_indices}")

indices_data = {}

for index_name in INDICES_TO_COMBINE:
    data_dir = PROJECT_ROOT / "data" / "processed" / f"itraxx_{index_name.lower()}"
    
    print(f"\n📂 Caricamento {index_name}...")
    
    basis_df = pd.read_parquet(data_dir / "itraxx_basis_wide.parquet")
    duration_df = pd.read_parquet(data_dir / "itraxx_duration_wide.parquet")
    maturity_df = pd.read_csv(data_dir / "itraxx_maturity_dates.csv", parse_dates=['Maturity'])
    
    basis_df = basis_df[~basis_df.index.isna()]
    duration_df = duration_df[~duration_df.index.isna()]
    
    series_list = [col.replace('_Basis', '') for col in basis_df.columns]
    maturity_dates = dict(zip(maturity_df['Serie'], maturity_df['Maturity']))
    
    indices_data[index_name] = {
        'basis': basis_df,
        'duration': duration_df,
        'maturity': maturity_dates,
        'series_list': series_list
    }
    
    print(f"   ✅ {index_name}: {basis_df.shape[0]} giorni, {len(series_list)} serie")

# Trova date comuni
all_dates = indices_data[INDICES_TO_COMBINE[0]]['basis'].index
for index_name in INDICES_TO_COMBINE[1:]:
    all_dates = all_dates.intersection(indices_data[index_name]['basis'].index)

all_dates = all_dates.sort_values()

print(f"\n✅ Date comuni: {len(all_dates)} giorni")
print(f"📅 Date range: {all_dates.min()} to {all_dates.max()}")

# --- Carica iTraxx Main per fee dinamiche ---
print(f"\n📂 Caricamento iTraxx Main per fee dinamiche...")
_itrx_raw = pd.read_excel(
    ITRAXX_MAIN_FILE,
    sheet_name="CDS_INDEX",
    skiprows=14,
    usecols=[0, 1],
    header=0
)
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

def get_fee_bps(date, index_name):
    """
    Restituisce il livello di fee in bps in base al livello di iTraxx Main
    alla data specificata e all'indice specifico (Main/SnrFin/SubFin/Xover).
    Usa forward-fill se la data esatta non e' disponibile.
    """
    available = itrx_main_daily[itrx_main_daily.index <= date]
    if len(available) == 0:
        regime = 1  # fallback: MID
    else:
        level = available.iloc[-1]
        if level < ITRAXX_MAIN_LOW_THRESHOLD:
            regime = 0  # LOW
        elif level >= ITRAXX_MAIN_HIGH_THRESHOLD:
            regime = 2  # HIGH
        else:
            regime = 1  # MID

    fees = FEE_BY_INDEX.get(index_name, FEE_BY_INDEX['Main'])
    return fees[regime]

# ============================================================================
# STEP 2: MATURITY DATES
# ============================================================================

print("\n" + "=" * 80)
print(f"STEP 2: Maturity dates ({INDICES_TO_COMBINE[0]} come riferimento)")
print("=" * 80)

maturity_dates_main = indices_data[INDICES_TO_COMBINE[0]]['maturity']

print(f"✅ Caricate {len(maturity_dates_main)} maturity dates per {INDICES_TO_COMBINE[0]}")
print(f"\n   Prime 5 Serie:")
for i, (serie, mat_date) in enumerate(sorted(maturity_dates_main.items(), 
                                               key=lambda x: int(x[0].replace('Ser', '')), 
                                               reverse=True)[:5]):
    print(f"      {serie} → {mat_date.strftime('%Y-%m-%d')}")

def months_to_maturity(date, serie, index_name):
    """Calcola quanti mesi mancano alla scadenza della Serie per l'indice specificato"""
    maturity_dates = indices_data[index_name]['maturity']
    if serie not in maturity_dates:
        return None
    mat_date = maturity_dates[serie]
    if date > mat_date:
        return 0
    months = (mat_date.year - date.year) * 12 + (mat_date.month - date.month)
    return months

# ============================================================================
# STEP 3: GENERA DATE DI CHECK ENTRY + DEFINIZIONE ON-THE-RUN + CALCOLO THETA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Generazione date di controllo + on-the-run + calcolo theta")
print("=" * 80)

if all_dates.isna().any():
    print("⚠️  ATTENZIONE: Trovati NaT nell'indice, rimozione in corso...")
    all_dates = all_dates[~all_dates.isna()]

if ENTRY_CHECK_FREQ == "daily":
    check_dates = all_dates
elif ENTRY_CHECK_FREQ == "weekly":
    check_dates = all_dates[all_dates.weekday == 4]
elif ENTRY_CHECK_FREQ == "monthly":
    check_dates = all_dates.to_series().groupby([all_dates.year, all_dates.month]).last()
else:
    raise ValueError(f"ENTRY_CHECK_FREQ non valida: {ENTRY_CHECK_FREQ}")

print(f"✅ Frequenza entry check: {ENTRY_CHECK_FREQ}")
print(f"✅ Date di check: {len(check_dates)} su {len(all_dates)} giorni totali")

# --- DEFINIZIONE ON-THE-RUN (serve per calcolo theta) ---

def get_on_the_run_series(date, index_name, max_rank=1):
    """Trova le serie on-the-run per un indice specifico."""
    basis_df = indices_data[index_name]['basis']
    series_list = indices_data[index_name]['series_list']
    
    valid_series = []
    for serie in series_list:
        col_basis = f"{serie}_Basis"
        if date not in basis_df.index:
            continue
        basis = basis_df.loc[date, col_basis]
        if pd.notna(basis):
            valid_series.append(serie)
    
    valid_series.sort(key=lambda s: int(s.replace('Ser', '')), reverse=True)
    return valid_series[:max_rank]

# --- CALCOLO THETA PER-INDICE GIORNALIERO (Rebonato Eq. 15-16) ---
# Ogni indice (Main, SnrFin, SubFin, Xover) ha il proprio theta_t, calcolato
# come expanding (o rolling) mean della basis delle sole serie tradabili
# (on-the-run, secondo MAX_SERIES_RANK) per quell'indice.

print(f"\n📐 Calcolo theta per-indice ({THETA_WINDOW_TYPE} window, "
      f"serie tradabili con MAX_SERIES_RANK={MAX_SERIES_RANK})...")

# Per ogni indice, costruisci la serie giornaliera della basis tradabile
# e poi calcola theta con expanding/rolling
theta_index = {}        # {index_name: pd.Series theta giornaliero}
index_first_date = {}   # {index_name: prima data con dati tradabili}

for index_name in INDICES_TO_COMBINE:
    basis_df_idx = indices_data[index_name]['basis']
    
    # Per ogni giorno, prendi la media della basis delle serie on-the-run
    basis_tradable = pd.Series(index=all_dates, dtype=float)
    
    for date in all_dates:
        otr_series = get_on_the_run_series(date, index_name, max_rank=MAX_SERIES_RANK)
        if len(otr_series) == 0:
            basis_tradable.loc[date] = np.nan
            continue
        
        basis_values = []
        for serie in otr_series:
            col_basis = f"{serie}_Basis"
            if date in basis_df_idx.index:
                val = basis_df_idx.loc[date, col_basis]
                if pd.notna(val):
                    basis_values.append(val)
        
        if len(basis_values) > 0:
            basis_tradable.loc[date] = np.mean(basis_values)
        else:
            basis_tradable.loc[date] = np.nan
    
    # Calcola theta: expanding o rolling
    if THETA_WINDOW_TYPE == "expanding":
        theta_series = basis_tradable.expanding(min_periods=1).mean()
    elif THETA_WINDOW_TYPE == "rolling":
        theta_series = basis_tradable.rolling(window=THETA_ROLLING_WINDOW, min_periods=1).mean()
    else:
        raise ValueError(f"THETA_WINDOW_TYPE non valido: {THETA_WINDOW_TYPE}")
    
    theta_index[index_name] = theta_series
    
    # Prima data con dati tradabili per questo indice
    first_valid = basis_tradable.dropna().index.min()
    index_first_date[index_name] = first_valid

# Calcola theta_start per-indice (warm-up)
index_theta_start = {
    idx_name: first_dt + pd.DateOffset(months=THETA_MIN_MONTHS)
    for idx_name, first_dt in index_first_date.items()
    if pd.notna(first_dt)
}

for index_name in INDICES_TO_COMBINE:
    theta_s = theta_index[index_name]
    last_val = theta_s.dropna().iloc[-1] if theta_s.dropna().shape[0] > 0 else np.nan
    start_str = index_first_date.get(index_name, pd.NaT)
    mature_str = index_theta_start.get(index_name, pd.NaT)
    print(f"   {index_name}: theta ultimo={last_val:.2f} bps, "
          f"dati da {start_str.strftime('%Y-%m-%d') if pd.notna(start_str) else 'N/A'}, "
          f"maturo da {mature_str.strftime('%Y-%m-%d') if pd.notna(mature_str) else 'N/A'}")

# ============================================================================
# STEP 4: (on-the-run logic definita in Step 3 — qui solo test)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Test logica on-the-run")
print("=" * 80)

test_date = all_dates[len(all_dates)//2]
print(f"✅ Test on-the-run logic:")
print(f"   Data test: {test_date.strftime('%Y-%m-%d')}")
for index_name in INDICES_TO_COMBINE:
    test_otr = get_on_the_run_series(test_date, index_name, max_rank=MAX_SERIES_RANK)
    print(f"   {index_name}: {test_otr}")

# ============================================================================
# STEP 5: SIMULAZIONE TRADING (IDENTICA A 02b, con aggiunta signal)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Simulazione trading combinata")
print("=" * 80)

trades_log = []
trade_id_counter = 1
first_trade_opened = False
open_trades = {}

print("🔄 Inizio simulazione...")
print()

for i, date in enumerate(all_dates):
    
    if i % 500 == 0:
        print(f"   Processing: {date.strftime('%Y-%m-%d')} ({i}/{len(all_dates)})")

    # === STEP 5.1: AGGIORNA P&L E CHECK EXIT PER TRADE APERTI ===

    trades_to_close = []

    for trade_id, trade in open_trades.items():
        index_name = trade['index']
        serie = trade['serie']
        params = INDEX_PARAMS[index_name]
        
        basis_df = indices_data[index_name]['basis']
        duration_df = indices_data[index_name]['duration']
        maturity_dates = indices_data[index_name]['maturity']
        
        col_basis = f"{serie}_Basis"
        col_duration = f"{serie}_Duration"
        
        if date not in basis_df.index:
            continue
        
        current_basis = basis_df.loc[date, col_basis]
        current_duration = duration_df.loc[date, col_duration]
        
        if pd.isna(current_basis):
            mat_date = maturity_dates.get(serie)
            if mat_date is not None and date >= mat_date:
                trade['exit_date'] = date
                trade['exit_basis'] = 0
                trade['exit_duration'] = 0.0   # indice scaduto
                trade['exit_reason'] = 'MATURITY'
                trades_to_close.append(trade_id)
                continue
            else:
                continue
        
        prev_basis = trade.get('prev_basis', trade['entry_basis'])
        days_elapsed = (date - trade['prev_date']).days if 'prev_date' in trade else 1
        
        if pd.notna(prev_basis) and pd.notna(current_basis) and pd.notna(current_duration):
            sign = 1 if trade['direction'] == 'LONG' else -1
            capital_gain_bps = sign * (prev_basis - current_basis) * current_duration
            carry_bps = sign * prev_basis * (days_elapsed / 365.0)
            daily_pnl_bps = capital_gain_bps + carry_bps
            daily_pnl_pct = daily_pnl_bps / 100.0
            
            trade['cumulative_pnl'] = trade.get('cumulative_pnl', 0) + daily_pnl_pct
            trade['cumulative_capital_gain'] = trade.get('cumulative_capital_gain', 0) + (capital_gain_bps / 100.0)
            trade['cumulative_carry'] = trade.get('cumulative_carry', 0) + (carry_bps / 100.0)
        
        trade['prev_basis'] = current_basis
        trade['prev_date'] = date
        if pd.notna(current_duration):
            trade['prev_duration'] = current_duration
        
        # Check exit
        months_left = months_to_maturity(date, serie, index_name)
        
        if months_left is not None and (months_left == 0 or date >= maturity_dates[serie]):
            trade['exit_date'] = date
            trade['exit_basis'] = 0
            trade['exit_duration'] = 0.0   # indice scaduto
            trade['exit_reason'] = 'MATURITY'
            trades_to_close.append(trade_id)
            continue
        
        can_close = len(open_trades) > MIN_OPEN_TRADES or not first_trade_opened
        
        if can_close:
            if trade['direction'] == 'LONG' and current_basis < params['exit_long']:
                trade['exit_date'] = date
                trade['exit_basis'] = current_basis
                trade['exit_duration'] = current_duration if pd.notna(current_duration) else trade.get('prev_duration', trade['entry_duration'])
                trade['exit_reason'] = 'TARGET_HIT'
                trades_to_close.append(trade_id)
            elif trade['direction'] == 'SHORT' and current_basis > params['exit_short']:
                trade['exit_date'] = date
                trade['exit_basis'] = current_basis
                trade['exit_duration'] = current_duration if pd.notna(current_duration) else trade.get('prev_duration', trade['entry_duration'])
                trade['exit_reason'] = 'TARGET_HIT'
                trades_to_close.append(trade_id)

    for trade_id in trades_to_close:
        trade = open_trades.pop(trade_id)
        trades_log.append(trade.copy())

    # === STEP 5.2: CHECK NUOVI ENTRY - TUTTI GLI INDICI ===
    
    if date in check_dates:
        all_candidates = []
        
        for index_name, params in INDEX_PARAMS.items():
            if index_name not in INDICES_TO_COMBINE:
                continue
                
            basis_df = indices_data[index_name]['basis']
            duration_df = indices_data[index_name]['duration']
            
            if date not in basis_df.index:
                continue
            
            eligible_series = get_on_the_run_series(date, index_name, max_rank=MAX_SERIES_RANK)
            
            for serie in eligible_series:
                col_basis = f"{serie}_Basis"
                col_duration = f"{serie}_Duration"
                
                basis = basis_df.loc[date, col_basis]
                duration = duration_df.loc[date, col_duration]
                
                if pd.isna(basis) or pd.isna(duration):
                    continue
                if basis > params['max_basis'] or basis < params['min_basis']:
                    continue
                
                # Check multiple entries
                if not ALLOW_MULTIPLE_ENTRIES:
                    already_open = any(t['index'] == index_name and t['serie'] == serie 
                                      for t in open_trades.values())
                    if already_open:
                        continue
                else:
                    if REENTRY_BASIS_WIDENING > 0:
                        last_entry_basis = None
                        for t in open_trades.values():
                            if t['index'] == index_name and t['serie'] == serie:
                                if last_entry_basis is None:
                                    last_entry_basis = t['entry_basis']
                                else:
                                    if abs(t['entry_basis']) > abs(last_entry_basis):
                                        last_entry_basis = t['entry_basis']
                        
                        if last_entry_basis is not None:
                            if last_entry_basis > 0:
                                if basis > 0 and basis < last_entry_basis + REENTRY_BASIS_WIDENING:
                                    continue
                            else:
                                if basis < 0 and basis > last_entry_basis - REENTRY_BASIS_WIDENING:
                                    continue
               
                # Cap max trade contemporanei su stesso indice+serie
                if MAX_CONCURRENT_PER_INDEX_SERIE is not None:
                    n_open_same = sum(1 for t in open_trades.values() 
                                      if t['index'] == index_name and t['serie'] == serie)
                    if n_open_same >= MAX_CONCURRENT_PER_INDEX_SERIE:
                        continue
                

                # Check soglie entry
                direction = None
                if basis > params['entry_long']:
                    direction = 'LONG'
                elif basis < params['entry_short']:
                    direction = 'SHORT'
                
                if direction:
                    # --- REBONATO: calcola il segnale all'entry (theta per-indice) ---
                    theta_t = np.nan
                    if index_name in theta_index and date in theta_index[index_name].index:
                        theta_t = theta_index[index_name].loc[date]
                    
                    if pd.notna(theta_t):
                        signal = abs(basis - theta_t)
                    else:
                        signal = 1.0
                    
                    # Warm-up per-indice: se questo indice non ha ancora
                    # THETA_MIN_MONTHS di storia sulle serie tradabili
                    if index_name not in index_theta_start or date < index_theta_start[index_name]:
                        signal = 1.0
                    
                    all_candidates.append({
                        'index': index_name,
                        'serie': serie,
                        'basis': basis,
                        'duration': duration,
                        'direction': direction,
                        'abs_basis': abs(basis),
                        'signal': signal,
                        'theta_t': theta_t if pd.notna(theta_t) else np.nan
                    })
        
        # Scegli il migliore
        if all_candidates:
            all_candidates.sort(key=lambda x: x['abs_basis'], reverse=True)
            best = all_candidates[0]
            
            new_trade = {
                'trade_id': trade_id_counter,
                'index': best['index'],
                'serie': best['serie'],
                'entry_date': date,
                'entry_basis': best['basis'],
                'entry_duration': best['duration'],
                'direction': best['direction'],
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
            trade_id_counter += 1
            first_trade_opened = True

# === CHIUDI TRADE APERTI ALLA FINE ===
last_date = all_dates[-1]

print(f"\n🔍 DEBUG: Trade ancora aperti alla fine: {len(open_trades)}")

for trade_id in list(open_trades.keys()):
    trade = open_trades[trade_id]
    index_name = trade['index']
    serie = trade['serie']
    
    basis_df = indices_data[index_name]['basis']
    maturity_dates = indices_data[index_name]['maturity']
    
    col_basis = f"{serie}_Basis"
    
    mat_date = maturity_dates.get(serie)
    if mat_date is not None and last_date >= mat_date:
        trade['exit_date'] = last_date
        trade['exit_basis'] = 0
        trade['exit_duration'] = 0.0   # indice scaduto
        trade['exit_reason'] = 'MATURITY'
    else:
        trade['exit_date'] = last_date
        if last_date in basis_df.index:
            exit_basis = basis_df.loc[last_date, col_basis]
            if pd.isna(exit_basis):
                exit_basis = trade.get('prev_basis', 0)
        else:
            exit_basis = trade.get('prev_basis', 0)
        trade['exit_basis'] = exit_basis
        if last_date in duration_df.index:
            exit_duration = duration_df.loc[last_date, col_duration]
            if pd.isna(exit_duration):
                exit_duration = trade.get('prev_duration', trade['entry_duration'])
        else:
            exit_duration = trade.get('prev_duration', trade['entry_duration'])
        trade['exit_duration'] = exit_duration
        trade['exit_reason'] = 'END_OF_SAMPLE'
    
    open_trades.pop(trade_id)
    trades_log.append(trade.copy())

print(f"\n✅ Simulazione completata!")
print(f"📊 Trade totali eseguiti: {len(trades_log)}")

# ============================================================================
# STEP 6: CREA DATAFRAME TRADES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Creazione log dei trade")
print("=" * 80)

if len(trades_log) == 0:
    print("❌ ERRORE: Nessun trade eseguito!")
    exit()

trades_df = pd.DataFrame(trades_log)
trades_df['duration_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
trades_df['duration_trading_days'] = trades_df.apply(
    lambda r: ((all_dates >= r['entry_date']) & (all_dates <= r['exit_date'])).sum(), axis=1)
trades_df['pnl_bps'] = trades_df['cumulative_pnl']

# Assicura che exit_duration esista (fallback a entry_duration se mancante)
if 'exit_duration' not in trades_df.columns:
    trades_df['exit_duration'] = trades_df['entry_duration']
trades_df['exit_duration'] = trades_df['exit_duration'].fillna(trades_df['entry_duration'])

print(f"✅ Trade log creato: {len(trades_df)} trade")
print(f"\n📈 Statistiche Trade:")
print(f"   Long:  {len(trades_df[trades_df['direction']=='LONG'])}")
print(f"   Short: {len(trades_df[trades_df['direction']=='SHORT'])}")
print(f"   Durata media: {trades_df['duration_days'].mean():.0f} giorni")
print(f"   P&L medio: {trades_df['pnl_bps'].mean():.2f}%")

print(f"\n📊 Trade per Indice:")
for index_name in INDICES_TO_COMBINE:
    count = len(trades_df[trades_df['index'] == index_name])
    pct = (count / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    print(f"   {index_name}: {count} trade ({pct:.1f}%)")

# --- REBONATO: Stampa statistiche segnale ---
print(f"\n📐 REBONATO SIGNAL STATISTICS (theta per-indice):")
print(f"   Signal medio: {trades_df['signal'].mean():.2f}")
print(f"   Signal mediano: {trades_df['signal'].median():.2f}")
print(f"   Signal min: {trades_df['signal'].min():.2f}")
print(f"   Signal max: {trades_df['signal'].max():.2f}")
print(f"   Theta medio all'entry: {trades_df['theta_at_entry'].mean():.2f} bps")
warm_up_trades = trades_df[trades_df['signal'] == 1.0]
signal_trades = trades_df[trades_df['signal'] != 1.0]
print(f"   Trade in warm-up (peso=1): {len(warm_up_trades)}")
print(f"   Trade signal-weighted: {len(signal_trades)}")
print(f"\n   Signal per indice:")
for index_name in INDICES_TO_COMBINE:
    idx_trades = trades_df[trades_df['index'] == index_name]
    if len(idx_trades) > 0:
        print(f"      {index_name}: signal medio={idx_trades['signal'].mean():.2f}, "
              f"theta medio entry={idx_trades['theta_at_entry'].mean():.2f} bps")

# Salva trades log
trades_path = RESULTS_DIR / "trades_log.csv"
trades_df.to_csv(trades_path, index=False)
print(f"\n💾 Salvato: {trades_path.name}")

# ============================================================================
# STEP 7: COSTRUZIONE INDICE - SIGNAL-WEIGHTED (REBONATO)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Costruzione indice dei return [REBONATO SIGNAL-WEIGHTED]")
print("=" * 80)

# --- Costruisci pesi con normalizzazione e clip opzionali ---

# --- Ex-post filter: rimuovi trade troppo brevi (probabile rumore dati) ---
if MIN_TRADE_DURATION_DAYS is not None:
    n_before = len(trades_df)
    trades_df_index = trades_df[trades_df['duration_trading_days'] >= MIN_TRADE_DURATION_DAYS].copy()
    n_removed = n_before - len(trades_df_index)
    print(f"   ⚠️ Filtro MIN_TRADE_DURATION_DAYS={MIN_TRADE_DURATION_DAYS} (trading days): "
          f"rimossi {n_removed}/{n_before} trade (durata < {MIN_TRADE_DURATION_DAYS} trading days)")
else:
    trades_df_index = trades_df.copy()

trades_sorted = trades_df_index.sort_values('entry_date').copy()

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

# Ricostruisci P&L giornaliero per ogni trade (identico a 02b)
daily_returns = pd.DataFrame(index=all_dates)

for _, trade in trades_df_index.iterrows():
    trade_id = trade['trade_id']
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    index_name = trade['index']
    serie = trade['serie']
    
    basis_df = indices_data[index_name]['basis']
    duration_df = indices_data[index_name]['duration']
    
    trade_dates = all_dates[(all_dates >= entry_date) & (all_dates <= exit_date)]
    if len(trade_dates) == 0:
        continue
    
    col_basis = f"{serie}_Basis"
    col_duration = f"{serie}_Duration"

    # -------------------------------------------------------------------------
    # CALCOLO FEE TOTALE DEL TRADE
    # Fee entrata = entry_duration * fee_bps(entry_date) / 100  (sempre)
    # Fee uscita  = exit_duration  * fee_bps(exit_date)  / 100  (solo se non MATURITY)
    # La fee totale viene spalmata SOLO sui giorni con return valido (non NaN),
    # cosi' i giorni con basis mancante restano NaN e non distorcono la media EW,
    # ma la fee totale pagata e' comunque esattamente quella attesa.
    # -------------------------------------------------------------------------
    fee_bps_entry = get_fee_bps(entry_date, index_name)
    fee_entry_pct = trade['entry_duration'] * fee_bps_entry / 100.0
    if trade['exit_reason'] != 'MATURITY':
        fee_bps_exit = get_fee_bps(exit_date, index_name)
        fee_exit_pct = trade['exit_duration'] * fee_bps_exit / 100.0
    else:
        fee_bps_exit = 0.0
        fee_exit_pct = 0.0
    fee_total_pct = fee_entry_pct + fee_exit_pct

    trade_returns = []
    prev_basis = trade['entry_basis']
    prev_date = entry_date

    # --- Passata 1: calcola return grezzi (senza fee), NaN dove mancano dati ---
    for date in trade_dates[1:]:
        if date not in basis_df.index:
            trade_returns.append(np.nan)
            continue

        current_basis = basis_df.loc[date, col_basis]
        current_duration = duration_df.loc[date, col_duration]

        if pd.isna(current_basis):
            mat_date = indices_data[index_name]['maturity'].get(serie)
            if mat_date is not None and date >= mat_date:
                current_basis = 0
            else:
                trade_returns.append(np.nan)
                continue

        days_elapsed = (date - prev_date).days

        if pd.notna(prev_basis) and pd.notna(current_basis) and pd.notna(current_duration):
            sign = 1 if trade['direction'] == 'LONG' else -1
            capital_gain_bps = sign * (prev_basis - current_basis) * current_duration
            carry_bps = sign * prev_basis * (days_elapsed / 365.0)
            daily_ret_bps = capital_gain_bps + carry_bps
            daily_ret_pct = daily_ret_bps / 100.0
        else:
            daily_ret_pct = np.nan

        trade_returns.append(daily_ret_pct)
        prev_basis = current_basis
        prev_date = date

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

# Matrice pesi
weights_matrix = pd.DataFrame(index=all_dates, columns=trade_cols, dtype=float)

for col in trade_cols:
    weight = trade_weights[col]
    active_mask = daily_returns[col].notna()
    weights_matrix.loc[active_mask, col] = weight

# Return pesato
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

valid_sw = daily_returns['index_return_sw'].dropna()
valid_ew = daily_returns['index_return_ew'].dropna()
valid_sel = daily_returns['index_return'].dropna()

print(f"✅ Indice costruito: {len(daily_returns)} giorni")
print(f"📅 Dal {daily_returns.index.min()} al {daily_returns.index.max()}")
print(f"\n✅ Primo trade entry: {first_trade_entry.strftime('%Y-%m-%d')}")
print(f"✅ Primo return index: {first_return_date.strftime('%Y-%m-%d')}")

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

# === SELECTED ===
daily_returns['index_value'] = 100 * (1 + daily_returns['index_return'] / 100).cumprod()
daily_returns['cumulative_return'] = (daily_returns['index_value'] / 100 - 1) * 100
# === SW ===
daily_returns['index_value_sw'] = 100 * (1 + daily_returns['index_return_sw'] / 100).cumprod()
daily_returns['cumulative_return_sw'] = (daily_returns['index_value_sw'] / 100 - 1) * 100
# === EW ===
daily_returns['index_value_ew'] = 100 * (1 + daily_returns['index_return_ew'] / 100).cumprod()
daily_returns['cumulative_return_ew'] = (daily_returns['index_value_ew'] / 100 - 1) * 100

print(f"📌 SELECTED MODE: {OUTPUT_RETURN_MODE.upper()}")
print(f"📈 Return totale (SELECTED): {daily_returns['cumulative_return'].iloc[-1]:.2f}%")
print(f"📈 Return totale (SW):       {daily_returns['cumulative_return_sw'].iloc[-1]:.2f}%")
print(f"📈 Return totale (EW):       {daily_returns['cumulative_return_ew'].iloc[-1]:.2f}%")
print(f"📈 Index finale (SELECTED):  {daily_returns['index_value'].iloc[-1]:.2f}")
print(f"📈 Index finale (SW):        {daily_returns['index_value_sw'].iloc[-1]:.2f}")
print(f"📈 Index finale (EW):        {daily_returns['index_value_ew'].iloc[-1]:.2f}")


# ============================================================================
# STEP 9: RESAMPLE A FREQUENZA DESIDERATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Resample a frequenza indice")
print("=" * 80)

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

print(f"✅ Indice resample a: {INDEX_FREQ}")
print(f"📊 Osservazioni: {len(index_final)}")

# ============================================================================
# STEP 10: SALVA RISULTATI
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Salvataggio risultati")
print("=" * 80)

index_path = RESULTS_DIR / f"index_{INDEX_FREQ}.csv"
index_final.to_csv(index_path)
print(f"💾 Salvato: {index_path.name}")

daily_path = RESULTS_DIR / "daily_returns_full.csv"
daily_returns.to_csv(daily_path)
print(f"💾 Salvato: {daily_path.name}")

# Salva parametri per tracciabilità
params_dict = {
    'FEE_MAIN_LOW_BPS':   FEE_MAIN_LOW_BPS,
    'FEE_MAIN_MID_BPS':   FEE_MAIN_MID_BPS,
    'FEE_MAIN_HIGH_BPS':  FEE_MAIN_HIGH_BPS,
    'FEE_SNRFIN_LOW_BPS': FEE_SNRFIN_LOW_BPS,
    'FEE_SNRFIN_MID_BPS': FEE_SNRFIN_MID_BPS,
    'FEE_SNRFIN_HIGH_BPS':FEE_SNRFIN_HIGH_BPS,
    'FEE_SUBFIN_LOW_BPS': FEE_SUBFIN_LOW_BPS,
    'FEE_SUBFIN_MID_BPS': FEE_SUBFIN_MID_BPS,
    'FEE_SUBFIN_HIGH_BPS':FEE_SUBFIN_HIGH_BPS,
    'FEE_XOVER_LOW_BPS':  FEE_XOVER_LOW_BPS,
    'FEE_XOVER_MID_BPS':  FEE_XOVER_MID_BPS,
    'FEE_XOVER_HIGH_BPS': FEE_XOVER_HIGH_BPS,
    'ITRAXX_MAIN_LOW_THRESHOLD': ITRAXX_MAIN_LOW_THRESHOLD,
    'ITRAXX_MAIN_HIGH_THRESHOLD': ITRAXX_MAIN_HIGH_THRESHOLD,
    'INDICES_TO_COMBINE': '|'.join(INDICES_TO_COMBINE),
    'THETA_WINDOW_TYPE': THETA_WINDOW_TYPE,
    'THETA_ROLLING_WINDOW': THETA_ROLLING_WINDOW,
    'THETA_MIN_MONTHS': THETA_MIN_MONTHS,
    'NORMALIZE_SIGNAL': NORMALIZE_SIGNAL,
    'SIGNAL_FLOOR': SIGNAL_FLOOR,
    'SIGNAL_CAP': SIGNAL_CAP,
    'MAX_SERIES_RANK': MAX_SERIES_RANK,
    'MIN_OPEN_TRADES': MIN_OPEN_TRADES,
    'ALLOW_MULTIPLE_ENTRIES': ALLOW_MULTIPLE_ENTRIES,
    'REENTRY_BASIS_WIDENING': REENTRY_BASIS_WIDENING,
    'ENTRY_CHECK_FREQ': ENTRY_CHECK_FREQ,
    'INDEX_FREQ': INDEX_FREQ,
    'OUTPUT_RETURN_MODE': OUTPUT_RETURN_MODE,

}
# Aggiungi parametri per indice
for idx_name, params in INDEX_PARAMS.items():
    if idx_name in INDICES_TO_COMBINE:
        for k, v in params.items():
            params_dict[f'{idx_name}_{k}'] = v

params_df = pd.DataFrame(list(params_dict.items()), columns=['Parameter', 'Value'])
params_path = RESULTS_DIR / "parameters.csv"
params_df.to_csv(params_path, index=False)
print(f"💾 Salvato: {params_path.name}")

# ============================================================================
# STEP 11: STATISTICHE FINALI + CONFRONTO
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICHE FINALI — CONFRONTO SIGNAL-WEIGHTED vs EQUAL-WEIGHT")
print("=" * 80)

print(f"\n📊 PARAMETRI USATI:")
print(f"   Indici: {', '.join(INDICES_TO_COMBINE)}")
print(f"   Max Series Rank: {MAX_SERIES_RANK}")
print(f"   Min open trades: {MIN_OPEN_TRADES}")
print(f"   Multiple entries: {'Yes' if ALLOW_MULTIPLE_ENTRIES else 'No'}")
if ALLOW_MULTIPLE_ENTRIES:
    print(f"   Reentry widening: {REENTRY_BASIS_WIDENING} bps")
print(f"\n💸 TRANSACTION FEES (dinamiche su iTraxx Main, per indice):")
print(f"   Soglie Main: LOW < {ITRAXX_MAIN_LOW_THRESHOLD} bps  |  HIGH >= {ITRAXX_MAIN_HIGH_THRESHOLD} bps")
print(f"   {'Indice':<10} {'LOW':>8} {'MID':>8} {'HIGH':>8}")
for idx_name, (fl, fm, fh) in FEE_BY_INDEX.items():
    marker = " ◄" if idx_name in INDICES_TO_COMBINE else ""
    print(f"   {idx_name:<10} {fl:>7.1f}b {fm:>7.1f}b {fh:>7.1f}b{marker}")
print(f"\n📐 REBONATO PARAMETERS:")
print(f"   Theta window: {THETA_WINDOW_TYPE}")
if THETA_WINDOW_TYPE == "rolling":
    print(f"   Rolling window: {THETA_ROLLING_WINDOW} days")
print(f"   Theta min months: {THETA_MIN_MONTHS}")
print(f"   Normalize signal: {'Yes' if NORMALIZE_SIGNAL else 'No'}")
if SIGNAL_FLOOR > 0 or SIGNAL_CAP > 0:
    print(f"   Signal floor: {SIGNAL_FLOOR}")
    print(f"   Signal cap: {SIGNAL_CAP}")

def compute_metrics(returns_series, freq_label="daily"):
    """Calcola Sharpe, return annualizzato, volatilità, max drawdown"""
    valid = returns_series.dropna()
    if len(valid) == 0:
        return {}
    
    if freq_label == "daily":
        ann_factor = 252
    elif freq_label == "weekly":
        ann_factor = 52
    elif freq_label == "monthly":
        ann_factor = 12
    else:
        ann_factor = 252
    
    avg_ret = valid.mean() * ann_factor
    vol = valid.std() * np.sqrt(ann_factor)
    sharpe = avg_ret / vol if vol > 0 else 0
    
    cum = valid.cumsum()
    running_max = cum.expanding().max()
    dd = cum - running_max
    max_dd = dd.min()
    
    skew = valid.skew()
    kurt = valid.kurtosis()
    
    return {
        'Avg Return (% p.a.)': avg_ret,
        'Volatility (% p.a.)': vol,
        'Sharpe (ann.)': sharpe,
        'Max Drawdown (%)': max_dd,
        'Skewness': skew,
        'Kurtosis': kurt,
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
print(f"   Trade totali: {len(trades_df)}")
print(f"   Trade Long:   {len(trades_df[trades_df['direction']=='LONG'])} ({len(trades_df[trades_df['direction']=='LONG'])/len(trades_df)*100:.1f}%)")
print(f"   Trade Short:  {len(trades_df[trades_df['direction']=='SHORT'])} ({len(trades_df[trades_df['direction']=='SHORT'])/len(trades_df)*100:.1f}%)")
print(f"   Durata media: {trades_df['duration_days'].mean():.0f} giorni")
print(f"   P&L medio per trade: {trades_df['pnl_bps'].mean():.2f}%")

print("\n" + "=" * 80)
print("✅ SIMULAZIONE REBONATO COMPLETATA CON SUCCESSO!")
print("=" * 80)

print(f"\n📁 File salvati in: {RESULTS_DIR}")
print(f"   • trades_log.csv")
print(f"   • index_{INDEX_FREQ}.csv (contiene sia signal-weighted che equal-weight)")
print(f"   • daily_returns_full.csv")
print(f"   • parameters.csv")

print("\n🎯 Prossimo step:")
print("   • Confronta con 02b (equal-weight originale)")
print("   • Calcola MPPM (04_mppm_analysis.py) su questo indice")
print("   • Calcola Moreira-Muir (05_moreira_muir.py) su questo indice")
print()