"""
Script 2b: Simulazione Trading Strategy BTP-Italia Basis — Signal-Weighted
==========================================================================
Versione con pesatura condizionale ispirata a Rebonato & Ronzani (2021),
"Is Convexity Efficiently Priced?", EDHEC Working Paper.

DIFFERENZA RISPETTO A 02a (equal-weight):
  In 02a, l'indice è la media equal-weighted dei return dei trade aperti.
  Qui, ogni trade è pesato da |basis_entry - theta_t|, dove theta_t è la
  media expanding (o rolling) della basis aggregata fino al giorno t.
  
  Il segnale misura quanto l'opportunità è "ricca" rispetto al livello 
  storico di equilibrio della basis, seguendo l'approccio di Rebonato per
  il curvature signal (Eq. 14 del paper).

COSA RIMANE IDENTICO A 02a:
  - Tutta la simulazione trading (soglie entry/exit, maturity rules, ecc.)
  - Il calcolo del P&L giornaliero per ogni trade
  - La logica di apertura/chiusura trade

COSA CAMBIA:
  - Step 6: l'indice usa pesi proporzionali al segnale invece di equal-weight
  - Nuovo parametro: THETA_WINDOW_TYPE e THETA_MIN_MONTHS
  - Output salvati in cartella separata (btp_italia_rebonato/)

Riferimento: Rebonato & Ronzani (2021), Section 3.2.2, Eq. 14-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI STRATEGIA - IDENTICI A 02a (MODIFICA QUESTI VALORI)
# ============================================================================

# === ENTRY THRESHOLDS ===
ENTRY_LONG_THRESHOLD = 40        # bps - long BTP-Italia se basis > questo
ENTRY_SHORT_THRESHOLD = -50      # bps - short BTP-Italia se basis < questo

# === EXIT THRESHOLDS ===
EXIT_LONG_THRESHOLD = 10         # bps - target per chiudere long
EXIT_SHORT_THRESHOLD = -10       # bps - target per chiudere short

# === OUTLIER FILTERS ===
MAX_BASIS_ENTRY = 300            # bps - ignora basis > questo (errore dato)
MIN_BASIS_ENTRY = -200           # bps - ignora basis < questo (errore dato)

# === MATURITY RULES ===
MIN_MONTHS_TO_MATURITY_ENTRY = 6 # mesi - non entrare se < 6 mesi a scadenza

# === HOLDING RULES ===
MIN_OPEN_TRADES = 1              # Dopo primo trade, mantieni sempre almeno N trade

# === MULTIPLE ENTRIES ===
ALLOW_MULTIPLE_ENTRIES = True    # True/False - permetti più trade sullo stesso ISIN
REENTRY_BASIS_WIDENING = 10      # bps - quanto deve allargarsi la basis per rientrare
MAX_CONCURRENT_PER_ISIN = None   # Max trade attivi sullo stesso ISIN (None = no cap)

# === DATA QUALITY FILTER ===
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
FEE_LOW_BPS  = 3.0    # bps - iTraxx Main sotto soglia bassa
FEE_MID_BPS  = 5.0    # bps - iTraxx Main tra le due soglie
FEE_HIGH_BPS = 8.0   # bps - iTraxx Main sopra soglia alta

# --- Soglie iTraxx Main in bps (modifica questi valori) ---
# Suggerimento: calibrare sui terzili storici della serie iTraxx Main
# es. quantile(0.33) e quantile(0.67) della serie giornaliera
ITRAXX_MAIN_LOW_THRESHOLD  = 60.0   # sotto questa -> FEE_LOW_BPS
ITRAXX_MAIN_HIGH_THRESHOLD = 100.0  # sopra questa -> FEE_HIGH_BPS
                                     # tra le due   -> FEE_MID_BPS

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
                                 # Rebonato testa W = 20, 60, 80 business days per la
                                 # curvatura. 252 ≈ 1 anno, ragionevole per la basis.

THETA_MIN_MONTHS = 6             # mesi - periodo minimo di dati prima di iniziare
                                 # la pesatura signal-weighted. Prima di questo,
                                 # i trade pesano tutti 1 (equal-weight warm-up).
                                 # Rebonato parte dal giorno 1 ma stima la volatilità
                                 # con warm-up di 100 gg (Section 3.2.1).

# === SIGNAL SCALING ===
NORMALIZE_SIGNAL = True          # True: weight = signal / mean(signal_history)
                                 # Porta il peso medio a ~1 (nozionale "normale")
                                 # False: weight = signal grezzo (|basis - theta|)

SIGNAL_FLOOR = 0.0               # 0 = disattivato. Peso minimo dopo normalizzazione.
                                 # Es. 0.25 = ogni trade pesa almeno 25% del medio
SIGNAL_CAP = 0.0                 # 0 = disattivato. Peso massimo dopo normalizzazione.
                                 # Es. 4.0 = nessun trade pesa più di 4x il medio

# ============================================================================
# PATHS - Output separato da 02a
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "btp_italia"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: CARICA DATI
# ============================================================================

print("=" * 80)
print("STEP 1: Caricamento dati processati")
print("       [REBONATO SIGNAL-WEIGHTED VERSION]")
print("=" * 80)

basis_df = pd.read_parquet(PROCESSED_DATA_DIR / "basis_wide.parquet")
dv01_df = pd.read_parquet(PROCESSED_DATA_DIR / "dv01_wide.parquet")

# Rimuovi eventuali NaT dall'indice
basis_df = basis_df[~basis_df.index.isna()]
dv01_df = dv01_df[~dv01_df.index.isna()]

print(f"✅ Basis loaded: {basis_df.shape}")
print(f"✅ DV01 loaded:  {dv01_df.shape}")
print(f"📅 Date range: {basis_df.index.min()} to {basis_df.index.max()}")

# Estrai lista ISIN
isin_list = [col.replace('_Basis', '') for col in basis_df.columns]
print(f"📊 ISIN: {len(isin_list)}")

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

# Stampa terzili storici come suggerimento per calibrare le soglie
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
    # Cerca il livello più recente disponibile fino a quella data
    available = itrx_main_daily[itrx_main_daily.index <= date]
    if len(available) == 0:
        return FEE_MID_BPS  # fallback se non ci sono dati
    level = available.iloc[-1]
    if level < ITRAXX_MAIN_LOW_THRESHOLD:
        return FEE_LOW_BPS
    elif level >= ITRAXX_MAIN_HIGH_THRESHOLD:
        return FEE_HIGH_BPS
    else:
        return FEE_MID_BPS

# ============================================================================
# STEP 2: MATURITY DATES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Maturity dates")
print("=" * 80)

maturity_dates = {
    'IT0005332835': pd.Timestamp('2026-05-21'),
    'IT0005388175': pd.Timestamp('2027-10-28'),
    'IT0005527223': pd.Timestamp('2028-05-14'),
    'IT0005517195': pd.Timestamp('2028-11-22'),
    'IT0005497000': pd.Timestamp('2030-06-26'),
    'IT0005648258': pd.Timestamp('2032-06-04'),
    'IT0004806888': pd.Timestamp('2016-03-26'),
    'IT0004821432': pd.Timestamp('2016-11-06'),
    'IT0004863608': pd.Timestamp('2016-10-22'),
    'IT0004917958': pd.Timestamp('2017-04-22'),
    'IT0004969207': pd.Timestamp('2017-11-12'),
    'IT0005012783': pd.Timestamp('2020-04-23'),
    'IT0005058919': pd.Timestamp('2020-10-27'),
    'IT0005361678': pd.Timestamp('2022-11-26'),
    'IT0005105843': pd.Timestamp('2023-04-20'),
    'IT0005253676': pd.Timestamp('2023-05-22'),
    'IT0005312142': pd.Timestamp('2023-11-20'),
    'IT0005174906': pd.Timestamp('2024-04-11'),
    'IT0005217770': pd.Timestamp('2024-10-24'),
    'IT0005410912': pd.Timestamp('2025-05-26'),
}

print(f"✅ Caricate {len(maturity_dates)} maturity dates")
print(f"\n   Primi 5 ISIN:")
for i, (isin, mat_date) in enumerate(list(maturity_dates.items())[:5]):
    print(f"      {isin} → {mat_date.strftime('%Y-%m-%d')}")

def months_to_maturity(date, isin):
    """Calcola quanti mesi mancano alla scadenza dell'ISIN"""
    if isin not in maturity_dates:
        return None
    mat_date = maturity_dates[isin]
    if date > mat_date:
        return 0
    months = (mat_date.year - date.year) * 12 + (mat_date.month - date.month)
    return months

# ============================================================================
# STEP 3: GENERA DATE DI CHECK ENTRY + CALCOLA THETA GIORNALIERO
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Generazione date di controllo + calcolo theta")
print("=" * 80)

all_dates = basis_df.index

if all_dates.isna().any():
    print("⚠️  ATTENZIONE: Trovati NaT nell'indice, rimozione in corso...")
    all_dates = all_dates[~all_dates.isna()]
    print(f"✅ NaT rimossi. Nuova data range: {all_dates.min()} to {all_dates.max()}")

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

# --- CALCOLO THETA AGGREGATO GIORNALIERO (Rebonato Eq. 15-16) ---
# theta_t = media della basis aggregata (cross-sectional mean di tutte le basis) fino a t

print(f"\n📐 Calcolo theta aggregato ({THETA_WINDOW_TYPE} window)...")

# Calcola basis aggregata giornaliera = media cross-sectional di tutte le basis disponibili
basis_agg_daily = basis_df.mean(axis=1)

# Calcola theta: expanding o rolling
if THETA_WINDOW_TYPE == "expanding":
    theta_daily = basis_agg_daily.expanding(min_periods=1).mean()
elif THETA_WINDOW_TYPE == "rolling":
    theta_daily = basis_agg_daily.rolling(window=THETA_ROLLING_WINDOW, min_periods=1).mean()
else:
    raise ValueError(f"THETA_WINDOW_TYPE non valido: {THETA_WINDOW_TYPE}")

# Calcola la data da cui theta è "maturo" (dopo THETA_MIN_MONTHS mesi di dati)
first_date = all_dates[0]
theta_start_date = first_date + pd.DateOffset(months=THETA_MIN_MONTHS)

print(f"✅ Theta calcolato: {len(theta_daily)} giorni")
print(f"📅 Prima data dati: {first_date.strftime('%Y-%m-%d')}")
print(f"📅 Theta maturo da: {theta_start_date.strftime('%Y-%m-%d')} (dopo {THETA_MIN_MONTHS} mesi)")
print(f"📊 Theta medio: {theta_daily.mean():.2f} bps")
print(f"📊 Theta ultimo: {theta_daily.iloc[-1]:.2f} bps")

# ============================================================================
# STEP 4: SIMULAZIONE TRADING (IDENTICA A 02a)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Simulazione trading")
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

    # === STEP 4.1: AGGIORNA P&L E CHECK EXIT PER TRADE APERTI ===

    trades_to_close = []

    for trade_id, trade in open_trades.items():
        isin = trade['isin']
        col_basis = f"{isin}_Basis"
        col_dv01 = f"{isin}_DV01"
        
        current_basis = basis_df.loc[date, col_basis]
        current_dv01 = dv01_df.loc[date, col_dv01]
        
        # Gestione NaN
        if pd.isna(current_basis):
            mat_date = maturity_dates.get(isin)
            if mat_date is not None and date >= mat_date:
                trade['exit_date'] = date
                trade['exit_basis'] = 0
                trade['exit_dv01'] = 0.0   # bond scaduto
                trade['exit_reason'] = 'MATURITY'
                trades_to_close.append(trade_id)
                continue
            else:
                continue
        
        prev_basis = trade.get('prev_basis', trade['entry_basis'])
        days_elapsed = (date - trade['prev_date']).days if 'prev_date' in trade else 1
        
        if pd.notna(prev_basis) and pd.notna(current_basis) and pd.notna(current_dv01):
            sign = 1 if trade['direction'] == 'LONG' else -1
            capital_gain_bps = sign * (prev_basis - current_basis) * current_dv01
            carry_bps = sign * prev_basis * (days_elapsed / 365.0)
            daily_pnl_bps = capital_gain_bps + carry_bps
            daily_pnl_pct = daily_pnl_bps / 100.0
            
            trade['cumulative_pnl'] = trade.get('cumulative_pnl', 0) + daily_pnl_pct
            trade['cumulative_capital_gain'] = trade.get('cumulative_capital_gain', 0) + (capital_gain_bps / 100.0)
            trade['cumulative_carry'] = trade.get('cumulative_carry', 0) + (carry_bps / 100.0)
        
        trade['prev_basis'] = current_basis
        trade['prev_date'] = date
        if pd.notna(current_dv01):
            trade['prev_dv01'] = current_dv01
        
        # Check exit
        months_left = months_to_maturity(date, isin)
        
        if months_left is not None and (months_left == 0 or date >= maturity_dates[isin]):
            trade['exit_date'] = date
            trade['exit_basis'] = 0
            trade['exit_dv01'] = 0.0   # bond scaduto, DV01 → 0
            trade['exit_reason'] = 'MATURITY'
            trades_to_close.append(trade_id)
            continue
        
        can_close = len(open_trades) > MIN_OPEN_TRADES or not first_trade_opened
        
        if can_close:
            if trade['direction'] == 'LONG' and current_basis < EXIT_LONG_THRESHOLD:
                trade['exit_date'] = date
                trade['exit_basis'] = current_basis
                trade['exit_dv01'] = current_dv01 if pd.notna(current_dv01) else trade.get('prev_dv01', trade['entry_dv01'])
                trade['exit_reason'] = 'TARGET_HIT'
                trades_to_close.append(trade_id)
            elif trade['direction'] == 'SHORT' and current_basis > EXIT_SHORT_THRESHOLD:
                trade['exit_date'] = date
                trade['exit_basis'] = current_basis
                trade['exit_dv01'] = current_dv01 if pd.notna(current_dv01) else trade.get('prev_dv01', trade['entry_dv01'])
                trade['exit_reason'] = 'TARGET_HIT'
                trades_to_close.append(trade_id)

    for trade_id in trades_to_close:
        trade = open_trades.pop(trade_id)
        trades_log.append(trade.copy())

    # === STEP 4.2: CHECK NUOVI ENTRY (solo nei giorni di check) ===
    
    if date in check_dates:
        candidates = []
        
        for isin in isin_list:
            col_basis = f"{isin}_Basis"
            col_dv01 = f"{isin}_DV01"
            
            basis = basis_df.loc[date, col_basis]
            dv01 = dv01_df.loc[date, col_dv01]
            
            if pd.isna(basis) or pd.isna(dv01):
                continue
            if basis > MAX_BASIS_ENTRY or basis < MIN_BASIS_ENTRY:
                continue
            
            months_left = months_to_maturity(date, isin)
            if months_left is None or months_left < MIN_MONTHS_TO_MATURITY_ENTRY:
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
                                if abs(t['entry_basis']) > abs(last_entry_basis):
                                    last_entry_basis = t['entry_basis']
                    
                    if last_entry_basis is not None:
                        if last_entry_basis > 0:
                            if basis > 0 and basis < last_entry_basis + REENTRY_BASIS_WIDENING:
                                continue
                        else:
                            if basis < 0 and basis > last_entry_basis - REENTRY_BASIS_WIDENING:
                                continue

                                
            # Cap max trade contemporanei sullo stesso ISIN
            if MAX_CONCURRENT_PER_ISIN is not None:
                n_open_same = sum(1 for t in open_trades.values() if t['isin'] == isin)
                if n_open_same >= MAX_CONCURRENT_PER_ISIN:
                    continue
            
           
            # Check soglie entry
            direction = None
            if basis > ENTRY_LONG_THRESHOLD:
                direction = 'LONG'
            elif basis < ENTRY_SHORT_THRESHOLD:
                direction = 'SHORT'
            
            if direction:
                # --- REBONATO: calcola il segnale all'entry ---
                theta_t = theta_daily.loc[date] if date in theta_daily.index else np.nan
                if pd.notna(theta_t):
                    signal = abs(basis - theta_t)
                else:
                    signal = 1.0  # fallback se theta non disponibile
                
                # Se theta non è ancora maturo, signal = 1 (equal-weight warm-up)
                if date < theta_start_date:
                    signal = 1.0
                
                candidates.append({
                    'isin': isin,
                    'basis': basis,
                    'dv01': dv01,
                    'direction': direction,
                    'abs_basis': abs(basis),
                    'signal': signal,
                    'theta_t': theta_t if pd.notna(theta_t) else np.nan
                })
        
        # Apri trade sul migliore
        if candidates:
            candidates.sort(key=lambda x: x['abs_basis'], reverse=True)
            best = candidates[0]
            
            new_trade = {
                'trade_id': trade_id_counter,
                'isin': best['isin'],
                'entry_date': date,
                'entry_basis': best['basis'],
                'entry_dv01': best['dv01'],
                'direction': best['direction'],
                'exit_date': None,
                'exit_basis': None,
                'exit_reason': None,
                'cumulative_pnl': 0,
                'prev_basis': best['basis'],
                'prev_date': date,
                # Rebonato signal fields
                'signal': best['signal'],
                'theta_at_entry': best['theta_t']
            }
            
            open_trades[trade_id_counter] = new_trade
            trade_id_counter += 1
            first_trade_opened = True

# Chiudi trade ancora aperti alla fine del sample
last_date = all_dates[-1]

print(f"\n🔍 DEBUG: Trade ancora aperti alla fine: {len(open_trades)}")
for tid, t in open_trades.items():
    print(f"   Trade #{tid}: {t['isin']}, entry={t['entry_date']}, exit={t.get('exit_date', 'None')}")

for trade_id in list(open_trades.keys()):
    trade = open_trades[trade_id]
    isin = trade['isin']
    col_basis = f"{isin}_Basis"
    
    mat_date = maturity_dates.get(isin)
    if mat_date is not None and last_date >= mat_date:
        trade['exit_date'] = last_date
        trade['exit_basis'] = 0
        trade['exit_dv01'] = 0.0   # bond scaduto
        trade['exit_reason'] = 'MATURITY'
    else:
        trade['exit_date'] = last_date
        exit_basis = basis_df.loc[last_date, col_basis]
        if pd.isna(exit_basis):
            exit_basis = trade.get('prev_basis', 0)
        trade['exit_basis'] = exit_basis
        col_dv01 = f"{isin}_DV01"
        exit_dv01 = dv01_df.loc[last_date, col_dv01]
        if pd.isna(exit_dv01):
            exit_dv01 = trade.get('prev_dv01', trade['entry_dv01'])
        trade['exit_dv01'] = exit_dv01
        trade['exit_reason'] = 'END_OF_SAMPLE'
    
    open_trades.pop(trade_id)
    trades_log.append(trade.copy())

print(f"\n✅ Simulazione completata!")
print(f"📊 Trade totali eseguiti: {len(trades_log)}")

# ============================================================================
# STEP 5: CREA DATAFRAME TRADES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Creazione log dei trade")
print("=" * 80)

if len(trades_log) == 0:
    print("❌ ERRORE: Nessun trade eseguito con questi parametri!")
    print("   Prova a:")
    print("   - Abbassare ENTRY_LONG_THRESHOLD")
    print("   - Alzare (in valore assoluto) ENTRY_SHORT_THRESHOLD")
    print("   - Aumentare MAX_BASIS_ENTRY")
    exit()

trades_df = pd.DataFrame(trades_log)
trades_df['duration_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
trades_df['duration_trading_days'] = trades_df.apply(
    lambda r: ((all_dates >= r['entry_date']) & (all_dates <= r['exit_date'])).sum(), axis=1)
trades_df['pnl_bps'] = trades_df['cumulative_pnl']

# Assicura che exit_dv01 esista (fallback a entry_dv01 se mancante)
if 'exit_dv01' not in trades_df.columns:
    trades_df['exit_dv01'] = trades_df['entry_dv01']
trades_df['exit_dv01'] = trades_df['exit_dv01'].fillna(trades_df['entry_dv01'])

print(f"✅ Trade log creato: {len(trades_df)} trade")
print(f"\n📈 Statistiche Trade:")
print(f"   Long:  {len(trades_df[trades_df['direction']=='LONG'])}")
print(f"   Short: {len(trades_df[trades_df['direction']=='SHORT'])}")
print(f"   Durata media: {trades_df['duration_days'].mean():.0f} giorni")
print(f"   P&L medio: {trades_df['pnl_bps'].mean():.2f}%")

# --- REBONATO: Stampa statistiche segnale ---
print(f"\n📐 REBONATO SIGNAL STATISTICS:")
print(f"   Signal medio: {trades_df['signal'].mean():.2f}")
print(f"   Signal mediano: {trades_df['signal'].median():.2f}")
print(f"   Signal min: {trades_df['signal'].min():.2f}")
print(f"   Signal max: {trades_df['signal'].max():.2f}")
print(f"   Theta medio all'entry: {trades_df['theta_at_entry'].mean():.2f} bps")
warm_up_trades = trades_df[trades_df['entry_date'] < theta_start_date]
signal_trades = trades_df[trades_df['entry_date'] >= theta_start_date]
print(f"   Trade in warm-up (peso=1): {len(warm_up_trades)}")
print(f"   Trade signal-weighted: {len(signal_trades)}")

# Stampa primi 10 trade con segnale
print("\n" + "=" * 80)
print("🔍 DEBUG: Primi 10 Trade (con Rebonato signal)")
print("=" * 80)
print(trades_df[['isin', 'entry_date', 'exit_date', 'entry_basis', 'exit_basis',
                 'entry_dv01', 'duration_days', 'cumulative_pnl', 'exit_reason',
                 'signal', 'theta_at_entry']].head(10).to_string(index=False))

# Salva trades log
trades_path = RESULTS_DIR / "trades_log.csv"
trades_df.to_csv(trades_path, index=False)
print(f"\n💾 Salvato: {trades_path.name}")

# ============================================================================
# STEP 6: COSTRUZIONE INDICE - SIGNAL-WEIGHTED (REBONATO)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Costruzione indice dei return [REBONATO SIGNAL-WEIGHTED]")
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

# --- Ricostruisci P&L giornaliero per ogni trade (identico a 02a) ---
daily_returns = pd.DataFrame(index=all_dates)

for _, trade in trades_df_index.iterrows():
    trade_id = trade['trade_id']
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    
    trade_dates = all_dates[(all_dates >= entry_date) & (all_dates <= exit_date)]
    if len(trade_dates) == 0:
        continue
    
    isin = trade['isin']
    col_basis = f"{isin}_Basis"
    col_dv01 = f"{isin}_DV01"
    
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

    if trade_id == 1:
        print(f"\n🔍 DEBUG Trade #1 (primi 5 giorni):")
        print(f"   ISIN: {isin}")
        print(f"   Entry date: {entry_date}")
        print(f"   Entry basis: {prev_basis:.2f}")
        print(f"   Signal (peso): {trade['signal']:.2f}")

    # --- Passata 1: calcola return grezzi (senza fee), NaN dove mancano dati ---
    for idx, date in enumerate(trade_dates[1:]):
        current_basis = basis_df.loc[date, col_basis]
        current_dv01 = dv01_df.loc[date, col_dv01]

        if pd.isna(current_basis):
            mat_date = maturity_dates.get(isin)
            if mat_date is not None and date >= mat_date:
                current_basis = 0
            else:
                trade_returns.append(np.nan)
                continue

        days_elapsed = (date - prev_date).days

        if pd.notna(prev_basis) and pd.notna(current_basis) and pd.notna(current_dv01):
            sign = 1 if trade['direction'] == 'LONG' else -1
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
                print(f"   Day {idx+1} ({date.strftime('%Y-%m-%d')}): SKIPPED!")

        trade_returns.append(daily_ret_pct)
        prev_basis = current_basis
        prev_date = date

    # --- Passata 2: spalma fee_total sui soli giorni validi ---
    # I giorni NaN restano NaN (non entrano nella media EW).
    # La fee totale e' garantita uguale a fee_total_pct.
    n_valid = sum(1 for r in trade_returns if pd.notna(r))
    if n_valid > 0 and fee_total_pct > 0:
        daily_fee_pct = fee_total_pct / n_valid
        trade_returns = [r - daily_fee_pct if pd.notna(r) else np.nan
                         for r in trade_returns]
    
    daily_returns.loc[trade_dates[1:], f"trade_{trade_id}"] = trade_returns

# --- REBONATO: Calcolo indice signal-weighted ---
# Per ogni giorno, il return dell'indice è la media pesata dei trade attivi,
# con pesi proporzionali a |signal_k| = |basis_entry_k - theta_t_entry|
#
# index_return_t = Σ (weight_k × return_k,t) / Σ weight_k
# dove le somme sono sui trade attivi al giorno t

trade_cols = [col for col in daily_returns.columns if col.startswith('trade_')]

print(f"\n📐 Calcolo indice signal-weighted su {len(trade_cols)} trade...")

# Costruisci matrice dei pesi: stessa shape di daily_returns[trade_cols],
# ma con il peso del trade dove il trade è attivo, NaN altrove
weights_matrix = pd.DataFrame(index=all_dates, columns=trade_cols, dtype=float)

for col in trade_cols:
    weight = trade_weights[col]
    # Il peso è attivo dove il return del trade non è NaN
    active_mask = daily_returns[col].notna()
    weights_matrix.loc[active_mask, col] = weight

# Return pesato = somma(return_k * weight_k) / somma(weight_k) per trade attivi
weighted_returns = daily_returns[trade_cols] * weights_matrix
numerator = weighted_returns.sum(axis=1, skipna=True)
denominator = weights_matrix.sum(axis=1, skipna=True)

# Evita divisione per zero
denominator = denominator.replace(0, np.nan)

# === Signal-weighted (SW) ===
daily_returns['index_return_sw'] = numerator / denominator

# === Equal-weight (EW) ===
daily_returns['index_return_ew'] = daily_returns[trade_cols].mean(axis=1, skipna=True)

# Filtra da primo return in poi
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

# Stampa medie (sempre entrambe + selected)
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
# STEP 7: CALCOLA CUMULATIVE RETURN (EQUITY CURVE)
# ============================================================================

# === SELECTED ===
daily_returns['cumulative_return'] = daily_returns['index_return'].cumsum()
daily_returns['index_value'] = 100 * (1 + daily_returns['cumulative_return'] / 100)

# === SW ===
daily_returns['cumulative_return_sw'] = daily_returns['index_return_sw'].cumsum()
daily_returns['index_value_sw'] = 100 * (1 + daily_returns['cumulative_return_sw'] / 100)

# === EW ===
daily_returns['cumulative_return_ew'] = daily_returns['index_return_ew'].cumsum()
daily_returns['index_value_ew'] = 100 * (1 + daily_returns['cumulative_return_ew'] / 100)

print(f"✅ Equity curve calcolata")
print(f"📌 SELECTED MODE: {OUTPUT_RETURN_MODE.upper()}")
print(f"📈 Return totale (SELECTED): {daily_returns['cumulative_return'].iloc[-1]:.2f}%")
print(f"📈 Return totale (SW):       {daily_returns['cumulative_return_sw'].iloc[-1]:.2f}%")
print(f"📈 Return totale (EW):       {daily_returns['cumulative_return_ew'].iloc[-1]:.2f}%")
print(f"📈 Index finale (SELECTED):  {daily_returns['index_value'].iloc[-1]:.2f}")
print(f"📈 Index finale (SW):        {daily_returns['index_value_sw'].iloc[-1]:.2f}")
print(f"📈 Index finale (EW):        {daily_returns['index_value_ew'].iloc[-1]:.2f}")


# ============================================================================
# STEP 8: RESAMPLE A FREQUENZA DESIDERATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Resample a frequenza indice")
print("=" * 80)

if INDEX_FREQ == "daily":
    index_final = daily_returns[[
    'index_return', 'cumulative_return', 'index_value',            # SELECTED
    'index_return_sw', 'cumulative_return_sw', 'index_value_sw',   # SW
    'index_return_ew', 'cumulative_return_ew', 'index_value_ew']].copy()    # EW

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
# STEP 9: SALVA RISULTATI
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Salvataggio risultati")
print("=" * 80)

# Salva indice (con entrambe le versioni)
index_path = RESULTS_DIR / f"index_{INDEX_FREQ}.csv"
index_final.to_csv(index_path)
print(f"💾 Salvato: {index_path.name}")

# Salva daily returns completo
daily_path = RESULTS_DIR / "daily_returns_full.csv"
daily_returns.to_csv(daily_path)
print(f"💾 Salvato: {daily_path.name}")

# Salva parametri Rebonato per tracciabilità
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
    'ENTRY_LONG_THRESHOLD': ENTRY_LONG_THRESHOLD,
    'ENTRY_SHORT_THRESHOLD': ENTRY_SHORT_THRESHOLD,
    'EXIT_LONG_THRESHOLD': EXIT_LONG_THRESHOLD,
    'EXIT_SHORT_THRESHOLD': EXIT_SHORT_THRESHOLD,
    'MIN_MONTHS_TO_MATURITY_ENTRY': MIN_MONTHS_TO_MATURITY_ENTRY,
    'MIN_OPEN_TRADES': MIN_OPEN_TRADES,
    'ALLOW_MULTIPLE_ENTRIES': ALLOW_MULTIPLE_ENTRIES,
    'REENTRY_BASIS_WIDENING': REENTRY_BASIS_WIDENING,
    'ENTRY_CHECK_FREQ': ENTRY_CHECK_FREQ,
    'INDEX_FREQ': INDEX_FREQ,
    'OUTPUT_RETURN_MODE': OUTPUT_RETURN_MODE,
}
params_df = pd.DataFrame(list(params_dict.items()), columns=['Parameter', 'Value'])
params_path = RESULTS_DIR / "parameters.csv"
params_df.to_csv(params_path, index=False)
print(f"💾 Salvato: {params_path.name}")

# ============================================================================
# STEP 10: STATISTICHE FINALI + CONFRONTO
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICHE FINALI — CONFRONTO SIGNAL-WEIGHTED vs EQUAL-WEIGHT")
print("=" * 80)

print(f"\n📊 PARAMETRI USATI:")
print(f"   Entry Long:  > {ENTRY_LONG_THRESHOLD} bps")
print(f"   Entry Short: < {ENTRY_SHORT_THRESHOLD} bps")
print(f"   Exit Long:   < {EXIT_LONG_THRESHOLD} bps")
print(f"   Exit Short:  > {EXIT_SHORT_THRESHOLD} bps")
print(f"   Min months to maturity: {MIN_MONTHS_TO_MATURITY_ENTRY}")
print(f"   Min open trades: {MIN_OPEN_TRADES}")
print(f"   Multiple entries: {'Yes' if ALLOW_MULTIPLE_ENTRIES else 'No'}")
if ALLOW_MULTIPLE_ENTRIES:
    print(f"   Reentry widening: {REENTRY_BASIS_WIDENING} bps")
print(f"\n💸 TRANSACTION FEES (dinamiche su iTraxx Main):")
print(f"   Fee LOW:  {FEE_LOW_BPS} bps  (iTraxx Main < {ITRAXX_MAIN_LOW_THRESHOLD} bps)")
print(f"   Fee MID:  {FEE_MID_BPS} bps  (iTraxx Main {ITRAXX_MAIN_LOW_THRESHOLD}-{ITRAXX_MAIN_HIGH_THRESHOLD} bps)")
print(f"   Fee HIGH: {FEE_HIGH_BPS} bps  (iTraxx Main > {ITRAXX_MAIN_HIGH_THRESHOLD} bps)")
print(f"\n📐 REBONATO PARAMETERS:")
print(f"   Theta window: {THETA_WINDOW_TYPE}")
if THETA_WINDOW_TYPE == "rolling":
    print(f"   Rolling window: {THETA_ROLLING_WINDOW} days")
print(f"   Theta min months: {THETA_MIN_MONTHS}")
print(f"   Normalize signal: {'Yes' if NORMALIZE_SIGNAL else 'No'}")
if SIGNAL_FLOOR > 0 or SIGNAL_CAP > 0:
    print(f"   Signal floor: {SIGNAL_FLOOR}")
    print(f"   Signal cap: {SIGNAL_CAP}")

# Calcola metriche per entrambe le versioni
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
    
    # Max drawdown
    cum = valid.cumsum()
    running_max = cum.expanding().max()
    dd = cum - running_max
    max_dd = dd.min()
    
    # Skewness & Kurtosis
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
print("   • Confronta con 02a (equal-weight originale)")
print("   • Calcola MPPM (04_mppm_analysis.py) su questo indice")
print("   • Calcola Moreira-Muir (05_moreira_muir.py) su questo indice")
print()