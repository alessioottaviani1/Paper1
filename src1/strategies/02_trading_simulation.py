"""
Script 2: Simulazione Trading Strategy BTP-Italia Basis
========================================================
Questo script simula una strategia di trading sulle basis BTP-Italia vs Nominale.

LOGICA STRATEGIA:
1. Apre trade quando basis supera soglie (>60 o <-80 bps)
2. Chiude trade quando basis raggiunge target (<10 o >-10 bps)
3. Mantiene sempre almeno 1 trade aperto dopo il primo
4. Non entra in trade se < 6 mesi a scadenza
5. Trade aperti vanno tenuti fino a scadenza se scendono sotto 6 mesi

P&L = Basis(t-1) × days + ΔBasis × DV01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI STRATEGIA - MODIFICA QUESTI VALORI
# ============================================================================

# === ENTRY THRESHOLDS ===
ENTRY_LONG_THRESHOLD = 60        # bps - long BTP-Italia se basis > questo
ENTRY_SHORT_THRESHOLD = -80      # bps - short BTP-Italia se basis < questo

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

# === FREQUENCY ===
ENTRY_CHECK_FREQ = "daily"       # "daily", "weekly", "monthly"
INDEX_FREQ = "daily"             # "daily", "weekly", "monthly"

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: CARICA DATI
# ============================================================================

print("=" * 80)
print("STEP 1: Caricamento dati processati")
print("=" * 80)

basis_df = pd.read_parquet(PROCESSED_DATA_DIR / "basis_wide.parquet")
dv01_df = pd.read_parquet(PROCESSED_DATA_DIR / "dv01_wide.parquet")

# ✅ RIMUOVI EVENTUALI NaT DALL'INDICE
basis_df = basis_df[~basis_df.index.isna()]
dv01_df = dv01_df[~dv01_df.index.isna()]

print(f"✅ Basis loaded: {basis_df.shape}")
print(f"✅ DV01 loaded:  {dv01_df.shape}")
print(f"📅 Date range: {basis_df.index.min()} to {basis_df.index.max()}")

# Estrai lista ISIN (rimuovi '_Basis' dai nomi colonne)
isin_list = [col.replace('_Basis', '') for col in basis_df.columns]
print(f"📊 ISIN: {len(isin_list)}")

# ============================================================================
# STEP 2: MATURITY DATES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Maturity dates")
print("=" * 80)

# Dizionario con le date di scadenza dei bond
# MODIFICA QUESTE DATE SE AGGIUNGI NUOVI ISIN
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

# Funzione per calcolare mesi a scadenza
def months_to_maturity(date, isin):
    """Calcola quanti mesi mancano alla scadenza dell'ISIN"""
    if isin not in maturity_dates:
        return None
    
    mat_date = maturity_dates[isin]
    if date > mat_date:
        return 0  # Già scaduto
    
    # Calcola differenza in mesi
    months = (mat_date.year - date.year) * 12 + (mat_date.month - date.month)
    return months

# ============================================================================
# STEP 3: GENERA DATE DI CHECK ENTRY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Generazione date di controllo")
print("=" * 80)

# Genera le date in cui controllare entry (daily/weekly/monthly)
all_dates = basis_df.index

# ✅ VERIFICA CHE NON CI SIANO NaT
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

# ============================================================================
# STEP 4: SIMULAZIONE TRADING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Simulazione trading")
print("=" * 80)

# Struttura dati per i trade
trades_log = []
trade_id_counter = 1
first_trade_opened = False

# Dizionario per tracciare trade aperti: {trade_id: trade_info}
open_trades = {}

print("🔄 Inizio simulazione...")
print()

# Itera su tutte le date (giornaliere) per aggiornare P&L e check exit
for i, date in enumerate(all_dates):
    
    # Progress bar
    if i % 500 == 0:
        print(f"   Processing: {date.strftime('%Y-%m-%d')} ({i}/{len(all_dates)})")



    

    # === STEP 4.1: AGGIORNA P&L E CHECK EXIT PER TRADE APERTI ===

    trades_to_close = []

    for trade_id, trade in open_trades.items():
        isin = trade['isin']
        col_basis = f"{isin}_Basis"
        col_dv01 = f"{isin}_DV01"
        
        # Prendi basis e DV01 correnti (al tempo t)
        current_basis = basis_df.loc[date, col_basis]
        current_dv01 = dv01_df.loc[date, col_dv01]
        
        # === GESTIONE NaN: DISTINGUI TRA MATURITY, MISSING DATA, E END OF SAMPLE ===
        if pd.isna(current_basis):
            mat_date = maturity_dates.get(isin)
            
            # Caso 1: Bond effettivamente scaduto (date >= maturity)
            if mat_date is not None and date >= mat_date:
                trade['exit_date'] = date
                trade['exit_basis'] = 0  # Basis converge a 0 a scadenza
                trade['exit_reason'] = 'MATURITY'
                trades_to_close.append(trade_id)
                continue
                     
            # Caso 3: Dati mancanti ma bond non scaduto e non fine sample
            # → Salta questo giorno e continua (non aggiornare P&L)
            else:
                continue
        
        # Se arriviamo qui, abbiamo dati validi → calcola P&L
        prev_basis = trade.get('prev_basis', trade['entry_basis'])
        days_elapsed = (date - trade['prev_date']).days if 'prev_date' in trade else 1
        
        # P&L giornaliero = Capital Gain + Carry
        if pd.notna(prev_basis) and pd.notna(current_basis) and pd.notna(current_dv01):
            sign = 1 if trade['direction'] == 'LONG' else -1
            capital_gain_bps = sign * (prev_basis - current_basis) * current_dv01
            carry_bps = sign * prev_basis * (days_elapsed / 365.0)
            daily_pnl_bps = capital_gain_bps + carry_bps
            
            # Converti in %
            daily_pnl_pct = daily_pnl_bps / 100.0
            
            # Aggiorna cumulative P&L
            trade['cumulative_pnl'] = trade.get('cumulative_pnl', 0) + daily_pnl_pct
            trade['cumulative_capital_gain'] = trade.get('cumulative_capital_gain', 0) + (capital_gain_bps / 100.0)
            trade['cumulative_carry'] = trade.get('cumulative_carry', 0) + (carry_bps / 100.0)
        
        trade['prev_basis'] = current_basis
        trade['prev_date'] = date
        
        # === CHECK EXIT CONDITIONS (solo se non siamo in caso NaN) ===
        
        months_left = months_to_maturity(date, isin)
        
        # Condizione 1: Scadenza (0 mesi rimasti o date >= maturity)
        if months_left is not None and (months_left == 0 or date >= maturity_dates[isin]):
            trade['exit_date'] = date
            trade['exit_basis'] = 0  # Basis converge a 0 a scadenza
            trade['exit_reason'] = 'MATURITY'
            trades_to_close.append(trade_id)
            continue
        
        # Condizione 2: Target hit (solo se non è l'ultimo trade)
        can_close = len(open_trades) > MIN_OPEN_TRADES or not first_trade_opened
        
        if can_close:
            if trade['direction'] == 'LONG' and current_basis < EXIT_LONG_THRESHOLD:
                trade['exit_date'] = date
                trade['exit_basis'] = current_basis
                trade['exit_reason'] = 'TARGET_HIT'
                trades_to_close.append(trade_id)
            elif trade['direction'] == 'SHORT' and current_basis > EXIT_SHORT_THRESHOLD:
                trade['exit_date'] = date
                trade['exit_basis'] = current_basis
                trade['exit_reason'] = 'TARGET_HIT'
                trades_to_close.append(trade_id)

    # Chiudi i trade
    for trade_id in trades_to_close:
        trade = open_trades.pop(trade_id)
        trades_log.append(trade.copy())

    
    # === STEP 4.2: CHECK NUOVI ENTRY (solo nei giorni di check) ===
    
    if date in check_dates:
        
        # Trova candidati validi
        candidates = []
        
        for isin in isin_list:
            col_basis = f"{isin}_Basis"
            col_dv01 = f"{isin}_DV01"
            
            basis = basis_df.loc[date, col_basis]
            dv01 = dv01_df.loc[date, col_dv01]
            
            # Skip se dati mancanti
            if pd.isna(basis) or pd.isna(dv01):
                continue
            
            # Skip outliers
            if basis > MAX_BASIS_ENTRY or basis < MIN_BASIS_ENTRY:
                continue
            
            # Check months to maturity
            months_left = months_to_maturity(date, isin)
            if months_left is None or months_left < MIN_MONTHS_TO_MATURITY_ENTRY:
                continue
            
            # Check se già aperto trade su questo ISIN
            already_open = any(t['isin'] == isin for t in open_trades.values())
            if already_open:
                continue
            
            # Check soglie entry
            direction = None
            if basis > ENTRY_LONG_THRESHOLD:
                direction = 'LONG'
            elif basis < ENTRY_SHORT_THRESHOLD:
                direction = 'SHORT'
            
            if direction:
                candidates.append({
                    'isin': isin,
                    'basis': basis,
                    'dv01': dv01,
                    'direction': direction,
                    'abs_basis': abs(basis)
                })
        
        # Se ci sono candidati, apri trade sul migliore (basis più alta in valore assoluto)
        if candidates:
            # Ordina per valore assoluto della basis (più alta = più attraente)
            candidates.sort(key=lambda x: x['abs_basis'], reverse=True)
            best = candidates[0]
            
            # Apri nuovo trade
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
                'prev_date': date
            }
            
            open_trades[trade_id_counter] = new_trade
            trade_id_counter += 1
            first_trade_opened = True

# ✅ FUORI DAL LOOP - Chiudi eventuali trade ancora aperti alla fine del sample
last_date = all_dates[-1]

print(f"\n🔍 DEBUG: Trade ancora aperti alla fine: {len(open_trades)}")
for tid, t in open_trades.items():
    print(f"   Trade #{tid}: {t['isin']}, entry={t['entry_date']}, exit={t.get('exit_date', 'None')}")

for trade_id in list(open_trades.keys()):
    trade = open_trades[trade_id]
    isin = trade['isin']
    col_basis = f"{isin}_Basis"
    
    print(f"\n🔍 Chiudendo trade #{trade_id} ({isin})...")
    
    mat_date = maturity_dates.get(isin)
    if mat_date is not None and last_date >= mat_date:
        # Bond scaduto
        trade['exit_date'] = last_date
        trade['exit_basis'] = 0
        trade['exit_reason'] = 'MATURITY'
        print(f"   → MATURITY: exit_date={last_date}")
    else:
        # Fine sample - usa ultimo basis disponibile
        trade['exit_date'] = last_date
        exit_basis = basis_df.loc[last_date, col_basis]
        if pd.isna(exit_basis):
            exit_basis = trade.get('prev_basis', 0)
        trade['exit_basis'] = exit_basis
        trade['exit_reason'] = 'END_OF_SAMPLE'
        print(f"   → END_OF_SAMPLE: exit_date={last_date}, exit_basis={exit_basis:.2f}")
    
    # Verifica che exit_date sia popolato
    print(f"   ✓ Trade dopo modifica: exit_date={trade['exit_date']}, exit_reason={trade['exit_reason']}")
    
    # Rimuovi da open_trades e aggiungi a log
    open_trades.pop(trade_id)
    trades_log.append(trade.copy())

print(f"\n✅ Simulazione completata!")
print(f"📊 Trade totali eseguiti: {len(trades_log)}")
print(f"📊 Trade in open_trades: {len(open_trades)}")

# 🔍 Verifica l'ultimo trade nel log
if len(trades_log) > 0:
    last_trade = trades_log[-1]
    print(f"\n🔍 Ultimo trade nel log:")
    print(f"   ID: {last_trade['trade_id']}")
    print(f"   ISIN: {last_trade['isin']}")
    print(f"   Entry: {last_trade['entry_date']}")
    print(f"   Exit: {last_trade['exit_date']}")
    print(f"   Reason: {last_trade['exit_reason']}")

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

# Calcola durata trade
trades_df['duration_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days

# Calcola P&L finale
trades_df['pnl_bps'] = trades_df['cumulative_pnl']

print(f"✅ Trade log creato: {len(trades_df)} trade")
print(f"\n📈 Statistiche Trade:")
print(f"   Long:  {len(trades_df[trades_df['direction']=='LONG'])}")
print(f"   Short: {len(trades_df[trades_df['direction']=='SHORT'])}")
print(f"   Durata media: {trades_df['duration_days'].mean():.0f} giorni")
print(f"   P&L medio: {trades_df['pnl_bps'].mean():.2f}%")

# === DEBUG: STAMPA PRIMI 10 TRADE ===
print("\n" + "=" * 80)
print("🔍 DEBUG: Primi 10 Trade (con dettaglio Carry vs Capital Gain)")
print("=" * 80)
print(trades_df[['isin', 'entry_date', 'exit_date', 'entry_basis', 'exit_basis', 
                 'entry_dv01', 'duration_days', 'cumulative_capital_gain', 'cumulative_carry',
                 'cumulative_pnl', 'exit_reason']].head(10).to_string(index=False))

# Verifica formula per primo trade
first = trades_df.iloc[0]
delta_basis = first['entry_basis'] - first['exit_basis']
expected_capital_gain_bps = delta_basis * first['entry_dv01']
expected_carry_bps = first['entry_basis'] * (first['duration_days'] / 365)  # Carry approssimato
expected_pnl_bps = expected_capital_gain_bps + expected_carry_bps
expected_pnl_pct = expected_pnl_bps / 100
print(f"\n🧮 Verifica Trade #1:")
print(f"   ΔBasis: {delta_basis:.2f} bps")
print(f"   DV01: {first['entry_dv01']:.4f}")
print(f"   Durata: {first['duration_days']:.0f} giorni")
print(f"   Capital Gain: {expected_capital_gain_bps:.2f} bps = {expected_capital_gain_bps/100:.2f}%")
print(f"   Carry (stimato): {expected_carry_bps:.2f} bps = {expected_carry_bps/100:.2f}%")
print(f"   P&L Teorico: {expected_pnl_pct:.2f}%")
print(f"   P&L Effettivo: {first['cumulative_pnl']:.2f}%")

# Verifica Trade #2 (quello problematico)
if len(trades_df) > 1:
    second = trades_df.iloc[2]  # IT0004821432
    delta_basis_2 = second['entry_basis'] - second['exit_basis']
    expected_cg_2 = delta_basis_2 * second['entry_dv01']
    expected_carry_2 = second['entry_basis'] * (second['duration_days'] / 365)
    print(f"\n🧮 Verifica Trade #2 (IT0004821432) - PROBLEMATICO:")
    print(f"   ISIN: {second['isin']}")
    print(f"   Entry Basis: {second['entry_basis']:.2f} bps")
    print(f"   Exit Basis: {second['exit_basis']:.2f} bps")
    print(f"   ΔBasis: {delta_basis_2:.2f} bps")
    print(f"   DV01: {second['entry_dv01']:.4f}")
    print(f"   Durata: {second['duration_days']:.0f} giorni")
    print(f"   --- ATTESO ---")
    print(f"   Capital Gain atteso: {expected_cg_2:.2f} bps = {expected_cg_2/100:.2f}%")
    print(f"   Carry atteso: {expected_carry_2:.2f} bps = {expected_carry_2/100:.2f}%")
    print(f"   P&L Totale atteso: {(expected_cg_2 + expected_carry_2)/100:.2f}%")
    print(f"   --- EFFETTIVO ---")
    print(f"   Capital Gain effettivo: {second['cumulative_capital_gain']:.2f}%")
    print(f"   Carry effettivo: {second['cumulative_carry']:.2f}%")
    print(f"   P&L Totale effettivo: {second['cumulative_pnl']:.2f}%")
    print(f"   --- DIFFERENZA ---")
    print(f"   Delta CG: {second['cumulative_capital_gain'] - expected_cg_2/100:.2f}%")
    print(f"   Delta Carry: {second['cumulative_carry'] - expected_carry_2/100:.2f}%")

# Salva trades log
trades_path = RESULTS_DIR / "trades_log.csv"
trades_df.to_csv(trades_path, index=False)
print(f"\n💾 Salvato: {trades_path.name}")

# ============================================================================
# STEP 6: COSTRUZIONE INDICE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Costruzione indice dei return")
print("=" * 80)

# Crea DataFrame con return giornaliero per ogni trade
# Struttura: date | trade_1_return | trade_2_return | ...

# Prima, ricostruiamo il P&L giornaliero di ogni trade
daily_returns = pd.DataFrame(index=all_dates)

for _, trade in trades_df.iterrows():
    trade_id = trade['trade_id']
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    
    # Filtra date del trade
    trade_dates = all_dates[(all_dates >= entry_date) & (all_dates <= exit_date)]
    
    if len(trade_dates) == 0:
        continue
    
    isin = trade['isin']
    col_basis = f"{isin}_Basis"
    col_dv01 = f"{isin}_DV01"  # AGGIUNTO: era mancante!
    
    # Calcola return giornaliero per questo trade
    trade_returns = []
    prev_basis = trade['entry_basis']
    prev_date = entry_date
    
    # DEBUG: stampa info primo trade
    if trade_id == 1:
        print(f"\n🔍 DEBUG Trade #1 (primi 5 giorni):")
        print(f"   ISIN: {isin}")
        print(f"   Entry date: {entry_date}")
        print(f"   Entry basis: {prev_basis:.2f}")
    
    for idx, date in enumerate(trade_dates[1:]):  # Skip entry date
        current_basis = basis_df.loc[date, col_basis]
        current_dv01 = dv01_df.loc[date, col_dv01]
        
        if pd.isna(current_basis):
            current_basis = 0  # Scadenza
        
        days_elapsed = (date - prev_date).days
        
        # P&L giornaliero = Capital Gain + Carry
        # Formula CORRETTA: usa current_dv01, non entry_dv01!
        if pd.notna(prev_basis) and pd.notna(current_basis) and pd.notna(current_dv01):
            sign = 1 if trade['direction'] == 'LONG' else -1
            capital_gain_bps = sign *(prev_basis - current_basis) * current_dv01  # current_dv01!
            carry_bps = sign *prev_basis * (days_elapsed / 365.0)
            daily_ret_bps = capital_gain_bps + carry_bps
            
            # Converti in % (formato Fama-French)
            daily_ret_pct = daily_ret_bps / 100.0
            
            # DEBUG: stampa primi 5 giorni del trade 1
            if trade_id == 1 and idx < 5:
                print(f"   Day {idx+1} ({date.strftime('%Y-%m-%d')}): prev={prev_basis:.2f}, curr={current_basis:.2f}, dv01={current_dv01:.4f}")
                print(f"      CG={capital_gain_bps:.4f} bps, Carry={carry_bps:.4f} bps, Total={daily_ret_pct:.6f}%")
        else:
            daily_ret_pct = 0  # Se mancano dati, assume 0 return
            
            # DEBUG: perché è 0?
            if trade_id == 1 and idx < 5:
                print(f"   Day {idx+1} ({date.strftime('%Y-%m-%d')}): SKIPPED!")
                print(f"      prev_basis={prev_basis}, current_basis={current_basis}, current_dv01={current_dv01}")
        
        trade_returns.append(daily_ret_pct)
        prev_basis = current_basis
        prev_date = date
    
    # Aggiungi al DataFrame (skip prima data = entry)
    daily_returns.loc[trade_dates[1:], f"trade_{trade_id}"] = trade_returns

# Calcola indice = equal-weighted average dei return
daily_returns['index_return'] = daily_returns.mean(axis=1, skipna=True)

# ⚠️ FIX: Rimuovi la entry date del primo trade (index return inizia il giorno DOPO)
first_trade_entry = trades_df['entry_date'].min()
first_return_date = first_trade_entry + pd.Timedelta(days=1)

# Filtra da first_return_date in poi
daily_returns = daily_returns[daily_returns.index >= first_return_date]

print(f"✅ Indice costruito: {len(daily_returns)} giorni")
print(f"📅 Dal {daily_returns.index.min()} al {daily_returns.index.max()}")
print(f"\n✅ Primo trade entry: {first_trade_entry.strftime('%Y-%m-%d')}")
print(f"✅ Primo return index: {first_return_date.strftime('%Y-%m-%d')}")
print(f"✅ Index return starts from: {daily_returns.index.min().strftime('%Y-%m-%d')}")

# Calcola statistiche SENZA il primo NaN
valid_returns = daily_returns['index_return'].dropna()
print(f"📊 Return medio giornaliero: {valid_returns.mean():.4f}%")

# === DEBUG: PRIME 20 DATE DELL'INDICE ===
print("\n" + "=" * 80)
print("🔍 DEBUG: Prime 20 Date dell'Indice")
print("=" * 80)
print(daily_returns[['index_return']].head(20).to_string())
print(f"\n   Somma primi 20 giorni: {daily_returns['index_return'].head(20).sum():.4f}%")

# ============================================================================
# STEP 7: CALCOLA CUMULATIVE RETURN (EQUITY CURVE)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Calcolo equity curve")
print("=" * 80)

# Cumulative return (in bps)
daily_returns['cumulative_return'] = daily_returns['index_return'].cumsum()

# Normalizza a base 100
daily_returns['index_value'] = 100 * (1 + daily_returns['cumulative_return'] / 100)

print(f"✅ Equity curve calcolata")
print(f"📈 Return totale: {daily_returns['cumulative_return'].iloc[-1]:.2f} bps")
print(f"📈 Index finale: {daily_returns['index_value'].iloc[-1]:.2f}")

# ============================================================================
# STEP 8: RESAMPLE A FREQUENZA DESIDERATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Resample a frequenza indice")
print("=" * 80)

if INDEX_FREQ == "daily":
    index_final = daily_returns[['index_return', 'cumulative_return', 'index_value']].copy()
elif INDEX_FREQ == "weekly":
    # Somma i return della settimana
    index_final = daily_returns[['index_return']].resample('W-FRI').sum()
    index_final['cumulative_return'] = index_final['index_return'].cumsum()
    index_final['index_value'] = 100 * (1 + index_final['cumulative_return'] / 100)

elif INDEX_FREQ == "monthly":
    # Somma i return del mese
    index_final = daily_returns[['index_return']].resample('M').sum()
    index_final['cumulative_return'] = index_final['index_return'].cumsum()
    index_final['index_value'] = 100 * (1 + index_final['cumulative_return'] / 100)
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

# Salva indice
index_path = RESULTS_DIR / f"index_{INDEX_FREQ}.csv"
index_final.to_csv(index_path)
print(f"💾 Salvato: {index_path.name}")

# Salva daily returns completo (per analisi)
daily_path = RESULTS_DIR / "daily_returns_full.csv"
daily_returns.to_csv(daily_path)
print(f"💾 Salvato: {daily_path.name}")

# ============================================================================
# STEP 10: STATISTICHE FINALI
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICHE FINALI")
print("=" * 80)

print(f"\n📊 PARAMETRI USATI:")
print(f"   Entry Long:  > {ENTRY_LONG_THRESHOLD} bps")
print(f"   Entry Short: < {ENTRY_SHORT_THRESHOLD} bps")
print(f"   Exit Long:   < {EXIT_LONG_THRESHOLD} bps")
print(f"   Exit Short:  > {EXIT_SHORT_THRESHOLD} bps")
print(f"   Min months to maturity: {MIN_MONTHS_TO_MATURITY_ENTRY}")
print(f"   Min open trades: {MIN_OPEN_TRADES}")

print(f"\n📈 PERFORMANCE:")
print(f"   Trade totali: {len(trades_df)}")
print(f"   Trade Long:   {len(trades_df[trades_df['direction']=='LONG'])} ({len(trades_df[trades_df['direction']=='LONG'])/len(trades_df)*100:.1f}%)")
print(f"   Trade Short:  {len(trades_df[trades_df['direction']=='SHORT'])} ({len(trades_df[trades_df['direction']=='SHORT'])/len(trades_df)*100:.1f}%)")
print(f"   Durata media: {trades_df['duration_days'].mean():.0f} giorni")
print(f"   P&L medio per trade: {trades_df['pnl_bps'].mean():.2f}%")
print(f"   Return totale: {index_final['cumulative_return'].iloc[-1]:.2f}%")
print(f"   Sharpe (annualizzato): {(index_final['index_return'].mean() / index_final['index_return'].std()) * np.sqrt(252):.2f}")

print("\n" + "=" * 80)
print("✅ SIMULAZIONE COMPLETATA CON SUCCESSO!")
print("=" * 80)

print(f"\n📁 File salvati in: {RESULTS_DIR}")
print(f"   • trades_log.csv")
print(f"   • index_{INDEX_FREQ}.csv")
print(f"   • daily_returns_full.csv")

print("\n🎯 Prossimo step:")
print("   • Analizza i risultati")
print("   • Modifica parametri se necessario")
print("   • Crea grafici (prossimo script)")
print()

last_date = all_dates[-1]

print(f"\n🔍 DEBUG last_date: {last_date}, tipo: {type(last_date)}")
print(f"🔍 all_dates[-1]: {all_dates[-1]}")
print(f"🔍 all_dates ultimi 3: {all_dates[-3:]}")