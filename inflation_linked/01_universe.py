"""01 - UNIVERSO: legge Govt_bonds.xlsx, assegna i mercati, applica i filtri dichiarati
(funged, valute pre-euro, scaduti prima del floor 2003), salva universe.parquet con
l'audit trail completo (ogni riga esclusa porta la sua ragione) e stampa il report.

Offline: NON richiede il terminale Bloomberg.
Prima di lanciare: metti Govt_bonds.xlsx in data/raw/ (o cambia il percorso qui sotto).
Dopo il run: controlla su DES i linker segnalati (BASE_CPI mancante, lag UK incoerente)."""
from pathlib import Path

# ----------------------------------------------------------------- impostazioni
UNIVERSE_FILE = None      # None = data/raw/Govt_bonds.xlsx (default di config); oppure Path(...)

# ----------------------------------------------------------------- esecuzione
import bbg

if UNIVERSE_FILE is not None:
    bbg.UNIVERSE_FILE = Path(UNIVERSE_FILE)

uni = bbg.build_universe()
print(bbg.universe_report(uni))
print(f"\nsalvato: {bbg.CACHE / 'universe.parquet'}  ({len(uni)} strumenti, "
      f"{int(uni['incl'].sum())} inclusi)")
