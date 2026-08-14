"""04 - REGRESSIONE: ricalcola la base BTP-eur-i con gli interruttori dell'ORIGINALE
(floor=True, is_total=True, ytm 'local', pub_lag=None, matching interpolato, prezzi ASK
se disponibili) e la confronta con btpei_basis.xlsx al centesimo di bp.

FINCHE' QUESTO NON PASSA, NON SI TOCCA NIENT'ALTRO: e' il vincolo che rende la
riscrittura una pulizia e non una riscrittura.

Prima di lanciare: 01, 02 (con TRADING=True), e btpei_basis.xlsx in data/."""

# ----------------------------------------------------------------- impostazioni
# DEVONO essere i valori usati per COSTRUIRE btpei_basis.xlsx: verificali nella
# chiamata a get_ytm dentro instructions_btpei.ipynb.
YEARS_HISTORY = 10
SEASON_WEIGHT = 1.0

# ----------------------------------------------------------------- esecuzione
import pipeline
pipeline.regression(years_history=YEARS_HISTORY, season_weight=SEASON_WEIGHT)
