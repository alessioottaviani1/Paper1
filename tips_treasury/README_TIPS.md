# src/tips_treasury — pipeline Paper 2 (TIPS–Treasury Floor)

## Setup (una volta sola)
1. Copia in `data/raw/Bloomberg/`: `TIPS_variables.xlsx`, `Cartel1.xlsx` (il `tips_cusip.xlsx` c'è già).
2. Copia da GitHub `TIPS data/` in `data/raw/`: le cartelle `Net_position/` e `OFR/` (HKM c'è già).
3. Crea `data/raw/FLL/` e mettici `fll_pairs_tableIII.csv` (incluso nel pacchetto).
4. `pip install statsmodels openpyxl` se non già nel venv.

## Ordine di esecuzione (dalla cartella src/tips_treasury)
`python 01_import_data.py` → processed in `data/processed/tips/`
poi `02` … `08` in ordine (ognuno scrive `results/tips/NN_*.txt` + figure in `results/tips/figures/`),
infine `09_make_tables.py` → frammenti LaTeX in `paper/tables_paper2/`.

## Mappa script → sezioni del paper (master v3)
| Script | Sezione | Output chiave |
|---|---|---|
| 02_validation | §4 | cross-check BE (0.996 al 10Y), CUSIP 0.926, FLL 46.8 vs 54.5 |
| 03_dynamics | §6.1–6.3 | medie NW, scala 2010-12/2017-12, OU/ADF, seasonal 52bp, U7 |
| 04_events_dispersion | §6.4–6.5 | eventi, regola pre uniforme, E7, U6 buckets, 2 figure |
| 05_constraint_test | §6.6 | Spec A–D, Chow quintile, LOO, composizione, 1 figura |
| 06_unwind_replay | §6.7 | superficie vintage×buffer (daily) |
| 07_structural_engine | §6.8 | stop-out daily, floors CE vs RRA, charge derivato 2.26/1.91% |
| 08_fll_replication | §6.9 | Fig-2 replica+extended, diagnostico Tabella III |

## Note
- Il design del constraint test è CONGELATO (Pilot Report v3 §4.3, 15-07-2026): non modificare
  LAG_H / Q_STRESS / GRID_Q in `config.py` senza dichiararlo come specification-curve.
- `plot_fill_figure2.py` (vecchio, refuso nel nome) è sostituito da `08_fll_replication.py`.
- Prossimo modulo: `10_smm.py` = merge dell'engine SMM di giugno (`tips_treasury_arb/`)
  con i momenti nuovi (superficie unwind + coppia di loading Spec-B + covarianza GKR).
