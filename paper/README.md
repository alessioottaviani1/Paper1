# PhD Dissertation - EDHEC Business School
# Alessio Ottaviani — Supervisor: Prof. Riccardo Rebonato

## Two Compilable Versions

| File | Purpose | Tables/Figures |
|------|---------|----------------|
| `main_draft.tex` | Working skeleton for supervisor | Inline in the text |
| `main_final.tex` | EDHEC submission | At the end (EDHEC format) |

## Paper 1 Structure (8 sections + appendix)

```
1  Introduction
2  Literature Review
   2.1–2.8  (8 subsections)
3  Data, Strategy Construction, and Performance
   3.1–3.9  (incl. MPPM, Moreira-Muir, SW vs EW)
4  Benchmark Factor Models
   4.1–4.4  (Duarte, Fung-Hsieh, Active FI, Discussion)
5  Principal Component Analysis
   5.1–5.4  (Rolling PCA, Spanning, Conditional Alpha, Discussion)
6  Penalized Factor Selection: Adaptive Elastic Net
   6.1–6.6  (Factors, Preprocessing, AEN, Stability, Inference, Discussion)
7  Cross-Strategy Interdependencies
   7.1–7.6  (Mispricing, Correlations, Spanning, VAR, Conditional, Discussion)
8  Conclusion
References
Appendix
   A.1  BTP Italia: Instrument Details and Arbitrage Mechanics
   A.2  Trade Construction Details
   A.3  Candidate Factor Universe (full list + stationarity tests)
   A.4  PCA Robustness (W×K grid, subperiod)
   A.5  AEN Robustness (correlation, gamma, ALASSO, Ridge)
   A.6  Subperiod and Rolling Window Analysis
   A.7  Cross-Strategy Interdependencies: Additional Tests (TVAR, DCC, subperiod)
   A.8  VIF Diagnostics
Tables
Figures
```

## How to Compile

```bash
cd thesis/
./build.sh draft     # → main_draft.pdf
./build.sh final     # → main_final.pdf
```

## Python → Thesis Mapping

| VSC folder | Thesis section |
|------------|----------------|
| `src/strategies/` | Section 3 |
| `src/factor_models/01-06` | Sections 3–4 |
| `src/pca/` | Section 5 |
| `src/machine_learning/` | Section 6 |
| `src/rq3/` | Section 7 |
