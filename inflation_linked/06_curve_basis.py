"""06 - METODO A (curva vs curva): lambda(tau) = reale + ILS(cc) - nominale a scadenza
costante, sui soli mercati con curva reale pubblica:
  UK: BoE reale + BPSWIT - BoE nominale, pillar 3/5/10/30
  US: GSW TIPS + USSWIT - GSW nominale,  pillar 2/5/10/30
Output: curve_basis_UK.parquet, curve_basis_US.parquet (bp).
Tutto convertito a capitalizzazione continua: l'identita' di Fisher e' additiva solo li'.

Prima di lanciare: 02 (per gli ILS) e 03 (per le curve)."""

# ----------------------------------------------------------------- esecuzione
import pipeline

for mkt in ("UK", "US"):
    pipeline.curve_basis(mkt)
