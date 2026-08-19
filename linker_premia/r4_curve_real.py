"""r4 - Curve REALI euro (IT, FR, DE) fittate dai linker col motore GSW-sui-prezzi
validato; per FR fitta anche il NOMINALE (nss_FR) se assente (Bundesbank copre DE,
l'Italia e' gia' fatta). Incrementale: le date gia' in cache vengono saltate.
Lancia:  python .\\src\\linker_premia\\r4_curve_real.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
import bbg, pipeline

ref = bbg.load("ref_linker")
col = next((c for c in ("MKT", "MARKET", "mercato", "mkt") if c in ref.columns), None)
if col is None:
    raise SystemExit(f"colonna mercato non trovata; colonne: {list(ref.columns)}")

# nominale FR (serve a bei_euro; IT gia' in cache, DE = Bundesbank via 03)
if not (pipeline.CACHE / "nss_FR.parquet").exists():
    pxn = pipeline.CACHE / "px_nom_FR.parquet"
    if not pxn.exists():
        print("!! px_nom_FR assente: lanciare il 02 (SOLO_MERCATO='FR', FASI NOMINALI=True)")
    else:
        ref_n = bbg.load("ref_nominal")
        ref_n_fr = ref_n[ref_n[col] == "FR"] if col in ref_n.columns else ref_n
        ytm = bbg.load("ytm_FR")
        px = pd.read_parquet(pxn); px.index = pd.to_datetime(px.index)
        par = pipeline._euro_nss_params("FR", ref_n_fr, ytm, px.index)
        print(f"nss_FR (nominale): {len(par)} date | rmse mediano {par['rmse_bp'].median():.1f}bp")

for mkt in ("IT", "FR", "DE"):
    pxp = pipeline.CACHE / f"px_mid_{mkt}.parquet"
    if not pxp.exists():
        print(f"{mkt}: px_mid_{mkt} assente, salto"); continue
    ref_m = ref[ref[col] == mkt]
    px = pd.read_parquet(pxp); px.index = pd.to_datetime(px.index)
    ytm_vuoto = pd.DataFrame(index=px.index)
    par = pipeline._euro_real_nss_params(mkt, ref_m, ytm_vuoto, px.index)
    print(f"nss_real_{mkt}: {len(par)} date | {len(ref_m)} linker | rmse mediano "
          f"{par['rmse_bp'].median():.1f}bp | >12bp: {(par['rmse_bp']>12).sum()}")
