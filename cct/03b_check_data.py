"""
03b - Controllo di qualita' dei dati scaricati. Gira in pochi secondi e dice se la
pipeline puo' proseguire o se manca qualcosa di essenziale.

Il controllo piu' importante e' FLT_SPREAD sui CCTeu: senza lo spread cedolare non si
puo' calcolare nessuna cedola CCTeu, quindi ne' lo scadenzario ne' la replica. Il log di
02 mostrava molte 'field exceptions' e solo 360 righe salvate in static_bbg.csv su ~1.460
titoli richiesti: va verificato che i 27 CCTeu siano fra quelle righe.
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

L=[]; P=L.append
P("=== 03b CONTROLLO DATI ===")

# ---- prezzi
for lab, tot in [("bot", 972), ("btp", 365), ("cct", 130)]:
    f = PROC / f"px_{lab}.csv"
    if not f.exists(): P(f"[MANCA] px_{lab}.csv"); continue
    px = pd.read_csv(f, index_col=0, parse_dates=True)
    cov = px.notna().sum(axis=1)
    P(f"{lab.upper():4s}: {px.shape[1]}/{tot} serie | {px.index.min().date()} -> {px.index.max().date()}"
      f" | titoli vivi per giorno: mediana {cov.median():.0f}, min {cov.min():.0f}")

# ---- curve
f = PROC / "curves_market.csv"
if f.exists():
    cm = pd.read_csv(f, index_col=0, parse_dates=True)
    P(f"\nCURVE: {cm.shape[1]} serie | {cm.index.min().date()} -> {cm.index.max().date()}")
    miss = [c for c in ["euribor6m", "irs5y", "ois5y"] if c not in cm.columns]
    if miss: P(f"  [!] mancano serie chiave: {miss}")
    if "euribor6m" in cm:
        e = cm["euribor6m"].dropna()
        P(f"  euribor6m: {len(e)} oss., min {e.min():.3f}% ({e.idxmin().date()}), "
          f"giorni negativi {(e<0).sum()}")

# ---- anagrafica Bloomberg: il punto critico
f = PROC / "static_bbg.csv"
P("\nANAGRAFICA BLOOMBERG (il controllo che conta):")
if not f.exists():
    P("  [STOP] static_bbg.csv assente")
else:
    S = pd.read_csv(f, index_col=0)
    S.columns = [c.upper() for c in S.columns]
    P(f"  righe: {len(S)} | colonne: {list(S.columns)}")
    C = pd.read_csv(PROC/"static_cct.csv")
    eu = C[C.regime == "CCTeu"]["isin"].tolist()
    got = [i for i in eu if i in S.index]
    P(f"  CCTeu presenti in anagrafica: {len(got)}/{len(eu)}")
    if "FLT_SPREAD" in S.columns:
        sp = pd.to_numeric(S.loc[[i for i in got if i in S.index], "FLT_SPREAD"], errors="coerce")
        P(f"  FLT_SPREAD popolato: {sp.notna().sum()}/{len(got)}")
        if sp.notna().any():
            P(f"    range spread: {sp.min():.1f} - {sp.max():.1f} (bp se >10, altrimenti %)")
            P(f"    [floor] con Euribor6M minimo ~-0.55%, il floor puo' mordere se spread<=55bp:"
              f" {(sp<=55).sum() if sp.max()>10 else (sp<=0.55).sum()} CCTeu esposti")
    else:
        P("  [!] colonna FLT_SPREAD assente: lo spread cedolare CCTeu va recuperato altrove")
        P("      alternativa: leggerlo dalla Security Description, o da 'Coupon' dell'anagrafica")

# ---- verdetto
P("\nVERDETTO:")
need = []
if not (PROC/"px_btp.csv").exists() or not (PROC/"px_bot.csv").exists(): need.append("prezzi BTP/BOT")
if not (PROC/"curves_market.csv").exists(): need.append("curve")
P("  " + ("pronti per 06_curve.py" if not need else f"mancano: {need}"))
save_txt("03b_check.txt", L); print("\n".join(L))
