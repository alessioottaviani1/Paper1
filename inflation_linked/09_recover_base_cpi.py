"""09 - RECUPERO del BASE_CPI mancante, per derivazione AUTOVALIDATA.

Offline. Non scrive nulla finche' non glielo dici (SCRIVI=False di default).

IDEA. Il coefficiente di indicizzazione e' per definizione 1 alla data base del titolo,
quindi  base_cpi = indice di riferimento alla data base.  L'indice di riferimento lo sa
gia' calcolare basis.mef_reference, con il lag e la regola di interpolazione giusti per
il segmento (gilt old-style: lag 8, NESSUNA interpolazione; tutto il resto: lag 3
interpolato). Resta da stabilire QUALE campo data sia la data base, e la documentazione
Bloomberg non e' univoca (START_ACC_DT / ISSUE_DT / FIRST_SETTLE_DT / FIRST_CPN_DT).

Non lo assumiamo: lo CALIBRIAMO. Per ogni candidato si ricostruisce il base_cpi dei
titoli che CE L'HANNO e si guarda l'errore. Il candidato che riproduce i noti entro
TOL_REL e' quello giusto, e solo allora lo si applica ai mancanti. Se nessun candidato
riproduce i noti, lo script NON propone nulla: vuol dire che la regola e' un'altra e il
base va preso dal prospetto DMO/AFT/Finanzagentur a mano.
"""
import numpy as np
import pandas as pd
import bbg
from basis import mef_reference
from config import MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO   = "UK"
TOL_REL   = 1e-4      # errore relativo massimo per dire "riprodotto"
SCRIVI    = False     # True: riscrive base_cpi_final in ref_linker.parquet per i mancanti
CANDIDATI = ["START_ACC_DT", "ISSUE_DT", "FIRST_SETTLE_DT", "FIRST_CPN_DT"]

# ----------------------------------------------------------------- esecuzione
m = MARKETS[MERCATO]
ref = bbg.load("ref_linker")
ref = ref[ref["mkt"] == MERCATO].copy()
cpi = bbg.load(f"cpi_{m.cpi}").iloc[:, 0]
cpi.index = pd.to_datetime(cpi.index)

# segmento -> (lag, interpolazione). Il gilt old-style e' l'unico non interpolato.
def _rule(isin):
    seg = str(ref.at[isin, "segment"]) if "segment" in ref.columns else ""
    if MERCATO == "UK" and seg == "old":
        return 8, False
    return m.index_lag, True

noti = ref[ref["base_cpi_final"].notna() & (ref["base_cpi_final"] > 0)]
mancanti = ref[ref["base_cpi_final"].isna() | (ref["base_cpi_final"] <= 0)]
print(f"=== {MERCATO}: {len(noti)} con base valida, {len(mancanti)} da recuperare ===\n")
if not len(mancanti):
    raise SystemExit("niente da fare")

def _derive(isin, campo):
    d = ref.at[isin, campo] if campo in ref.columns else None
    if pd.isna(d):
        return np.nan
    lag, interp = _rule(isin)
    try:
        return mef_reference(cpi, pd.Timestamp(d).date(), lag=lag, interpolate=interp)
    except Exception:
        return np.nan

# --- calibrazione sui noti -------------------------------------------------------
print("--- calibrazione: quale campo data riproduce i base_cpi NOTI? ---")
print(f"{'campo':<18}{'testati':>9}{'riprodotti':>12}{'err.mediano':>14}")
best, best_score = None, -1
for campo in CANDIDATI:
    der = pd.Series({i: _derive(i, campo) for i in noti.index})
    err = (der / noti["base_cpi_final"] - 1).abs().dropna()
    ok = int((err < TOL_REL).sum())
    print(f"{campo:<18}{len(err):>9}{ok:>12}{(err.median() if len(err) else np.nan):>14.2e}")
    if len(err) and ok / len(err) > best_score:
        best, best_score = campo, ok / len(err)

print(f"\nmigliore: {best}  ({best_score:.1%} riprodotti entro {TOL_REL:.0e})")
if best_score < 0.95:
    raise SystemExit(
        "\nNESSUN campo riproduce i noti in modo affidabile.\n"
        "-> la regola della data base e' diversa da quelle testate: prendi il base dal\n"
        "   prospetto (DMO per i gilt) e inseriscilo a mano in ref_linker.parquet.\n"
        "   NON applicare una derivazione che non passa la calibrazione.")

# --- errore per segmento, per non nascondere un segmento rotto dentro la media ----
if "segment" in ref.columns:
    der_all = pd.Series({i: _derive(i, best) for i in noti.index})
    e = (der_all / noti["base_cpi_final"] - 1).abs()
    print("\nerrore per segmento:")
    for seg, g in e.groupby(noti["segment"]):
        print(f"  {str(seg):<10} n={len(g):<4} riprodotti {int((g < TOL_REL).sum())}/{len(g)}"
              f"   err mediano {g.median():.2e}")

# --- applicazione ai mancanti ----------------------------------------------------
print(f"\n--- derivazione dei {len(mancanti)} mancanti (campo {best}) ---")
print(f"{'ISIN':<16}{'nome':<28}{'segmento':<10}{'data base':<12}{'base derivata':>14}")
prop = {}
for isin in mancanti.index:
    v = _derive(isin, best)
    d = ref.at[isin, best] if best in ref.columns else pd.NaT
    seg = str(ref.at[isin, "segment"]) if "segment" in ref.columns else "-"
    nome = str(ref.at[isin, "SECURITY_NAME"]) if "SECURITY_NAME" in ref.columns else ""
    print(f"{isin:<16}{nome[:27]:<28}{seg:<10}"
          f"{(f'{pd.Timestamp(d):%Y-%m-%d}' if pd.notna(d) else 'n/d'):<12}"
          f"{v:>14.5f}" if pd.notna(v) else f"{isin:<16}{nome[:27]:<28}{seg:<10}{'n/d':<12}{'FALLITA':>14}")
    if pd.notna(v):
        prop[isin] = v

print(f"\nderivate {len(prop)}/{len(mancanti)}")
print("VERIFICA A MANO almeno una contro il prospetto DMO prima di scrivere.")

if SCRIVI and prop:
    full = bbg.load("ref_linker")
    for isin, v in prop.items():
        full.at[isin, "base_cpi_final"] = v
    full.to_parquet(bbg.CACHE / "ref_linker.parquet")
    print(f"\nSCRITTE {len(prop)} base_cpi_final in ref_linker.parquet")
    print("-> rilancia 04_basis_markets.py per rigenerare i pannelli")
else:
    print("\n(SCRIVI=False: nulla e' stato modificato. Metti SCRIVI=True per applicare.)")
