"""Verifica del CI calcolato contro il MEF UFFICIALE, con precisione al quinto decimale.
Il CI storico dipende solo dagli indici Eurostat gia' pubblicati (nessuna proiezione):
CI(d,m) = IR(d,m) / base_cpi_final, con IR interpolato (formula MEF) e troncato/arrotondato.
Testa in particolare i bond SCADUTI (dove il ribasamento IAPC morde).
Lancia dalla root:  python .\src\inflation_linked\_verifica_ci.py
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import bbg
from basis import mef_reference, ci_round

cpi = bbg.load("cpi_CPTFEMU"); cpi.index = pd.to_datetime(cpi.index)
series = cpi.iloc[:, 0]
series.index = series.index + pd.offsets.MonthEnd(0)
ref = bbg.load("ref_linker")

def ci_nuovo(isin: str, d: str) -> float:
    base = float(ref.loc[isin, "base_cpi_final"])   # gia' ribasato in enrich_reference
    ir = mef_reference(series, pd.Timestamp(d).date(), lag=3, interpolate=True)
    return ci_round(ir / base, floor=False, mode="mef")

# ============================================================
# INSERISCI I CI UFFICIALI MEF: (ISIN, 'YYYY-MM-DD', CI_ufficiale)
# Metti None se non l'hai ancora: stampa solo il calcolato.
# Scegli DATE su cui hai il coefficiente ufficiale dal sito Debito Pubblico.
# Includi BOND SCADUTI (i 3 ribasati) e bond vivi, per copertura.
# ============================================================
CASI = [
    # --- i 3 bond RIBASATI (scaduti, caso critico) ---
    ("IT0005004426", "2022-07-31", 1.16059),   # BTP€i 2.35 09/15/24 (gia' verificato: deve dare 1.25984)
    ("IT0004380546", "2019-07-01", None),       # BTP€i 2.35 09/15/19 (scaduto 2019)
    ("IT0004243512", "2020-12-18", 1.19240),       # BTP€i 2.6  09/15/23 (scaduto 2023)
    ("IT0004243512", "2023-09-03", 1.40272),       # BTP€i 2.6  09/15/23 (scaduto 2023)
    ("IT0004243512", "2021-05-01", 1.19760),       # BTP€i 2.6  09/15/23 (scaduto 2023)
    # --- bond vivi (controllo che restino giusti) ---
    ("IT0005436701", "2022-07-31", 1.10735),       # BTP€i 0.4 05/15/30 (vivo)
    ("IT0005329344", "2022-07-31", 1.13708),
    ("IT0005415416", "2022-07-31", 1.10156),
    ("IT0005482994", "2022-07-31", 1.07354),
    ("IT0004604671", "2021-07-29", 1.15640),
    ("IT0003532915", "2008-02-29", 1.10443),
    ("IT0003625909", "2008-02-29", 1.10443),
    ("IT0003805998", "2008-02-29", 1.08182),
    ("IT0003745541", "2008-02-29", 1.08182),
    ("IT0004085210", "2008-02-29", 1.05176),
    ("IT0004216351", "2008-02-29", 1.03318),
    ("IT0004243512", "2008-02-29", 1.03318),
    # --- aggiungi altri scaduti che trovi sul MEF, es. i piu' vecchi ---
    # ("IT0003625909", "2014-07-01", None),     # BTP€i 2.15 09/15/14 (scaduto 2014, PRE-ribasamento 2016!)
]

print(f"{'ISIN':16s} {'data':12s} {'CI calcolato':>14s} {'CI MEF':>12s} {'diff':>12s}  esito")
print("-" * 78)
for isin, d, mef in CASI:
    try:
        c = ci_nuovo(isin, d)
    except Exception as e:
        print(f"{isin:16s} {d:12s}  ERRORE: {e}"); continue
    if mef is None:
        print(f"{isin:16s} {d:12s} {c:14.5f} {'(cerca MEF)':>12s} {'-':>12s}")
    else:
        diff = c - mef
        esito = "OK" if abs(diff) < 1e-5 else ("QUASI" if abs(diff) < 1e-4 else "DIVERGE")
        print(f"{isin:16s} {d:12s} {c:14.5f} {mef:12.5f} {diff:12.2e}  {esito}")

print()
print("base_cpi_final (dopo ribasamento) dei bond testati:")
for isin in dict.fromkeys(c[0] for c in CASI):
    try:
        print(f"  {isin}: {float(ref.loc[isin,'base_cpi_final']):.5f}  "
              f"({str(ref.loc[isin].get('SECURITY_NAME',''))[:24]})")
    except Exception as e:
        print(f"  {isin}: {e}")
