"""config - percorsi e costanti del pacchetto convexity.

Struttura dati (organizzata per FONTE; il config legge in place, scrive SOLO in output/):
    ROOT/ (repo)                        [RAW = ROOT/data/raw oppure ROOT/raw: entrambi accettati]
    |-- {data/}raw/
    |     |-- Bloomberg/    bbg_paper2.xlsx (3 fogli: swap | cds | rates & vol)
    |     |                 [opz.] swap_legs_bidask.xlsx o Convexity.xlsx (bid/ask legacy per i costi)
    |     |                 dealer_cds.csv, market_states.csv (creati una-tantum da 00)
    |     |-- Barclays Live/ (o Barclays/)  swaption_vols_batch*.xlsx / data*.xlsx
    |     +-- Fed Board/    feds200628.csv (GSW)
    |-- src/convexity/      QUESTO pacchetto
    +-- output/convexity/   processed/ results/ figures/
"""
from pathlib import Path
import glob as _glob
PKG  = Path(__file__).resolve().parent
ROOT = PKG.parents[1]
RAW  = next((p for p in (ROOT/"data"/"raw", ROOT/"raw") if p.exists()), ROOT/"data"/"raw")
BLOOMBERG = RAW/"Bloomberg"
FED       = RAW/"Fed Board"
BARCLAYS  = next((RAW/n for n in ("Barclays Live","Barclays","BarclaysLive") if (RAW/n).exists()), RAW/"Barclays Live")

def _one(folder, patterns, what):
    for pat in patterns:
        hits=sorted(_glob.glob(str(folder/"**"/pat), recursive=True))
        if hits: return Path(hits[0])
    raise FileNotFoundError(f"{what}: nessun file {patterns} sotto {folder}")
def _all(folder, patterns):
    hits=[]
    for pat in patterns: hits+=_glob.glob(str(folder/"**"/pat), recursive=True)
    return sorted({Path(h) for h in hits})

# --- INPUT ---
BBG   = _one(BLOOMBERG, ["bbg_paper2.xlsx"], "bbg_paper2 (Bloomberg, 3 fogli)")
SHEET_SWAP, SHEET_CDS, SHEET_RV = "swap", "cds", "rates & vol"
GSW   = _one(FED, ["feds200628.csv"], "GSW (Fed Board)")
VOLS  = _all(BARCLAYS, ["swaption_vols*.xlsx","data*.xlsx"])
# NB: nessun file bid/ask. Gli half-spread base sono i livelli HS_BASE (sotto), che il
# modello di costo MOVE-scalato rendera' tempo-varianti. (Convexity.xlsx non serve piu'.)

# --- OUTPUT / derivati una-tantum (00) ---
OUT  = ROOT/"output"/"convexity"
PROC, RES, FIG = OUT/"processed", OUT/"results", OUT/"figures"
for d in (OUT,PROC,RES,FIG): d.mkdir(parents=True, exist_ok=True)
# NB: nessun CSV derivato. CDS e stati si leggono DIRETTAMENTE dai fogli di bbg_paper2
# (utils.load_dealer_cds / load_market_states): un'unica fonte di verita', nessun file intermedio.

DT, SEED = 1/12, 42
APPLY_COSTS = False   # costi SPENTI: quote dealer non ancora disponibili -> risultati GROSS.
                      # Quando arrivano: metti i livelli veri in HS_BASE e APPLY_COSTS=True.
# fly per mercato: gambe (ticker senza suffisso), tenor.
# EUR: default 2/10/25 = costruzione CERTIFICATA (numeri del master v2.4).
# EUR_FLY="2_10_30" attiva la costruzione uniforme (EUSA30 ora disponibile in bbg_paper2):
# da usare SOLO in una ri-baseline dichiarata (cambia tutti i numeri EUR).
EUR_FLY = "2_10_30"
_EUR = {"2_10_25": (("EUSA2","EUSA10","EUSA25"), (2,10,25)),
        "2_10_30": (("EUSA2","EUSA10","EUSA30"), (2,10,30))}
MK = {"USDswap": (("USOSFR2","USOSFR10","USOSFR30"), (2,10,30)),
      "EUR":     _EUR[EUR_FLY],
      "GBP":     (("BPSWS2","BPSWS10","BPSWS30"),    (2,10,30)),
      "JPY":     (("JYSO2","JYSO10","JYSO30"),       (2,10,30))}
IVMAP = {"USDswap":"USD","USTgovt":"USD","EUR":"EUR","GBP":"GBP","JPY":"JPY"}
CDSREGION = {"USDswap":"US","USTgovt":"US","EUR":"EU","GBP":"EU","JPY":"US"}
# CDS dealer canonici (composite del master): match per prefisso sul foglio cds
CDS_CANON_US = ["MS CDS USD SR 5Y","JPMCC CDS USD SR 5Y","BOFA CDS USD SR 5Y"]   # GS escluso: parte 2012 + 29% piatto
CDS_CANON_EU = ["BNP CDS EUR SR 5Y","DB CDS EUR SLA 5Y","BARCLAY CDS EUR SR 5Y","UBS AG CDS EUR SR 5Y"]   # DB: SLA (2001), non SR (2019)
# ---- MODELLO DI COSTO (architettura Paper 1, Tab. A.III trasposta ai tassi) ----
# hs_gamba,t = HS_BASE(gamba) x m(MOVE_t). Tre tier come nel Paper 1 (Main<60/60-100/>=100):
# soglie MOVE ottenute per MATCHING DI PERCENTILE sulle soglie Main (43esimo e 79esimo pct
# sul campione comune 2004-2026 -> MOVE 73 e 107, arrotondate). Moltiplicatori = quelli degli
# strumenti liquidi del Paper 1 (x1 / x1.5 / x2). "Costs conservative precisely in the stress
# periods when arbitrage returns are largest" (Paper 1, Sez. I.B).
MOVE_TIERS = (70.0, 110.0)
COST_MULT  = (1.0, 1.5, 2.0)
# half-spread BASE per gamba (bp) - PLACEHOLDER (in attesa quote dealer). NON impegnare finche' APPLY_COSTS=False:
HS_BASE = {"USOSFR2":1.29,"USOSFR10":1.96,"USOSFR30":2.50,
           "EUSA2":0.87,"EUSA10":0.89,"EUSA25":1.05,"EUSA30":1.05,
           "BPSWS2":1.01,"BPSWS10":1.16,"BPSWS30":1.77,
           "JYSO2":1.23,"JYSO10":2.97,"JYSO30":3.48}
EPISODES = ["2008-12-31","2011-11-30","2020-04-30","2022-10-31"]
EXPECTED_SIGN = {"2008-12-31":{"EUR":+1,"USTgovt":+1},
                 "2011-11-30":{"USDswap":+1,"EUR":+1,"GBP":+1,"JPY":+1,"USTgovt":+1},
                 "2022-10-31":{"USDswap":-1,"EUR":0,"GBP":-1,"JPY":+1,"USTgovt":-1}}
