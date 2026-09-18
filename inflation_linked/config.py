"""config - anagrafica dei mercati e parametri.

PRINCIPIO. L'unica cosa che non si automatizza e' l'ENUMERAZIONE dell'universo: la lista degli
strumenti mai emessi, scaduti compresi, richiede una ricerca sul terminale. Tutto il resto
(cedole, scadenze, settle date, base CPI, prezzi, YTM, CPI storico, curve ILS) lo prende il
codice via bdp/bdh.

DA SCARICARE: data/raw/Bloomberg/Govt_bonds.xlsx (fogli 'IL' e 'Nominal'), letto da
bbg.build_universe(). SRCH -> issuer/country, Security Type = Government, Coupon Type =
Index Linked oppure Fixed, con "Include Matured" ATTIVO. Export di ISIN e ID_BB_UNIQUE.
Per la Germania includere anche gli OBL fra i nominali: i Bund-ei a 5 anni non hanno gemelli
fra i soli DBR.

Queste liste sostituiscono le vecchie list_isin_* e isin_to_id_number_dict del progetto
originale, che erano due strutture da sincronizzare a mano e avevano buchi (list_isin_dbri
conteneva 3 ISIN contro gli 8 Bund-ei emessi).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import holidays
import pandas as pd

# --------------------------------------------------------------------------- percorsi
ROOT       = Path(__file__).resolve().parent          # .../src/inflation_linked (il codice)
PROJECT    = ROOT.parent.parent                       # .../THESIS (la root del progetto)
DATA       = PROJECT / "data"
CACHE    = DATA / "cache"          # parquet: prezzi, YTM, curve ILS, CPI
OUT      = ROOT / "output"
# i dati esterni stanno in data/raw/<fonte>/ (cfr. bbg.UNIVERSE_FILE); data/universe
# era un percorso legacy, usato solo da load_universe(), rimossa perche' mai chiamata.
for _p in (DATA, CACHE, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- tenor ILS
# invariato dal progetto originale
YEARS_FORWARD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]

# --------------------------------------------------------------------------- calendari
# NB: il progetto originale aggiunge a mano i Venerdi Santo al calendario italiano
# (further_holidays, config.py righe 18-38). Quella lista va riportata QUI VERBATIM:
# senza di essa le business date italiane cambiano e il test di regressione non passa.
FURTHER_IT_HOLIDAYS: list[date] = [
    # date(2000, 4, 21), date(2001, 4, 13), date(2002, 3, 29), ...  <-- incollare dall'originale
]

def _calendar(country: str) -> dict:
    cal = {"IT": holidays.Italy, "FR": holidays.France, "DE": holidays.Germany,
           "UK": holidays.UK, "US": holidays.US, "ES": holidays.Spain}[country]()
    if country == "IT":
        for d in FURTHER_IT_HOLIDAYS:
            cal.append(d)
    return {d: n for d, n in cal.items()}

# --------------------------------------------------------------------------- mercati
@dataclass(frozen=True)
class Market:
    code: str
    name: str
    ccy: str
    cpi: str              # ticker Bloomberg dell'indice dei prezzi
    ils: str              # radice ticker della curva zero-coupon inflation swap
    ils_source: str       # pricing source della curva (BGN / ICPL)
    index_lag: int        # mesi di lag dell'indice di riferimento (3 = euro/US, 2 = ramo UKRPI)
    px_source: str        # pricing source dei prezzi storici (bdh)
    ref_source: str       # pricing source per l'anagrafica (bdp)
    country: str          # calendario
    notes: str = ""

    @property
    def holidays(self) -> dict:
        return _calendar(self.country)

    def ils_ticker(self, year: int) -> str:
        return f"{self.ils}{year} {self.ils_source} Curncy"

    def px_ticker(self, bb_id: str) -> str:
        return f"{bb_id}@{self.px_source} Corp"

    def ref_ticker(self, bb_id: str) -> str:
        return f"{bb_id}@{self.ref_source} Corp"


# ils_source e index_lag riproducono ticker_dict e i rami per-mercato di cpi.py:
#   ticker_dict = {'ITCPIUNR':'ILSWI', 'CPTFEMU':'EUSWI', 'FRCPXTOB':'FRSWI',
#                  'CPURNSA':'USSWIT', 'UKRPI':'BPSWIT'}
# ATTENZIONE UK: in cpi.py il ramo UKRPI ancora la proiezione a pprev_eom (t-2) invece di
# ppprev_eom (t-3) come gli altri, e usa months_first_year/index_adj diversi. index_lag=2 qui
# riproduce quel ramo. Va verificato contro le convenzioni dei gilt prima di fidarsi.
MARKETS: dict[str, Market] = {
    "IT": Market("IT", "Italy BTP\u20aci", "EUR", "CPTFEMU", "EUSWI", "BGN", 3,
                 "CBBT", "MILA", "IT",
                 "HICP ex-tobacco. Nominali: BTPS."),
    "FR": Market("FR", "France OAT\u20aci", "EUR", "CPTFEMU", "EUSWI", "BGN", 3,
                 "CBBT", "CBBT", "FR",
                 "HICP ex-tobacco: stesso indice e stessa curva dell'Italia, "
                 "quindi confrontabile. Nominali: FRTR."),
    "FR_CPI": Market("FR_CPI", "France OATi", "EUR", "FRCPXTOB", "FRSWI", "BGN", 3,
                     "CBBT", "CBBT", "FR",
                     "CPI francese ex-tabacco. Storia piu' lunga dell'OAT\u20aci "
                     "(1998 vs 2001): estensione di robustezza, non misura primaria."),
    "DE": Market("DE", "Germany Bund\u20aci", "EUR", "CPTFEMU", "EUSWI", "BGN", 3,
                 "CBBT", "CBBT", "DE",
                 "8 emissioni in tutto: mercato sottile. Nominali: DBR + OBL "
                 "(i Bund\u20aci a 5a non hanno gemelli fra i soli DBR)."),
    "UK": Market("UK", "UK index-linked gilts", "GBP", "UKRPI", "BPSWIT", "BGN", 2,
                 "CBBT", "CBBT", "UK",
                 "RPI. I linker pre-2005 hanno lag di indicizzazione a 8 MESI, i nuovi a 3: "
                 "distinzione non gestita dal codice originale, da aggiungere per la storia "
                 "lunga (FLL 2014 verificano che i risultati non dipendono da quale)."),
    "US": Market("US", "US TIPS", "USD", "CPURNSA", "USSWIT", "BGN", 3,
                 "CBBT", "CBBT", "US",
                 "CPI-U NSA. Nominali: T. Verificare TIPS_nominals.xlsx nel progetto TIPS "
                 "prima di riscaricare."),
}

EURO_HICP = ["IT", "FR", "DE"]   # stesso indice e stessa curva: cross-section confrontabile

# --------------------------------------------------------------------------- curva DE
# Curva nominale tedesca per il C-esatto: 'bundesbank' = parametri Svensson ufficiali
# (gold standard: istituzionale, replicabile, dal 1997; risparmia ~257 nominali su
# Bloomberg) oppure 'fit' = NSS nostro sui DBR/OBL/BKO. Per IT e FR non esiste una curva
# ufficiale pubblica giornaliera: il fit Svensson proprio con obiettivo GSW-equivalente
# E' lo standard di letteratura (BIS Paper 25), non un'approssimazione. Le curve fittate
# Bloomberg (BVAL/YCGT) sono proprietarie e non replicabili: mai come misura primaria.
DE_NOMINAL_CURVE = "bundesbank"

# --------------------------------------------------------------------------- prezzi
# Il progetto originale usa PX_ASK. Per la misura teorica si usa il MID; l'ASK resta per la
# versione tradabile, coerentemente con la divisione teoria/trading.
PX_FIELD_THEORY  = "PX_MID"
PX_FIELD_TRADING = "PX_ASK"

# --------------------------------------------------------------------------- matching
# Due misure, entrambe prodotte e riportate insieme:
#   "interp"  -> interpolazione lineare fra i due nominali adiacenti pesata per la distanza
#                in giorni dalla scadenza del linker. Misura TEORICA: toglie il rumore di
#                mismatch, indispensabile sul tratto lungo dove la griglia dei nominali si
#                dirada a un bond all'anno (gap mediano 365gg oltre il 2040).
#   "nearest" -> singolo nominale con scadenza piu' vicina. Misura TRADABILE ed e' l'oggetto
#                di FLL, confrontabile con la loro Table III.
# Disciplina: mostrare che le due coincidono dove la griglia e' densa (2015-2030, gap mediano
# 31gg), cosi' l'interpolazione non puo' fabbricare il risultato.
MATCHING = ("interp", "nearest")
MAX_MISMATCH_DAYS = 183      # oltre questo la coppia viene scartata e dichiarata

# --------------------------------------------------------------------------- regressione
# Vincolo di sicurezza della riscrittura: il nuovo codice deve riprodurre la base BTP\u20aci
# esistente al centesimo di bp sui 22 ISIN del campione 2015-2026. Finche' questo non passa,
# non si tocca nulla d'altro.
REGRESSION_TARGET = DATA / "btpei_basis.xlsx"
REGRESSION_TOL_BP = 0.01

