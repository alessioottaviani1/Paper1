"""
================================================================================
config.py — Paper "The CCT-BTP Basis" (base tra floater e fisso, stesso emittente)
================================================================================
Convenzioni identiche al package tips_treasury: percorsi da ROOT, script numerati,
parametri centralizzati qui. Eseguire dalla cartella del package.

DISEGNO DEL CAMPIONE (deciso e congelato)
  primario   1999-01-01 -> oggi : era euro. Euribor nasce il 30-12-1998 (primo
             fissaggio 04-01-1999) e il mercato IRS in EUR e' liquido: tutte e tre
             le misure di base sono costruibili.
  estensione 1995-01-01 -> 1998-12-31 : solo CCT indicizzati ai BOT, replica
             SWAP-FREE (la gamba variabile e' il rendimento BOT + spread, quindi si
             replica con BOT osservabili). Regime dichiarato a parte: denominazione
             in lire e trade di convergenza EMU (rendimenti da ~12% a ~4%).
  Prima del 1995 non si va: senza BOT non esiste gamba variabile replicabile.

DUE REGIMI DI INDICIZZAZIONE, MAI POOLED SENZA TEST
  CCT-BOT  (< 2010) : indicizzato al debito a breve dello STESSO emittente. Il
                      credito italiano e' dentro l'indice: quasi un auto-hedge.
  CCTeu    (>=2010) : indicizzato a Euribor 6M, cioe' credito bancario europeo.
                      Mismatch sovrano-vs-banche: oggetto economico diverso.
  Il passaggio del 2010 e' un ESPERIMENTO NATURALE (cambia il tasso di riferimento
  tenendo fissi emittente, investitori e microstruttura) ed e' testabile come break.
================================================================================
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- percorsi --------------------------------------
ROOT   = Path(__file__).resolve().parents[2]           # .../THESIS
RAW    = ROOT / "data" / "raw"
BBG    = RAW / "Bloomberg"
PROC   = ROOT / "data" / "processed" / "cct"
RES    = ROOT / "results" / "cct"
FIG    = RES / "figures"
CACHE  = PROC / "cache"                                # cache dei pull Bloomberg
TABDIR = ROOT / "paper" / "tables_cct"

FILE_STATIC = BBG / "BTP_CCT.xlsx"                     # anagrafica: fogli CCTS, BOTS, BTPS
for d in (PROC, RES, FIG, CACHE, TABDIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------- campione --------------------------------------
START_PRIMARY   = "1999-01-01"
START_EXTENDED  = "1995-01-01"
END_SAMPLE      = "2026-08-12"
CCTEU_CUTOFF    = "2010-01-01"    # primo CCTeu: giugno 2010. Da verificare su prospetto MEF.
EURIBOR_START   = "1999-01-04"    # primo fissaggio Euribor

# ----------------------------- appaiamento -----------------------------------
# Regola ISIN-vs-ISIN: fra i BTP entro MAX_MISMATCH_D dalla scadenza del CCT,
# tieni quelli con copertura >= MIN_COVERAGE della finestra utile del CCT, poi
# minimo mismatch; a parita' preferisci il BTP che scade PRIMA, poi l'on-the-run.
MAX_MISMATCH_D  = 92        # +- 3 mesi. I CCT scadono 15/1-4-7-10, i BTP 1/8 di vari mesi:
                            # una soglia a 31gg come nei TIPS scarterebbe troppo.
MIN_COVERAGE    = 0.90      # il BTP deve vivere >=90% della finestra utile del CCT
MIN_MONTHS_PAIR = 12        # coppie con meno di 12 mesi osservabili non entrano

# ----------------------------- curva -----------------------------------------
# Svensson fittato ai PREZZI (convenzione GSW), pesi inversi alla duration.
# I BOT ancorano il tratto 0-1y, che con i soli BTP sarebbe mal identificato.
CURVE_MODEL     = "svensson"
CURVE_WEIGHT    = "inv_duration"
CURVE_MIN_BONDS = 8
CURVE_MAX_TAU   = 30.0
CURVE_EXCL_TAU  = 0.08      # esclude titoli sotto ~1 mese (prezzi rumorosi)

# ----------------------------- engine ----------------------------------------
CPN_FREQ        = 2                # semestrale per tutti e tre gli strumenti

# --- CONVENZIONI: DIVERSE FRA I TRE STRUMENTI (fonte: schede MEF) ---------------
# ATTENZIONE: BTP e CCT usano ACT/ACT, il CCTeu usa ACT/360. Non e' un dettaglio di
# arrotondamento: 365/360 = 1.39% di differenza sul rateo, che su una cedola del 3%
# vale ~4bp l'anno. Va gestito nella replica, non assorbito nel residuo.
DC_BTP          = "ACT/ACT-ICMA"   # "giorni effettivi/giorni effettivi"
DC_CCT          = "ACT/ACT"        # idem, scheda CCT
DC_CCTEU        = "ACT/360"        # scheda CCTeu: "giorni effettivi/360", Mod.Following unadjusted

# --- INDICIZZAZIONE CCT (BOT-indexed) ------------------------------------------
# Regola post-1995: cedola_semestrale = 0.5 * rendimento_BOT6m_ultima_asta + spread,
# arrotondata ai 5 centesimi piu' vicini. NB: lo spread si somma DOPO aver dimezzato,
# quindi in termini annui pesa il doppio del suo valore nominale.
CCT_ROUND_STEP  = 0.05             # "arrotondato ai cinque centesimi piu' vicini"
CCT_SPREAD_BP   = [                # (valido per emissioni fino a, spread in punti percentuali)
    ("1993-08-01", 0.50),          # 50bp fino all'emissione 1-8-1993/00 inclusa
    ("1996-09-01", 0.30),          # 30bp per emissioni dal 1-10-1993 al 1-9-1996
    ("2099-12-31", 0.15),          # 15bp dal CCT 1-11-1996/03 in poi
]
CCT_RULE_CHANGE = "1995-01-01"     # prima: media aritmetica dei BOT ANNUALI collocati nel
                                   # bimestre che precede di un mese il godimento, cedola
                                   # pagata 8 mesi dopo la determinazione. Regime diverso:
                                   # i CCT emessi 1988-94 (scadenza 1995-2001) lo usano ancora.

# --- INDICIZZAZIONE CCTeu ------------------------------------------------------
EURIBOR_LAG_BD  = 2                # fissaggio 2 gg lavorativi prima del PRIMO giorno di godimento
EURIBOR_ROUND   = 0.001            # Euribor arrotondato al 3o decimale prima di sommare lo spread

# --- FLOOR A ZERO --------------------------------------------------------------
# Circolare MEF n. 5619 del 21 marzo 2016: se il parametro e' negativo al punto da erodere
# e superare lo spread, la cedola e' posta pari a zero. Vale per CCT e CCTeu.
# L'Euribor 6M entra in negativo prima della circolare (nov-2015): la finestra di esposizione
# e' quindi piu' ampia della vigenza formale, e prima di marzo-2016 il trattamento e' ambiguo.
FLOOR_CIRCULAR  = "2016-03-21"
FLOOR_FLAG_FROM = "2015-11-01"     # Euribor 6M sotto zero
FLOOR_FLAG_TO   = "2022-07-31"     # rientro sopra zero
FLOOR_AS_OPTION = False            # v1: floor NON prezzato, solo flag. Ganci pronti in 05.
PRICE_FIELD     = "PX_MID"         # solo mid: bid/ask non affidabili su questi mercati.
                                   # I costi entrano come layer separato, non nella misura.

# ----------------------------- Bloomberg -------------------------------------
# Suffisso del ticker: da verificare con 00_smoketest prima del download massivo.
# Il codice precedente usava '@MILA Corp' per l'anagrafica e '@CBBT Corp' per i prezzi.
# CBBT (Composite Bloomberg Bond Trader) e' di norma la fonte migliore sui govie euro;
# BGN (Bloomberg Generic) e' l'alternativa. MILA e' il listino di Borsa Italiana (retail).
PX_SUFFIX_CANDIDATES = ["@CBBT Corp", "@BGN Corp", " Corp", "@MILA Corp"]
PX_SUFFIX     = "@CBBT Corp"     # sovrascritto dopo lo smoketest
PULL_BLOCK    = 150              # titoli per sessione: sotto il limite giornaliero
YLD_FIELD_BOT = "YLD_YTM_MID"    # i BOT sono zero-coupon: lo yield e' piu' pulito del prezzo

# --- BOT: convenzioni e derivazione del rendimento d'asta -----------------------
# Scheda MEF BOT: zero coupon, rimborso alla pari, "giorni effettivi/360 per il calcolo
# del rendimento". Il rendimento lordo semplice annuo che entra nell'indicizzazione CCT
# e' quindi gia' ACT/360: NON va convertito in ACT/ACT.
#   y_semplice_annuo = (100 - P)/P * 360/n
# Dal aprile 2009 le offerte d'asta sono espresse in RENDIMENTO; prima in PREZZO.
DC_BOT          = "ACT/360"
BOT_6M_MIN_D    = 160        # finestra per identificare i BOT "semestrali" fra i flessibili
BOT_6M_MAX_D    = 200
BOT_AUCTION_SRC = "derived"  # "derived" = ricavato dal prezzo del BOT alla data di emissione;
                             # "official" = serie ufficiale dt.tesoro.it (piu' precisa, da caricare
                             # in PROC/bot_auction_6m_official.csv se disponibile).

# --- STRIPS: strumento di chiusura per la replica esatta ------------------------
# Scheda MEF BTP: programma STRIPS dal 1998, disponibile per BTP di durata >= 5 anni.
# E' l'analogo italiano degli STRIPS usati da Fleckenstein-Longstaff per chiudere i flussi
# residui. Prima del 1998 non esistono: nel campione esteso (1995-97) il residuo va chiuso
# con BOT/zero-coupon sintetici dalla curva, e la cosa va dichiarata.
STRIPS_FROM     = "1998-01-01"
STRIPS_MIN_TENOR= 5

# --- CURVE DI MERCATO: griglia completa, ticker generati dal codice ------------
# Stesso schema del progetto inflation_linked (YEARS_FORWARD in config, ticker costruiti
# in codice, nessun Excel da fornire a mano).
#
# Perche' la griglia deve essere fitta: i CCT scadono il 15 di gen/apr/lug/ott di anni
# qualsiasi, quindi lo swap di replica ha scadenza arbitraria. Con soli 4 punti (2/5/10/30)
# l'interpolazione sul tratto 3-7 anni -- dove sta la massa dei CCT -- sarebbe grossolana.
YEARS_SWAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]

# EUSA<n> = IRS EUR contro Euribor 6M (gamba variabile del CCTeu: nessun basis swap).
# EESWE<n> = OIS EUR su ESTR: serve per l'ATTUALIZZAZIONE nel regime post-collateral.
def swap_tickers():   return {f"EUSA{n} Curncy":  f"irs{n}y" for n in YEARS_SWAP}
def ois_tickers():    return {f"EESWE{n} Curncy": f"ois{n}y" for n in YEARS_SWAP}
def index_tickers():  return {"EUR006M Index": "euribor6m", "EUR003M Index": "euribor3m",
                              "EONIA Index": "eonia", "ESTRON Index": "estr"}

# --- REGIME DI ATTUALIZZAZIONE -------------------------------------------------
# Fino a ~2010 il mercato swap scontava sulla curva Euribor; dal 2010-11 e' passato
# all'OIS discounting (garanzia in contanti remunerata al tasso overnight). Su un campione
# 1999-2026 ci si passa attraverso: due regimi dichiarati, non una convenzione unica.
OIS_DISCOUNTING_FROM = "2011-01-01"
