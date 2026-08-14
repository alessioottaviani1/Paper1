"""02 - DOWNLOAD BLOOMBERG: anagrafica (bdp) + storici (bdh), con cache incrementale,
CHECKPOINT dopo ogni blocco, throttle fra le richieste e ripartenza senza sprechi.

STRATEGIA CONTRO IL LIMITE GIORNALIERO (code -4002 WORKFLOW_REVIEW_NEEDED):
  - PROVA=True: una micro-richiesta (3 titoli) dice se il limite e' attivo, poi esce.
  - PROVA=False: scarica sul serio, con throttle e chunk ridotti, un mercato alla volta
    se SOLO_MERCATO e' impostato. Ogni blocco scaricato viene salvato SUBITO: se il
    limite scatta a meta', si riparte dai soli mancanti.

VOLUME: per US/UK/DE la curva nominale viene da fonte istituzionale (GSW/BoE/Bundesbank),
quindi dei nominali si scaricano solo i gemelli del matching. Per IT/FR lo spettro pieno.
FR_CPI usa i nominali di FR: lanciare anche FR.
"""

# ----------------------------------------------------------------- impostazioni
PROVA         = False   # True: solo micro-richiesta di verifica, poi esce.
SOLO_MERCATO  = "IT"    # es. "IT": scarica solo quel mercato. None = tutti, in ordine
                        # IT, FR_CPI, DE, FR, UK, US. Con limite giornaliero: uno al giorno.
FASI = dict(ANAGRAFICA=True, PREZZI=False, NOMINALI=False, ILS=False, CPI=False)
TRADING       = True    # PX_BID/PX_ASK dei linker (necessario alla regressione: ASK)
THROTTLE_SEC  = 0.5     # pausa fra richieste API
CHUNK_BDP     = 40      # titoli per richiesta anagrafica (default libreria: 80)
CHUNK_BDH     = 10      # titoli per richiesta storici (default libreria: 25)

# ----------------------------------------------------------------- esecuzione
try:
    import xbbg  # noqa: F401
except ImportError:
    raise SystemExit("xbbg non disponibile: questo script va lanciato sul terminale Bloomberg.")

import bbg

bbg.THROTTLE_SEC = THROTTLE_SEC
bbg.CHUNK_BDP = CHUNK_BDP
bbg.CHUNK_BDH = CHUNK_BDH

if PROVA:
    uni = bbg.build_universe(save=False)
    tre = uni[uni["incl"] & uni["kind"].eq("linker") & uni["mkt"].eq("IT")].head(3)
    ticks = [bbg._ref_ticker(r) for _, r in tre.iterrows()]
    print(f"PROVA: bdp su {ticks}")
    df = bbg._bdp(ticks, ["SECURITY_NAME", "CPN"])
    if len(df):
        print(f"-> RISPOSTA OK ({len(df)} righe): il limite NON e' attivo. Metti PROVA=False.")
    else:
        print("-> ANCORA BLOCCATO: riprova domani; la cache e' incrementale, riparte da dove era.")
    raise SystemExit(0)

bbg.fetch_all(trading=TRADING,
              markets=[SOLO_MERCATO] if SOLO_MERCATO else None,
              phases=FASI)