"""02 - DOWNLOAD BLOOMBERG: anagrafica (bdp) + storici (bdh), con cache incrementale,
CHECKPOINT dopo ogni blocco, throttle fra le richieste e ripartenza senza sprechi.

STRATEGIA CONTRO IL LIMITE GIORNALIERO (code -4002 WORKFLOW_REVIEW_NEEDED):
  1. PRIMO LANCIO con PROVA=True: una micro-richiesta (3 titoli) dice in 10 secondi se il
     limite e' ancora attivo o se il blocco era solo sul volume/ritmo del bulk.
  2. Se la prova passa: PROVA=False e si scarica con THROTTLE e chunk ridotti, un mercato
     alla volta se serve (SOLO_MERCATO), in ordine di dimensione. Ogni blocco scaricato
     viene salvato SUBITO: se il limite scatta a meta', si riparte dai soli mancanti.
  3. Se la prova fallisce: limite attivo -> riprovare domani. Nulla va perso.

VOLUME: per US e UK la curva nominale viene dai file istituzionali (GSW/BoE), quindi dei
nominali si scaricano solo i gemelli del matching (K=2 per lato di ogni scadenza linker:
US 1313 -> ~394, UK 156 -> ~114), anagrafica compresa. Per IT/FR/DE serve lo spettro
completo (la curva la fittiamo noi). FR_CPI usa i nominali di FR: lanciare anche FR.
"""

# ----------------------------------------------------------------- impostazioni
PROVA         = False    # True: SOLO la micro-richiesta di verifica, poi esce.
SOLO_MERCATO  = "IT"    # es. "IT": scarica solo quel mercato. None = tutti, in ordine
                        # IT, FR_CPI, DE, FR, UK, US. Con limite giornaliero stretto:
                        # un mercato (o due) al giorno.
FASI = dict(ANAGRAFICA=False, PREZZI=False, NOMINALI=True, ILS=False, CPI=False)
TRADING       = False   # PX_BID/PX_ASK dei linker: non servono (regressione conclusa); True solo per la versione trading
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
        print(f"-> RISPOSTA OK ({len(df)} righe): il limite NON e' attivo in questo momento.")
        print("   Il blocco di prima era sul volume/ritmo. Metti PROVA=False e rilancia:")
        print("   throttle e chunk ridotti sono gia' impostati; se vuoi andare per gradi,")
        print("   usa SOLO_MERCATO (es. 'IT' oggi, 'FR' domani...).")
    else:
        print("-> ANCORA BLOCCATO: il limite giornaliero e' attivo. Riprova domani;")
        print("   la cache e' incrementale, si riparte esattamente da dove si era.")
    raise SystemExit(0)

bbg.fetch_all(trading=TRADING,
              markets=[SOLO_MERCATO] if SOLO_MERCATO else None,
              phases=FASI)
