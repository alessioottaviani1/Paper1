"""04 - BASI PER MERCATO: per ogni mercato in MERCATI produce i pannelli wide (date x ISIN):
  basis_{mkt}_nearest   metodo 1, FLL/Kita-Tortorice matched-maturity senza STRIPS:
                        singolo gemello nominale <=183gg, mismatch dichiarato in giorni
  basis_{mkt}_cexact    metodo 2, ISIN vs curva: IRR osservato - IRR sintetico sulla
                        curva zero nominale (mismatch zero per costruzione)
  totalytm, ynom_nearest, mismatch (+ breakeven / floorval a richiesta)

Convenzione yield ANNUAL (default): UK/US semi->annuale per confrontabilita' con l'IRR
del linker; euro invariato. Il floor di deflazione e' per-strumento (MARKET_FLOOR:
gilt = nessun floor). UK include ENTRAMBI gli stili salvo UK_SOLO_NEW.
Il fit NSS euro stampa la sua diagnostica (rmse_bp): guardala prima di fidarti del C-esatto.

Prima di lanciare: 01, 02, 03."""

# ----------------------------------------------------------------- impostazioni
MERCATI          = ["ES", "IT", "DE", "FR"]
DATA_DA          = None
CON_BREAKEVEN    = False      # calcola anche RealYtM e breakeven (circa raddoppia i tempi)
CON_VALORE_FLOOR = False      # valore del floor in bp (port Black dell'originale)
UK_SOLO_NEW      = False      # True per riprodurre il comportamento pre-old-style
PUB_LAG_DAYS     = None       # None = originale; es. 20 = modalita' onesta (paper)
PRICE_FIELD      = "px_mid"   # 'px_ask' per la versione trading
ESCLUDI_CODA     = False      # True: esclude l'ultimo anno di vita dei linker
MIN_VITA_GG      = 365        # vita residua minima. NON la soglia GSW dei 3 mesi: quella
                              # serve a fittare una curva, dove basta che il titolo abbia
                              # un prezzo sensato. Lo z-spread scala come 1/duration, e a
                              # 3 mesi ha ~40x la sensibilita' che ha a 10 anni: sotto
                              # l'anno sta il 52-66% degli estremi contro il 3-5% delle
                              # osservazioni (rapporto 13-32x su tutti e quattro i
                              # mercati). Da dichiarare come criterio di campionamento,
                              # con robustezza a 90/180/365.
YTM_CONVENTION   = "annual"   # annual = confronto omogeneo (UK/US semi->annuale); 'local' solo diagnostica

# ----------------------------------------------------------------- esecuzione
import pipeline

for mkt in MERCATI:
    print(f"=== {mkt} ===")
    pipeline.build_market(mkt,
                          with_real=CON_BREAKEVEN,
                          with_floor_value=CON_VALORE_FLOOR,
                          uk_new_only=UK_SOLO_NEW,
                          pub_lag_days=PUB_LAG_DAYS,
                          price_field=PRICE_FIELD,
                          exclude_tail=ESCLUDI_CODA,
                          min_ttm_days=MIN_VITA_GG,
                          ytm_convention=YTM_CONVENTION,
                          date_from=DATA_DA)
