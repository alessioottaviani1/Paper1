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
MERCATI          = ["UK"]
CON_BREAKEVEN    = False      # calcola anche RealYtM e breakeven (circa raddoppia i tempi)
CON_VALORE_FLOOR = False      # valore del floor in bp (port Black dell'originale)
UK_SOLO_NEW      = False      # True per riprodurre il comportamento pre-old-style
PUB_LAG_DAYS     = None       # None = originale; es. 20 = modalita' onesta (paper)
PRICE_FIELD      = "px_mid"   # 'px_ask' per la versione trading
ESCLUDI_CODA     = False      # True: esclude l'ultimo anno di vita dei linker
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
                          ytm_convention=YTM_CONVENTION)
