"""03 - CURVE PUBBLICHE: costruisce i pannelli parquet dalle curve BoE (nominale, reale,
inflation; spot e forward; std e short-end) e dai file Fed/GSW (nominale, TIPS, Kim-Wright),
con i sanity check: identita' di Fisher additiva sulla BoE e verdetto sul compounding GSW
(le colonne SVENY/TIPSY vengono confrontate con la formula NSS sui parametri pubblicati).

Offline. Prima di lanciare, in data/raw/ servono:
  BoE/  -> i 21 xlsx dei tre ZIP (glcnominalddata, glcrealddata, glcinflationddata), unzippati
  Fed/  -> feds200628.csv, feds200805.csv, feds200533.csv"""

# ----------------------------------------------------------------- impostazioni
SCARICA_BUNDESBANK = True   # scarica via API i parametri Svensson DE (non serve Bloomberg)

# ----------------------------------------------------------------- esecuzione
import curves
curves.build_all()

print("=== Bundesbank (curva nominale DE, gold standard) ===")
if SCARICA_BUNDESBANK:
    try:
        curves.download_bundesbank()
    except Exception as e:
        print(f"  download API fallito ({e}): scaricare a mano dal portale in data/raw/Bundesbank")
try:
    curves.build_bundesbank()
except FileNotFoundError as e:
    print(f"  saltato: {e}")
