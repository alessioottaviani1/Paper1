"""36 - I PREZZI DEI NOMINALI TEDESCHI. RICHIEDE IL TERMINALE, ma e' uno scarico minuscolo.

PERCHE' NON BASTA IL 02. In fetch_nominal_ytm c'e' questo ramo:

    if mkt not in CURVE_FROM_FILE:
        _update_wide(f"px_nom_{mkt}", t2c, "PX_MID")

e CURVE_FROM_FILE contiene DE finche' DE_NOMINAL_CURVE vale 'bundesbank'. La logica e'
giusta per il suo scopo: se la curva nominale tedesca viene dai parametri Svensson
ufficiali, i PREZZI dei nominali non servono a fittare niente e non si scaricano. Lanciare
il 02 con SOLO_MERCATO = "DE" scaricherebbe quindi solo gli YTM -- che fra l'altro ci sono
gia' -- e px_nom_DE resterebbe mancante, senza un errore che lo dica.

MA A NOI I PREZZI SERVONO LO STESSO, per un motivo che quel ramo non poteva prevedere: la
nostra base non usa la curva nominale per scontare. Usa la curva SWAP, e i nominali le
servono come GEMELLI -- e lo Z-spread di un gemello si calcola dai suoi flussi contro il
suo dirty price osservato. Senza prezzo non c'e' gamba nominale, qualunque cosa faccia la
curva tedesca.

PERCHE' NON CAMBIARE DE_NOMINAL_CURVE A 'fit'. Funzionerebbe -- toglierebbe DE da
CURVE_FROM_FILE e i prezzi arriverebbero -- ma porterebbe dietro due cose che non abbiamo
chiesto. Toglierebbe anche il filtro del pool di matching, quindi si scaricherebbe lo
spettro completo dei DBR/OBL/BKO, circa 257 titoli in piu' con anagrafica da rifare. E
soprattutto cambierebbe il significato di un'ALTRA misura: quel parametro dice da dove
viene la curva nominale per il C-esatto, ed e' una decisione metodologica che non va presa
come effetto collaterale del voler dei prezzi.

Quindi: si scaricano i prezzi dei 34 nominali che sono GIA' in anagrafica, e si lascia
config come sta. Incrementale come tutto il resto: se il file esiste gia', chiede solo i
giorni e i titoli mancanti.
"""
import pandas as pd
import bbg
from config import CACHE

MERCATO = "DE"

try:
    from xbbg import blp  # noqa: F401
except ImportError:
    raise SystemExit("xbbg non disponibile: questo script va lanciato sul terminale.")

ref_n = bbg.load("ref_nominal")
sub = ref_n[ref_n["mkt"] == MERCATO]
if not len(sub):
    raise SystemExit(f"nessun nominale {MERCATO} in ref_nominal: lancia prima 01 e 02.")

t2c = {f"{str(r['bb_id']).strip()} Corp": isin for isin, r in sub.iterrows()
       if pd.notna(r.get("bb_id"))}
print(f"=== prezzi dei nominali {MERCATO} ===")
print(f"    {len(t2c)} titoli in anagrafica, campo PX_MID, da {bbg.PULL_FLOOR:%Y-%m-%d}")
p = CACHE / f"px_nom_{MERCATO}.parquet"
if p.exists():
    vecchio = pd.read_parquet(p)
    print(f"    cache esistente: {vecchio.shape[0]} date x {vecchio.shape[1]} titoli "
          f"-> chiedo solo i mancanti")
else:
    print(f"    nessuna cache: scarico completo")

out = bbg._update_wide(f"px_nom_{MERCATO}", t2c, "PX_MID")

if out is None or not len(out):
    raise SystemExit("\nrisposta vuota: controlla il limite giornaliero o i ticker.")
viva = out.index[out.notna().any(axis=1)]
print(f"\nfatto: {out.shape[0]} date x {out.shape[1]} titoli, "
      f"{viva.min():%Y-%m-%d} -> {viva.max():%Y-%m-%d}")
print(f"    quotati per data: mediana {int(out.notna().sum(axis=1).median())}, "
      f"minimo {int(out.notna().sum(axis=1).min())}")
manca = [c for c in t2c.values() if c not in out.columns]
if manca:
    print(f"    !! {len(manca)} titoli senza nessun prezzo: {', '.join(manca[:6])}")
    print("       se sono tanti, il pool tedesco non regge il matching e va rivista")
    print("       la scelta di DE_NOMINAL_CURVE prima di calcolare la base.")
print(f"\nsalvato: {p}")
print("    ora il 24 con MERCATO = 'DE' ha tutto quello che gli serve.")
