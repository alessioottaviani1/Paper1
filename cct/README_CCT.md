# src/cct — pipeline della base CCT-BTP

Script numerati, da lanciare **in ordine e senza argomenti**, dalla root di THESIS:

```powershell
python .\src\cct\00_smoketest.py      # verifica xbbg + suffisso ticker (una volta sola)
python .\src\cct\01_static.py         # anagrafica, regimi, diagnostica campione
python .\src\cct\02_pull_prices.py    # download completo (rilanciare se si ferma al limite)
python .\src\cct\03_bot_auction.py    # rendimenti d'asta BOT 6m derivati dai prezzi
python .\src\cct\04_pairing.py        # appaiamento CCT->BTP con vincolo di convivenza
python .\src\cct\05_indexation.py     # scadenzario cedolare (le 3 regole MEF)
```

In arrivo: `06_curve.py` (Svensson sui prezzi BTP+BOT), `07/08/09` i tre engine di base.

## Note
- `02` ha una cache per titolo: rilanciarlo non riscarica nulla di gia' presente.
  Se il terminale rifiuta per limite giornaliero, lo script si ferma, salva e lo dichiara.
- `03` deriva il rendimento d'asta dal prezzo del BOT semestrale alla data di emissione
  (y = (100-P)/P * 360/n, convenzione MEF). Se disponibile, la serie ufficiale
  dt.tesoro.it in `bot_auction_6m_official.csv` ha la precedenza.
- `05` non usa STRIPS: il residuo della replica e' valutato sulla curva fittata (vedi sotto).

## Perche' NIENTE STRIPS
Fleckenstein-Longstaff (JFE 2020) chiudono i flussi residui della replica con STRIPS.
Hartley-Jermann (JFE 2024) demoliscono proprio quel punto: la replica "soffre dell'estrema
illiquidita' degli strumenti usati per costruirla". Gli STRIPS italiani esistono dal 1998
per BTP >= 5 anni ma sono molto illiquidi, e il loro premio di liquidita' finirebbe dentro
la base come falso mispricing.

Alternativa adottata: il residuo -- che e' piccolo (lo spread cedolare e i disallineamenti
di data) -- viene attualizzato sulla **curva zero sovrana fittata su BTP e BOT**, cioe' sugli
strumenti piu' liquidi del mercato. La parte pesante della replica (BTP + swap) resta fatta
di strumenti quotati e liquidi; solo la coda usa la curva. Il trade-off e' dichiarato: la
misura e' marginalmente meno "eseguibile" e marginalmente piu' model-based, ma non e'
contaminata dal premio di illiquidita' degli STRIPS. Se in futuro si vogliono gli STRIPS,
entrano come robustezza, non come costruzione primaria.
