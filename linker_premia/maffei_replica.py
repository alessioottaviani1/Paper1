"""maffei_replica - VERIFICA DI ALLINEAMENTO al periodo di Maffei.

Non e' una risposta a Rebonato: le tre domande stanno in r1 (Q1), r2_surprises (Q2),
r3_predictability (Q3). Questo file serve a UNA cosa sola: girare la specificazione di
Maffei sul SUO campione esatto (ago-2007 -> set-2024, tesi sez. 4) e confrontare i numeri
con le sue Tabelle III e V, per dimostrare che la pipeline replica la sua.

La specificazione NON e' duplicata: viene importata da r2_surprises, che la implementa
(regressioni separate eq. 7, sorpresa MA120 piena su CPI SA, liquidita' PC1 a 4 proxy,
campionamento alla release). Cosi' i due file non possono divergere.

Stampa in piu' rispetto a r2:
  - la riga con i numeri pubblicati da Maffei (Tab. V) accanto ai nostri;
  - la SENSIBILITA' AL LAG di Newey-West sul tratto lungo (6/12/18 lag + OLS): il suo
    t-7.2 a 20y viene da OLS semplice, e la sorpresa contiene lo YoY (finestra
    sovrapposta a 12 mesi), quindi serve sapere se la significativita' del lungo
    sopravvive a un troncamento adeguato o se viveva sul lag corto.
"""
import pandas as pd
from r2_surprises import run, MAFFEI_SPAN

if __name__ == "__main__":
    print(">>> maffei_replica v4 -- campione di Maffei (2007-08 -> 2024-09) <<<")
    res = []
    for mkt in ("US", "UK"):
        t = run(mkt, MAFFEI_SPAN)
        if not t.empty:
            res.append(t)
    if res:
        pd.concat(res).to_csv("an_maffei_replica_span.csv", index=False)
        print("\nsalvato: an_maffei_replica_span.csv")
    print("\nlettura: se la riga t_OLS riproduce i t di Maffei (Tab. V: 10y -2.7, 12y -3.9,")
    print("20y -7.2), fonti e metodo sono i suoi. La riga t_NW e la sensibilita' al lag")
    print("dicono quanto di quella significativita' resta con errori standard corretti.")
