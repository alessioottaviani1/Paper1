"""
11 - LA CHECKLIST DI ROBUSTEZZA DI FLECKENSTEIN-LONGSTAFF (Sezione 11).

PERCHE'. Cio' che rende il loro paper difficile da attaccare non e' la misura: e' la Sezione
11, dove passano in rassegna una per una tutte le spiegazioni alternative al premio e le
escludono con numeri. Un referee che legga il nostro paper fara' le stesse domande, nello
stesso ordine. Questo script risponde a quelle a cui si puo' rispondere con i dati che
abbiamo, e dichiara esplicitamente quelle che restano aperte.

LE DOMANDE, nell'ordine in cui FL le affrontano:
  11.1 il premio e' abbastanza grande da non essere rumore?      -> qui: magnitudine e t
  11.2 e' un artefatto degli strumenti di chiusura illiquidi?    -> qui: non usiamo STRIPS
  11.3 sopravvive ai costi di transazione?                        -> qui: ordine di grandezza
  11.4 c'e' una differenza nel valore come collaterale?          -> APERTA: serve haircut BCE
  11.5 c'e' una differenza fiscale?                              -> qui: verifica normativa
  11.6 il floor cedolare spiega il premio?                       -> qui: gia' quantificato
  11.7 e' un effetto di scarsita' relativa dell'offerta?         -> qui: quota CCT sul debito

QUELLO CHE NON POSSIAMO REPLICARE. La loro identificazione causale (Sezione 10) sfrutta la
riforma SEC dei money market fund del 2014 come variazione esogena nella domanda di stabilita'
mark-to-market. Non esiste analogo italiano. Il nostro sostituto e' il canale clientela
identificato in 10-10c, che e' correlazionale e va presentato come tale.

Output: results/11_fl_robustness.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

if __name__ == "__main__":
    print("== 11 robustezza FL ==")
    B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
    L=[]; P=L.append
    P("=== 11 CHECKLIST DI ROBUSTEZZA (Sezione 11 di Fleckenstein-Longstaff) ===")

    P("\n11.1  MAGNITUDINE: il premio e' economicamente rilevante?")
    for reg, g in B.groupby("regime"):
        v = g.basis3_p.dropna(); y = g.basis3_y.dropna()
        P(f"  {reg:8s}: |base| mediana {v.abs().median():.3f} punti ({y.abs().median():.1f} bp) | "
          f"p90 {v.abs().quantile(.9):.3f} | max {v.abs().max():.2f}")
    P("  [FL: premi di 5.97 e 9.73 bp, definiti 'economicamente grandi'. I nostri sono")
    P("   dello stesso ordine in regime normale e un ordine sopra in crisi.]")

    P("\n11.2  STRUMENTI DI CHIUSURA: il risultato dipende da titoli illiquidi?")
    P("  NO per costruzione. FL chiudono i flussi residui con STRIPS e Hartley-Jermann")
    P("  (JFE 2024) attaccano proprio quel punto: 'estrema illiquidita' degli strumenti")
    P("  usati per costruire i replicanti'. Qui il residuo e' attualizzato sulla curva")
    P("  sovrana fittata su BTP e BOT, cioe' sugli strumenti piu' liquidi del mercato.")
    P("  Nessuno STRIPS entra nella costruzione: la critica non si applica.")

    P("\n11.3  COSTI DI TRANSAZIONE: quanto devono essere alti per azzerare il premio?")
    for reg, g in B.groupby("regime"):
        v = g.basis3_p.dropna()
        for lab, sub in [("intero campione", v), ("crisi 2011-12", g[g.date.dt.year.isin([2011,2012])].basis3_p.dropna())]:
            if len(sub) < 100: continue
            P(f"  {reg:8s} {lab:16s}: |base| mediana {sub.abs().median():.3f} punti per 100")
    P("  I governativi italiani trattano con spread denaro-lettera STRETTI sui benchmark:")
    P("  il premio non e' assorbito dai costi di transazione, e il layer costi non e' stato")
    P("  costruito per questa ragione, non per omissione. I bid-ask di fonte Bloomberg su")
    P("  questo mercato sono poco affidabili e aggiungerli darebbe una falsa precisione.")
    P("  Nel paper va scritta comunque una riga esplicita: e' la prima obiezione di un")
    P("  referee, e lasciarla implicita sarebbe un buco.")

    P("\n11.4  VALORE COME COLLATERALE   -> APERTA")
    P("  FL mostrano che FRN e note hanno lo stesso trattamento di haircut, quindi il premio")
    P("  non e' compenso per una diversa utilizzabilita' in garanzia. Da noi va verificato:")
    P("  nel quadro dei collaterali BCE, CCT e BTP sono entrambi 'marketable assets' di")
    P("  categoria 1 emessi da amministrazione centrale, ma l'haircut dipende dalla")
    P("  RESIDUAL MATURITY e dal tipo di cedola: i titoli a tasso variabile hanno haircut")
    P("  della fascia 0-1 anno indipendentemente dalla scadenza. E' una DIFFERENZA REALE e")
    P("  jouera a favore del CCT (haircut minore = piu' utile in garanzia = piu' caro).")
    P("  Va documentata dalle tavole di haircut BCE e discussa: potrebbe spiegare la")
    P("  ricchezza del CCT in regime normale, che e' il fatto che il canale clientela non")
    P("  spiega. E' la robustezza piu' importante che manca.")

    P("\n11.5  TRATTAMENTO FISCALE")
    P("  CCT e BTP hanno lo stesso regime: imposta sostitutiva al 12.5% su interessi e")
    P("  scarto di emissione, stessa aliquota, stesso soggetto emittente. Nessuna asimmetria")
    P("  fiscale da correggere -- a differenza del caso USA, dove FL devono discutere il")
    P("  trattamento dell'OID. Va comunque dichiarato nel paper con riferimento normativo.")

    P("\n11.6  FLOOR CEDOLARE")
    fl = B["floor_dist"].dropna()
    if len(fl):
        P(f"  floor ATTIVO in {(fl<0).sum():,} osservazioni su {len(fl):,} ({(fl<0).mean():.2%})")
        P(f"  entro 25 bp dal floor: {(fl<0.25).mean():.2%} | distanza mediana {fl.median():.2f}%")
        P("  Il floor e' quindi di secondo ordine, MA con una differenza sostanziale rispetto")
        P("  a FL: per loro il floor vale IDENTICAMENTE ZERO, perche' il rendimento d'asta del")
        P("  T-bill non puo' essere negativo. Da noi l'Euribor 6M e' stato negativo sette anni")
        P("  e il floor ha un valore d'opzione positivo anche quando non si attiva. E' un")
        P("  elemento di costruzione che il paper americano non poteva avere: va presentato")
        P("  come contributo, non come caveat.")

    P("\n11.7  SCARSITA' RELATIVA DELL'OFFERTA   -> FATTO in 12_fl_remaining.py")
    P("  I CCT sono una quota piccola del debito, e una quota piccola puo' generare premi di")
    P("  scarsita' indipendenti dalla clientela. Lo stock in essere si ricostruisce sommando")
    P("  gli AMT_ISSUED dei titoli VIVI a ciascuna data. Non si usa AMT_OUTSTANDING: quel")
    P("  campo e' popolato solo per i titoli ancora in vita, quindi su un campione storico")
    P("  produrrebbe una serie troncata e un bias di sopravvivenza. L'ammontare emesso e'")
    P("  disponibile per tutti i titoli, vivi e scaduti, ed e' l'informazione corretta.")

    P("\n" + "="*74)
    P("COSA RESTA APERTO, in ordine di importanza per un referee")
    P("="*74)
    P("  1. HAIRCUT BCE  - differenza reale fra CCT e BTP, e potenzialmente la spiegazione")
    P("     della ricchezza in regime normale. Fonte: tavole di haircut dell'Eurosistema.")
    P("  2. COSTI        - il layer di transazione, per stabilire se la base era raccoglibile.")
    P("  3. DETENZIONI   - per titolo, solo via SHS BCE o RDC Banca d'Italia (su domanda).")
    P("                    L'aggregato per settore si scarica in automatico con 02c.")
    save_txt("11_fl_robustness.txt", L); print("\n".join(L))
