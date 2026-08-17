"""
02b - Serie aggiuntive per il test del meccanismo. Prova i ticker e scarica quelli validi.

PERCHE' UNO SCRIPT SEPARATO. La sintassi dei ticker CDS e vol su Bloomberg varia per
licenza e per convenzione; una lista fissa rischia di tornare vuota senza dirlo. Qui ogni
serie ha PIU' CANDIDATI: si prova il primo, se torna vuoto si passa al successivo, e alla
fine si stampa quale ha funzionato. Cosi' il fallimento e' visibile e diagnosticabile.

COSA SERVE E PERCHE' (in ordine di quanto sposta il paper):

  1. CDS DI CINQUE BANCHE ITALIANE (UniCredit, BPM/Banco, Banca Popolare di Milano,
     Intesa, MPS) -- e' la misura DIRETTA dello stress della clientela. Oggi quel canale
     e' catturato da Euribor-OIS, che e' funding interbancario EUROPEO: un proxy indiretto.
     Con cinque nomi si costruisce una mediana di settore, robusta ai salti idiosincratici
     (fusioni, aumenti di capitale, il caso MPS) che renderebbero fragile un singolo CDS.
     Il test diventa "quando le banche italiane sono sotto stress il CCT si sconta", che e'
     l'affermazione che il paper vuole fare.

  2. CDS SOVRANO ITALIA -- separa il rischio di credito puro dal premio Italia-sopra-swap
     usato ora, che mescola credito e frizioni di mercato. Con entrambi si puo' mostrare che
     il canale NON e' solo credito sovrano: e' la parte non-credito a muovere la base.

  3. RENDIMENTI BUND -- permettono lo spread BTP-Bund, la misura di stress sovrano euro
     piu' usata in letteratura. Serve per confrontabilita' con gli altri paper sull'Italia.

  4. VOLATILITA' CAP/FLOOR EUR -- l'unico ingrediente mancante per prezzare il floor
     cedolare dei CCTeu come opzione invece che trattarlo come diagnostica.

  5. AZIONARIO BANCARIO ITALIANO -- controllo di robustezza sul canale clientela, e
     disponibile con storia lunga anche dove i CDS non arrivano (pre-2004).
"""
import time
import numpy as np, pandas as pd
from config import *
from utils import save_txt

try:
    import bbg
    from bbg import BbgLimitReached, BbgBadSecurity
    HAVE = True
except Exception as e:
    HAVE, ERR = False, str(e)

START, END = pd.Timestamp(START_EXTENDED), pd.Timestamp(END_SAMPLE)
S_STR, E_STR = START.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d")

# nome -> lista di ticker candidati, provati in ordine
# --- CDS: ticker esatti, forniti dall'utente e verificati sul terminale --------
# Convenzione D14 (2014 ISDA Definitions). Cinque emittenti bancari italiani, non uno solo:
# la clientela dei CCT non e' una banca, e' il SISTEMA bancario, e un indice costruito su
# cinque nomi e' molto piu' robusto di un singolo CDS a salti idiosincratici (fusioni,
# aumenti di capitale, il caso MPS). Da questi si costruisce in 10b la mediana cross-sectional,
# che e' la misura di stress della clientela.
# BAMIIM (Banca Popolare di Milano) e' stata rimossa: confluita in Banco BPM nel 2017,
# il CDS e' cessato e il ticker torna BAD_SEC. Restano quattro nomi, sufficienti per una
# mediana di settore robusta.
CDS_BANKS = {
    "cds_unicredit": "UCGIM CDS EUR SR 5Y D14 Corp",
    "cds_bpm":       "BACRED CDS EUR SR 5Y D14 Corp",
    "cds_intesa":    "ISPIM CDS EUR SR 5Y D14 Corp",
    "cds_mps":       "MONTE CDS EUR SR 5Y D14 Corp",
}
CDS_SOV = {"cds_italy": "ITALY CDS USD SR 5Y D14 Corp"}

# nome -> lista di ticker candidati, provati in ordine. Per i CDS il candidato e' uno solo
# perche' il ticker e' noto; per il resto restano le alternative.
CANDIDATES = {
    **{k: [v] for k, v in CDS_BANKS.items()},
    **{k: [v] for k, v in CDS_SOV.items()},
    # Bund, per lo spread BTP-Bund (misura standard di stress sovrano euro in letteratura)
    "bund2":      ["GDBR2 Index"], "bund5": ["GDBR5 Index"], "bund10": ["GDBR10 Index"],
    "btp5_bbg":   ["GBTPGR5 Index"], "btp10_bbg": ["GBTPGR10 Index"],
    # NB: la volatilita' cap/floor EUR, che servirebbe a prezzare il floor cedolare dei
    # CCTeu come opzione, NON e' in questa lista: i ticker plausibili (EUFV/EUSV) tornano
    # tutti BAD_SEC su questa licenza. Va individuato sul terminale (VCUB per il cubo di
    # volatilita', oppure BVOL) e aggiunto a mano. Finche' manca, il floor resta trattato
    # come diagnostica di distanza, che per il 99% delle osservazioni e' sufficiente.
    "srvix":      ["SRVIX Index"],
    # azionario bancario italiano: controllo di robustezza con storia piu' lunga dei CDS
    "banks_it":   ["IT8300 Index"],
    "ftsemib":    ["FTSEMIB Index"],
}

if __name__ == "__main__":
    print("== 02b serie aggiuntive ==")
    L=[]; P=L.append
    P("=== 02b SERIE AGGIUNTIVE PER IL MECCANISMO ===")
    if not HAVE:
        P(f"[STOP] modulo bbg non disponibile: {ERR}"); save_txt("02b_pull_extra.txt",L)
        print("\n".join(L)); raise SystemExit
    P(f"periodo {START.date()} -> {END.date()} | {len(CANDIDATES)} serie da cercare\n")

    got, miss = {}, {}
    for name, tickers in CANDIDATES.items():
        found = None
        for tk in tickers:
            try:
                d = bbg.bdh([tk], "PX_LAST", S_STR, E_STR, verbose=False)
                if d is not None and not d.empty:
                    ser = pd.to_numeric(d.iloc[:,0], errors="coerce").dropna()
                    if len(ser) > 100:
                        got[name] = ser; found = tk; break
            except BbgLimitReached:
                P("[LIMITE BLOOMBERG] rilanciare domani"); found = "LIMIT"; break
            except BbgBadSecurity:
                print(f"  {name:14s} .. {tk}: ticker inesistente"); continue
            except Exception:
                pass
            time.sleep(0.3)
        if found == "LIMIT": break
        if found:
            ser = got[name]
            line = (f"  {name:14s} OK  {found:34s} {len(ser):>6,} oss.  "
                    f"{ser.index.min().date()} -> {ser.index.max().date()}")
        else:
            miss[name] = tickers
            line = f"  {name:14s} --  nessun candidato valido: {tickers}"
        P(line); print(line)            # stampa SUBITO: un'interruzione non cancella l'esito
        # salvataggio incrementale: se si interrompe, cio' che c'e' resta su disco
        if got: pd.DataFrame(got).sort_index().to_csv(PROC/"extra_series.csv")

    if got:
        df = pd.DataFrame(got).sort_index()
        df.to_csv(PROC/"extra_series.csv")
        P(f"\n[saved] {PROC/'extra_series.csv'}  ({df.shape[1]} serie)")
    if miss:
        P("\nSERIE NON TROVATE: verificare la sintassi sul terminale con <ticker> <GO>.")
        P("  Per i CDS, provare il monitor CDSW o cercare l'emittente con SRCH e leggere il")
        P("  ticker esatto; le convenzioni cambiano fra licenze (D14, CBIL, Curncy, Corp).")
    save_txt("02b_pull_extra.txt", L); print("\n".join(L))
