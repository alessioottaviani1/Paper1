"""
02c - Detenzioni bancarie di titoli pubblici, dall'API SDMX della BCE. Automatico.

PERCHE' LA BCE E NON BANCA D'ITALIA. La Base Dati Statistica di Banca d'Italia e' libera ma
si consulta da un'applicazione interattiva: l'export si fa a mano, non c'e' endpoint
documentato. La BCE invece espone tutti i dati diffusi via SDMX 2.1 REST, quindi lo stesso
contenuto -- il bilancio delle istituzioni finanziarie monetarie italiane, che e' la fonte
da cui Banca d'Italia stessa deriva le proprie tavole -- si scarica in automatico.

COSA SERVE. Il dataset BSI (Balance Sheet Items) contiene i titoli di debito DETENUTI dalle
IFM italiane con controparte Amministrazioni pubbliche. E' il regressore che manca al test
del canale clientela: finora l'attribuzione alle banche e' inferita dall'interazione con la
dimensione dei titoli; con questa serie diventa diretta, perche' si osserva quando le banche
stanno effettivamente riducendo o aumentando il portafoglio sovrano.

LIMITE DA DICHIARARE NEL PAPER. Il dato e' aggregato per STRUMENTO: non separa CCT da BTP.
La Tabella 7 di Fleckenstein-Longstaff, che confronta i detentori dell'FRN con quelli della
note appaiata, richiede microdati a livello di singolo titolo -- Securities Holdings
Statistics della BCE o Research Data Center di Banca d'Italia, entrambi su domanda per
finalita' di ricerca. Con il dato aggregato si testa il COMPORTAMENTO della clientela nel
tempo, non la sua composizione per titolo: e' un test piu' debole ma onesto, e va scritto
come tale.

CHIAVI CANDIDATE. La sintassi della serie key BSI ha undici dimensioni e le convenzioni sono
cambiate nel tempo; qui si provano piu' varianti e si riporta quale risponde, come si e'
fatto per i ticker Bloomberg.

Output: PROC/ecb_series.csv + results/02c_pull_ecb.txt
"""
import io, urllib.request, urllib.error
import numpy as np, pandas as pd
from config import *
from utils import save_txt

BASE = "https://data-api.ecb.europa.eu/service/data"
ALT  = "https://sdw-wsrest.ecb.europa.eu/service/data"

# nome -> lista di (dataset, series_key) candidati
CANDIDATES = {
    # titoli di debito DETENUTI dalle IFM italiane, controparte Amministrazioni pubbliche
    "mfi_govt_holdings_it": [
        ("BSI", "M.IT.N.A.A30.A.1.U6.2100.Z01.E"),
        ("BSI", "M.IT.N.A.A30.A.1.U2.2100.Z01.E"),
        ("BSI", "M.IT.N.A.A30.A.1.IT.2100.Z01.E"),
    ],
    # totale attivo delle IFM italiane: per normalizzare la quota sovrana
    "mfi_total_assets_it": [
        ("BSI", "M.IT.N.A.T00.A.1.Z5.0000.Z01.E"),
        ("BSI", "M.IT.N.A.A20.A.1.U6.2100.Z01.E"),
    ],
    # depositi delle IFM italiane: proxy del passivo a vista da coprire coi CCT
    "mfi_deposits_it": [
        ("BSI", "M.IT.N.A.L20.A.1.U6.2250.Z01.E"),
        ("BSI", "M.IT.N.A.L20.A.1.U2.2250.Z01.E"),
    ],
    # prestiti delle IFM italiane al settore privato: stato del bilancio bancario
    "mfi_loans_private_it": [
        ("BSI", "M.IT.N.A.A20.A.1.U6.2240.Z01.E"),
    ],
}

def fetch(dataset, key, timeout=60):
    """Scarica una serie SDMX in CSV. Prova l'endpoint nuovo e poi quello storico."""
    for base in (BASE, ALT):
        url = f"{base}/{dataset}/{key}?format=csvdata"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
            if not raw.strip() or "<" == raw.strip()[0]: continue
            df = pd.read_csv(io.StringIO(raw))
            cols = {c.upper(): c for c in df.columns}
            if "TIME_PERIOD" not in cols or "OBS_VALUE" not in cols: continue
            s = (df[[cols["TIME_PERIOD"], cols["OBS_VALUE"]]]
                   .dropna().set_index(cols["TIME_PERIOD"])[cols["OBS_VALUE"]])
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[s.index.notna()].astype(float).sort_index()
            if len(s) > 50: return s, url
        except Exception:
            continue
    return None, None

if __name__ == "__main__":
    print("== 02c serie BCE ==")
    L=[]; P=L.append
    P("=== 02c DETENZIONI BANCARIE DI TITOLI PUBBLICI (API SDMX BCE) ===")
    got, miss = {}, []
    for name, cands in CANDIDATES.items():
        s, url = None, None
        for ds, key in cands:
            s, url = fetch(ds, key)
            if s is not None: break
        if s is not None:
            got[name] = s
            line = (f"  {name:22s} OK  {len(s):>5,} oss.  "
                    f"{s.index.min().date()} -> {s.index.max().date()}  key={key}")
        else:
            miss.append(name)
            line = f"  {name:22s} --  nessuna chiave valida"
        P(line); print(line)
        if got: pd.DataFrame(got).sort_index().to_csv(PROC/"ecb_series.csv")

    if got:
        P(f"\n[saved] {PROC/'ecb_series.csv'}  ({len(got)} serie)")
    if miss:
        P(f"\nSERIE NON TROVATE: {miss}")
        P("  La serie key BSI ha 11 dimensioni e le convenzioni cambiano. Per trovarla:")
        P("   1. aprire https://data.ecb.europa.eu/data/datasets/BSI")
        P("   2. filtrare Reference area = Italy, BS item = Debt securities held,")
        P("      Counterpart sector = General Government, Data type = Outstanding amounts")
        P("   3. la chiave completa compare nella pagina della serie: incollarla in CANDIDATES")
    save_txt("02c_pull_ecb.txt", L); print("\n".join(L))
