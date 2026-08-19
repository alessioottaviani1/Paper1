"""r1b - PERCHE' il premio residuo differisce fra US e UK (seconda meta' della richiesta RR n.1).

RR chiede: "Compare the residual risk premium for US and UK. Similar? If different, WHY?".
r1 risponde alla prima meta' (li calcola e li confronta); questo script affronta la seconda.

DUE DIFFERENZE ISTITUZIONALI, non congetture: sono scritte nei prospetti.

  (A) FLOOR DI DEFLAZIONE. I TIPS rimborsano almeno il nominale: il capitale indicizzato
      non puo' scendere sotto la pari. I linker britannici NON hanno questa protezione.
      E' un'opzione put sull'inflazione cumulata che i TIPS hanno e i gilt no. Vale di piu'
      quando il rischio di deflazione sale, alza il prezzo dei TIPS, abbassa il loro
      rendimento reale e quindi ALZA il BEI americano.
      PREDIZIONE: il divario US-UK si allarga negli spaventi deflazionistici.
      EVENTI: Q4 2008 (Lehman) e marzo-aprile 2020 (COVID).

  (B) DOMANDA LDI BRITANNICA. I fondi pensione UK comprano linker lunghi per il matching
      delle passivita' -- una clientela strutturale che gli USA non hanno. Rende i linker
      britannici cari, quindi il BEI del Regno Unito alto, in via ordinaria.
      PREDIZIONE con segno OPPOSTO: nella crisi LDI di settembre-ottobre 2022 le vendite
      forzate invertono il canale e il premio britannico crolla.

Le due predizioni hanno SEGNO e FINESTRA diversi, quindi sono distinguibili: e' questo che
rende il test informativo invece di una narrazione.

  (C) CONTROLLO. Il floor vale solo se il livello dei prezzi puo' scendere sotto quello
      d'emissione: e' profondamente "out of the money" quando l'inflazione cumulata dal
      collocamento e' alta. Su titoli emessi da tempo il canale (A) deve essere DEBOLE.
      Se invece risultasse forte ovunque, non e' il floor ma qualcos'altro.

Output: results/r1b_why.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np, pandas as pd
from config import CACHE

EVENTI = {
    "Lehman / deflazione 2008":   ("2008-09-01", "2009-03-31", "A"),
    "COVID / deflazione 2020":    ("2020-03-01", "2020-05-31", "A"),
    "crisi LDI UK 2022":          ("2022-09-01", "2022-10-31", "B"),
    "picco inflazione 2022":      ("2022-03-01", "2022-08-31", "-"),
}

def nw_t(y, X, lags=18):
    """OLS con errori Newey-West. Ritorna (coef, t) per ogni colonna di X."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        A = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return b, np.divide(b, se, out=np.full_like(b, np.nan), where=se > 0)

if __name__ == "__main__":
    L = []; P = L.append
    P("=== r1b PERCHE' il premio differisce fra US e UK ===")
    f = CACHE / "an_bei_premia.csv"
    if not f.exists():
        f = Path("results") / "an_bei_premia.csv"
    if not f.exists():
        P("[STOP] an_bei_premia.csv non trovato: lanciare prima r1_bei_premia.py")
        print("\n".join(L)); raise SystemExit
    D = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
    P(f"campione: {D.index.min().date()} -> {D.index.max().date()}, {len(D)} osservazioni")

    for k in (5, 10):
        cu, cuk = f"BEI-ISR_US_{k}y", f"BEI-ISR_UK_{k}y"
        if cu not in D.columns or cuk not in D.columns:
            P(f"\n[{k}y] colonne assenti: {cu} / {cuk}"); continue
        d = D[[cu, cuk]].dropna().copy()
        d["gap"] = d[cu] - d[cuk]
        P(f"\n{'='*72}\nSCADENZA {k} ANNI  (premio = BEI - swap di inflazione)\n{'='*72}")
        P(f"  US  : media {d[cu].mean():+.3f}%  sd {d[cu].std():.3f}")
        P(f"  UK  : media {d[cuk].mean():+.3f}%  sd {d[cuk].std():.3f}")
        P(f"  gap : media {d.gap.mean():+.3f}%  sd {d.gap.std():.3f}  "
          f"corr(US,UK) {d[cu].corr(d[cuk]):+.3f}")
        P(f"  n {len(d)} | {d.index.min().date()} -> {d.index.max().date()}")

        P("\n  --- EVENT STUDY: le due predizioni hanno finestre e segni diversi ---")
        base = d.gap.median()
        P(f"  {'finestra':>28}{'gap medio':>12}{'vs mediana':>12}{'canale':>9}{'n':>6}")
        for lab, (a, b, ch) in EVENTI.items():
            w = d.loc[a:b, "gap"]
            if len(w) == 0: continue
            P(f"  {lab:>28}{w.mean():>12.3f}{w.mean()-base:>12.3f}{ch:>9}{len(w):>6}")
        P(f"  {'(mediana di campione)':>28}{base:>12.3f}")
        P("\n  attesi: canale A (floor) -> gap POSITIVO negli spaventi deflazionistici;")
        P("          canale B (LDI)   -> nel 2022 UK il premio britannico CROLLA, quindi")
        P("                              gap positivo ma per una ragione diversa.")
        P("          Se il gap si allarga in ENTRAMBI, i due canali non sono distinguibili")
        P("          con questi soli eventi e serve un proxy continuo (sotto).")

        P("\n  --- PROXY CONTINUI ---")
        reg, nomi = [], []
        b1 = D.get(f"BEI_US_{k}y")
        if b1 is not None:
            # rischio di deflazione: BEI americano molto basso = deflazione prezzata
            defl = (-b1.reindex(d.index)).rename("defl")
            reg.append(defl); nomi.append("deflazione (=-BEI_US)")
        bu = D.get(f"BEI_UK_{k}y")
        if bu is not None and b1 is not None:
            slope = (D.get(f"BEI_UK_10y") - D.get(f"BEI_UK_5y"))
            if slope is not None:
                reg.append(slope.reindex(d.index).rename("uk_slope"))
                nomi.append("pendenza BEI UK 10-5 (proxy domanda LDI lunga)")
        if reg:
            R = pd.concat([d.gap] + reg, axis=1).dropna()
            if len(R) > 60:
                bcoef, t = nw_t(R.iloc[:, 0].values, R.iloc[:, 1:].values, lags=18)
                P(f"  regressione del gap su {len(nomi)} proxy, Newey-West 18 lag, n {len(R)}")
                for i, nm in enumerate(nomi):
                    P(f"    {nm:44s} coef {bcoef[i+1]:+.4f}  t {t[i+1]:+.2f}")
                P("    [atteso: deflazione POSITIVO -- il floor dei TIPS vale di piu' e")
                P("     allarga il gap; pendenza UK con segno da interpretare secondo la")
                P("     direzione della domanda LDI]")
            else:
                P(f"  campione troppo corto dopo l'allineamento (n={len(R)})")

    P("\n" + "="*72)
    P("COSA MANCA PER CHIUDERE IL 'WHY'")
    P("="*72)
    P("  1. Il floor andrebbe PREZZATO, non solo proxato: il suo valore dipende")
    P("     dall'inflazione cumulata dall'emissione (out of the money sui titoli vecchi).")
    P("     Con i dati per ISIN si costruisce l'indicatore di 'moneyness' del floor e la")
    P("     predizione diventa cross-sezionale, non solo temporale -- molto piu' forte.")
    P("  2. La domanda LDI andrebbe misurata: le statistiche di detenzione dei gilt per")
    P("     settore (ONS / BoE) danno la quota dei fondi pensione, che e' il vero proxy.")
    P("  3. Restano due differenze non testate qui e da dichiarare: il lag di indicizzazione")
    P("     (3 mesi US, 3 mesi UK nuovo stile, 8 mesi vecchio stile) e il trattamento")
    P("     fiscale. Entrambe spostano il BEI e nessuna delle due e' controllata.")
    out = Path("results"); out.mkdir(exist_ok=True)
    (out / "r1b_why.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
