"""ecb_factors - costruisce i fattori di sorpresa di politica monetaria BCE dall'EA-MPD
(Altavilla et al. 2019, JME; file ufficiale Dataset_EA-MPD.xlsx in Ecb/).

METODO -- ortogonalizzazioni sequenziali "poor man's factors", NON factor analysis.
Validato dagli autori stessi (Altavilla-Gurkaynak-Kind-Laeven 2025, ECB WP 3157): questo
schema produce fattori correlati >90% con quelli della factor analysis completa con
rotazione. Definizione (WP 3157, Sez. 2):
  - TARGET           = variazione OIS 1M nella Press Release Window (la sorpresa sul tasso
                       di policy: la decisione, annunciata alle 13:45).
  - FORWARD GUIDANCE = variazione OIS 2Y nella Monetary Event Window, ORTOGONALE al Target
                       (revisione delle aspettative sul sentiero futuro, oltre la decisione).
  - QE               = variazione OIS 10Y (o Bund 10Y pre-2011) nella Monetary Event Window,
                       ORTOGONALE ai primi due (muove il long-end, dove vivono i breakeven).

VINCOLO DATI (dichiarato nel paper): l'OIS 10Y ad alta frequenza esiste solo dal 2011-07.
Prima si usa il rendimento tedesco DE10Y come proxy del long-end (Altavilla et al. fanno
esattamente questo). Colonna long_src indica quale fonte e' stata usata per riga.

INPUT : Ecb/Dataset_EA-MPD.xlsx  (fogli 'Press Release Window', 'Monetary Event Window')
OUTPUT: data/cache/ecb_factors.parquet  -- per riunione: date, target, fg, qe,
        + spread sovrano diretto IT10Y-DE10Y (l'effetto BCE sul differenziale) e i grezzi.

Lancia:  python .\\src\\linker_premia\\ecb_factors.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# usa la STESSA cache di tutti gli altri script (config.CACHE), non una euristica
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
from config import CACHE


# ------------------------------------------------------------------ percorsi
# cerca l'Excel EA-MPD in modo robusto (repo, THESIS, data/raw/Ecb, ...)
def _find_xlsx():
    here = Path(__file__).resolve()
    roots = [here.parent.parent.parent, here.parent.parent, Path.cwd(), CACHE.parent.parent]
    for base in roots:
        for f in base.rglob("Dataset_EA-MPD.xlsx"):
            return f
    return None


def _load_window(xlsx, sheet):
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    # tutto in numerico (bp per i tassi)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _orth(y, X):
    """Residuo di y sullo spazio di X (con costante). y, X allineati, no-NaN.
    Ritorna la serie residua sull'indice di y."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < 10:
        # troppo pochi punti: ritorna y demeaned (nessuna ortogonalizzazione affidabile)
        return (y - y.mean())
    Xv = np.column_stack([np.ones(len(d))] + [d[c].values for c in X.columns])
    b, *_ = np.linalg.lstsq(Xv, d["y"].values, rcond=None)
    resid = d["y"].values - Xv @ b
    out = pd.Series(np.nan, index=y.index)
    out.loc[d.index] = resid
    return out


def build_factors():
    xlsx = _find_xlsx()
    if xlsx is None:
        raise SystemExit("[!] Dataset_EA-MPD.xlsx non trovato (cartella Ecb/).")
    print(f"[ea-mpd] leggo {xlsx}")
    prw = _load_window(xlsx, "Press Release Window")     # decisione (13:45)
    mew = _load_window(xlsx, "Monetary Event Window")    # release + conferenza

    idx = mew.index  # tutte le riunioni
    F = pd.DataFrame(index=idx)

    # --- TARGET: OIS 1M nella Press Release Window (la sorpresa sulla decisione)
    F["target"] = prw["OIS_1M"].reindex(idx)

    # --- long-end source: OIS_10Y dal 2011-07, prima Bund DE10Y (come Altavilla et al.)
    ois10 = mew["OIS_10Y"].reindex(idx)
    de10 = mew["DE10Y"].reindex(idx)
    long10 = ois10.copy()
    long10 = long10.where(long10.notna(), de10)          # riempi i buchi (pre-2011) col Bund
    F["long_src"] = np.where(ois10.notna(), "OIS_10Y", "DE10Y")

    ois2 = mew["OIS_2Y"].reindex(idx)

    # --- FORWARD GUIDANCE: OIS 2Y nella Monetary Event Window, ortogonale al Target
    F["fg"] = _orth(ois2, F[["target"]])

    # --- QE: long-end (OIS 10Y o Bund) ortogonale a Target e FG
    F["qe"] = _orth(long10, F[["target", "fg"]])

    # --- effetto BCE DIRETTO sul differenziale sovrano IT-DE (per il controllo del test)
    it10 = mew["IT10Y"].reindex(idx)
    F["it10"] = it10
    F["de10"] = de10
    F["sov_spread_shock"] = it10 - de10                  # quanto la riunione muove IT-DE (bp)
    # anche 5y se serve (piu' coperto per l'Italia in alcune fasi)
    if "IT5Y" in mew.columns:
        F["it5"] = mew["IT5Y"].reindex(idx)
        F["de5"] = mew["DE5Y"].reindex(idx) if "DE5Y" in mew.columns else np.nan
        F["sov_spread_shock_5y"] = F["it5"] - F["de5"]

    # grezzi utili
    F["ois2_raw"] = ois2
    F["long10_raw"] = long10

    return F


def monthly_aggregate(F):
    """Aggrega le sorprese per MESE (somma delle riunioni nel mese di calendario).
    I mesi senza riunione avranno sorpresa 0 (nessuno shock monetario quel mese).
    Ritorna un DataFrame indicizzato a fine mese, pronto per il merge con lambda mensile."""
    cols = ["target", "fg", "qe", "sov_spread_shock"]
    m = F[cols].copy()
    m.index = m.index.to_period("M").to_timestamp("M")   # fine mese
    agg = m.groupby(level=0).sum(min_count=1)             # somma nel mese; NaN se nessun dato
    # conteggio riunioni per mese (per diagnostica)
    agg["n_meetings"] = F.groupby(F.index.to_period("M").to_timestamp("M")).size()
    return agg


if __name__ == "__main__":
    F = build_factors()
    cache = CACHE
    out = cache / "ecb_factors.parquet"
    F.to_parquet(out)

    # diagnostica
    print(f"\n[ok] {len(F)} riunioni BCE, {F.index.min().date()} -> {F.index.max().date()}")
    print(f"     long-end: {(F.long_src=='OIS_10Y').sum()} da OIS_10Y, "
          f"{(F.long_src=='DE10Y').sum()} da Bund (pre-2011)")
    print("\n  fattori (bp) -- media e dev.std:")
    for c in ["target", "fg", "qe", "sov_spread_shock"]:
        s = F[c].dropna()
        print(f"    {c:18s} n={len(s):3d}  media {s.mean():+.2f}  sd {s.std():.2f}  "
              f"[{s.min():+.0f}, {s.max():+.0f}]")

    # correlazioni tra fattori (devono essere ~0 per costruzione: target/fg/qe ortogonali)
    print("\n  correlazioni (target/fg/qe ~0 per costruzione):")
    cc = F[["target", "fg", "qe"]].corr()
    print(cc.round(2).to_string())

    # aggregato mensile
    M = monthly_aggregate(F)
    Mout = cache / "ecb_factors_monthly.parquet"
    M.to_parquet(Mout)
    print(f"\n  aggregato mensile: {len(M)} mesi con riunioni, salvato in {Mout.name}")
    print(f"  mesi con >=1 riunione: {(M.n_meetings>=1).sum()} | "
          f"con 2 riunioni: {(M.n_meetings>=2).sum()}")

    print(f"\nsalvato: {out.name} (per-riunione) + {Mout.name} (mensile)")
    print("\nprossimo: r2f usa ecb_factors_monthly per controllare lambda_IT-DE ~ InfS + target+fg+qe")
