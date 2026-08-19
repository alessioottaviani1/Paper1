"""r5 - CENSIMENTO DELLE CURVE REALI: quali mercati, quali nodi e quali periodi sono
realmente utilizzabili, e quali risultati vanno rifatti.

PERCHE'. Le curve reali euro (IT/FR/DE) sono fittate DAI LINKER. Il fit (curves.
fit_nss_prices_panel) ha due soglie: sotto 4 bond salta la data, sotto 8 bond passa a
Nelson-Siegel (4 parametri, beta3=0) invece della NSS piena (6 parametri). Con 4-5 bond
si stimano 4 parametri su 4-5 osservazioni: ZERO o UNO gradi di liberta'. Il fit
interpola esattamente i titoli, quindi l'RMSE crolla a ~0 -- un numero che SEMBRA
ottimo ma non dice nulla: un sistema esattamente determinato azzera i residui per
costruzione. Fra un titolo e l'altro, e oltre l'ultimo, e' estrapolazione del modello.

Quindi l'RMSE NON e' il criterio. I criteri veri sono due:
  (a) GRADI DI LIBERTA' del fit = n_bond - n_parametri (4 per NS, 6 per NSS);
  (b) BRACKETING del nodo: quanti titoli stanno intorno alla scadenza che si legge.
      Un nodo letto FRA due titoli e' interpolazione (ok); un nodo OLTRE l'ultimo
      titolo e' estrapolazione (non e' un dato, e' il modello).

COSA STAMPA, per ogni mercato:
  1. quanti linker vivi e prezzati per data (media, min, max) e come evolvono nel tempo;
  2. la ripartizione delle date fra i tre regimi (saltata / NS 4 param / NSS 6 param)
     e i gradi di liberta' medi;
  3. per ogni nodo target (2,5,10,20y): quante date sono INTERPOLATE (nodo dentro
     l'intervallo dei titoli vivi) e quante ESTRAPOLATE (nodo oltre l'ultimo titolo),
     piu' il numero medio di titoli entro +-2 anni;
  4. un VERDETTO per mercato/nodo: utilizzabile, fragile, o non utilizzabile.

Non modifica nulla: legge px_mid_{mkt} (i prezzi effettivamente usati dal fit) e le
scadenze dell'anagrafica. Output: an_curve_census.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np
import pandas as pd
from config import CACHE

MKTS = ("US", "UK", "IT", "FR", "DE")
NODES = (2.0, 5.0, 10.0, 20.0)
MIN_BONDS = 4          # curves.fit_nss_prices_panel: sotto questo la data e' saltata
MIN_BONDS_NSS = 8      # sotto questo -> Nelson-Siegel (4 param) invece di NSS (6)
N_PAR_NS, N_PAR_NSS = 4, 6


def _maturities():
    """MATURITY per ISIN dei linker, con il mercato. Prova ref_linker, poi universe."""
    for name, matcol in (("ref_linker", "MATURITY"), ("universe", "maturity")):
        p = CACHE / f"{name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        mcol = next((c for c in (matcol, "MATURITY", "maturity") if c in df.columns), None)
        kcol = next((c for c in ("MKT", "MARKET", "mercato", "mkt") if c in df.columns), None)
        if mcol is None:
            continue
        out = pd.DataFrame({"maturity": pd.to_datetime(df[mcol], errors="coerce")},
                           index=df.index)
        if kcol:
            out["mkt"] = df[kcol].astype(str).str.upper()
        if "kind" in df.columns:      # universe: tiene solo i linker
            out = out[df["kind"].astype(str).str.lower().str.contains("link", na=False)]
        return out.dropna(subset=["maturity"])
    return None


def _prices(mkt):
    """px_mid_{mkt}: i prezzi che il fit usa davvero (date x ISIN)."""
    p = CACHE / f"px_mid_{mkt}.parquet"
    if not p.exists():
        return None
    px = pd.read_parquet(p)
    px.index = pd.to_datetime(px.index)
    return px.sort_index()


def census_market(mkt, mat):
    px = _prices(mkt)
    if px is None:
        print(f"\n{'='*78}\n{mkt}: px_mid_{mkt}.parquet assente -- salto\n{'='*78}")
        return []
    mm = mat[mat["mkt"] == mkt]["maturity"] if "mkt" in mat.columns else mat["maturity"]
    mm = mm[mm.index.isin(px.columns)]
    if mm.empty:
        print(f"\n{mkt}: nessuna scadenza trovata per gli ISIN di px_mid -- salto")
        return []

    print(f"\n{'='*78}\n{mkt} -- censimento della curva reale\n{'='*78}")
    print(f"  linker in anagrafica con prezzi: {len(mm)}")
    print(f"  scadenze: {', '.join(sorted({d.strftime('%Y') for d in mm}))}")

    # --- per ogni data: quali titoli sono prezzati E vivi (>3 mesi a scadenza, come il fit)
    rows, per_date = [], []
    for dt, row in px.iterrows():
        alive = row.dropna().index
        alive = [i for i in alive if i in mm.index]
        if not alive:
            continue
        ttm = np.array([(mm[i] - dt).days / 365.25 for i in alive])
        ttm = ttm[(ttm > 0.25) & (ttm < 50)]           # stessi filtri del fit (GSW)
        n = len(ttm)
        if n == 0:
            continue
        per_date.append((dt, n, ttm.min() if n else np.nan, ttm.max() if n else np.nan, ttm))

    if not per_date:
        print("  nessuna data utilizzabile")
        return []

    ns_ = np.array([p[1] for p in per_date])
    print(f"\n  titoli utilizzabili per data: media {ns_.mean():.1f} | "
          f"min {ns_.min()} | max {ns_.max()} | date {len(per_date):,}")

    # --- regimi del fit
    skipped = (ns_ < MIN_BONDS).sum()
    ns_only = ((ns_ >= MIN_BONDS) & (ns_ < MIN_BONDS_NSS)).sum()
    full = (ns_ >= MIN_BONDS_NSS).sum()
    dof_ns = ns_[(ns_ >= MIN_BONDS) & (ns_ < MIN_BONDS_NSS)] - N_PAR_NS
    dof_full = ns_[ns_ >= MIN_BONDS_NSS] - N_PAR_NSS
    print(f"\n  regime del fit (soglie del codice: <{MIN_BONDS} salta, "
          f"<{MIN_BONDS_NSS} Nelson-Siegel, >={MIN_BONDS_NSS} NSS piena):")
    print(f"    saltate           {skipped:6,} date ({100*skipped/len(ns_):4.0f}%)")
    print(f"    Nelson-Siegel 4p  {ns_only:6,} date ({100*ns_only/len(ns_):4.0f}%)"
          + (f"  gradi di liberta' medi {dof_ns.mean():+.1f}" if len(dof_ns) else ""))
    print(f"    NSS piena 6p      {full:6,} date ({100*full/len(ns_):4.0f}%)"
          + (f"  gradi di liberta' medi {dof_full.mean():+.1f}" if len(dof_full) else ""))
    if len(dof_ns) and dof_ns.mean() <= 1:
        print("    [!] gradi di liberta' ~0: il fit INTERPOLA i titoli, l'RMSE va a ~0")
        print("        per costruzione e NON e' evidenza di una buona curva")

    # --- bracketing per nodo
    print(f"\n  copertura per nodo (interpolato = il nodo cade fra due titoli vivi):")
    print(f"    {'nodo':>6s} {'interpolato':>12s} {'estrapolato':>12s} {'titoli +-2y (media)':>21s}   verdetto")
    for node in NODES:
        n_in = n_out = 0
        near = []
        for dt, n, tmin, tmax, ttm in per_date:
            if n < MIN_BONDS:
                continue
            if tmin <= node <= tmax:
                n_in += 1
            else:
                n_out += 1
            near.append(int(((ttm >= node - 2) & (ttm <= node + 2)).sum()))
        tot = n_in + n_out
        if tot == 0:
            continue
        share_in = 100 * n_in / tot
        avg_near = np.mean(near) if near else 0.0
        if share_in >= 90 and avg_near >= 2:
            verdict = "utilizzabile"
        elif share_in >= 70 and avg_near >= 1:
            verdict = "FRAGILE"
        else:
            verdict = "NON utilizzabile"
        print(f"    {node:5.0f}y {n_in:11,} {n_out:12,} {avg_near:20.1f}   {verdict}")
        rows.append({"mkt": mkt, "node": node, "dates_interp": n_in, "dates_extrap": n_out,
                     "share_interp_pct": round(share_in, 1), "avg_bonds_pm2y": round(avg_near, 2),
                     "n_bonds_mean": round(ns_.mean(), 2), "verdict": verdict})

    # --- evoluzione nel tempo (quanti titoli per anno)
    per_year = {}
    for dt, n, *_ in per_date:
        per_year.setdefault(dt.year, []).append(n)
    yrs = sorted(per_year)
    print("\n  titoli utilizzabili per anno:")
    line = "    " + "  ".join(f"{y}:{int(np.mean(per_year[y])):2d}" for y in yrs)
    for chunk in [line[i:i + 96] for i in range(0, len(line), 96)]:
        print(chunk)

    # --- RMSE del fit, se in cache (con l'avvertenza sui gradi di liberta')
    for nm in (f"nss_real_{mkt}", f"nss_{mkt}"):
        p = CACHE / f"{nm}.parquet"
        if p.exists():
            par = pd.read_parquet(p)
            if "rmse_bp" in par.columns:
                print(f"\n  fit {nm}: rmse mediano {par['rmse_bp'].median():.2f}bp, "
                      f">12bp in {(par['rmse_bp'] > 12).sum()} date su {len(par)}")
                if len(dof_ns) and dof_ns.mean() <= 1:
                    print("    (rmse basso NON rassicurante qui: gradi di liberta' ~0)")
            break
    return rows


if __name__ == "__main__":
    mat = _maturities()
    if mat is None:
        raise SystemExit("[!] ne' ref_linker ne' universe in cache: impossibile censire")

    print("=" * 78)
    print("CENSIMENTO CURVE REALI -- quali mercati/nodi/periodi sono utilizzabili")
    print("=" * 78)
    print("criterio: NON l'RMSE (che con pochi titoli va a 0 per costruzione), ma i")
    print("gradi di liberta' del fit e il bracketing del nodo che si legge.")

    allrows = []
    for mkt in MKTS:
        allrows += census_market(mkt, mat)

    if allrows:
        df = pd.DataFrame(allrows)
        df.to_csv(CACHE / "an_curve_census.csv", index=False)
        print("\n" + "=" * 78)
        print("SINTESI -- verdetto per mercato e nodo")
        print("=" * 78)
        piv = df.pivot(index="mkt", columns="node", values="verdict")
        print(piv.to_string())
        print("\nletture:")
        print("  utilizzabile     -> il nodo e' interpolato fra titoli veri, >=2 titoli vicini")
        print("  FRAGILE          -> spesso interpolato ma con 1 solo titolo vicino:")
        print("                      il 'nodo' e' di fatto quel singolo titolo")
        print("  NON utilizzabile -> nodo spesso oltre l'ultimo titolo: e' il modello,")
        print("                      non un dato. I risultati su questo nodo vanno ritirati")
        print("                      o rifatti a livello ISIN (linker vs curva NOMINALE).")
        print("\nsalvato: an_curve_census.csv")
