"""curves - curve pubbliche (BoE, Fed/GSW) + toolkit NSS + convenzioni di compounding.

COSA COSTRUISCE
  1. BoE: legge i file "GLC {Nominal|Real|Inflation} daily data_*.xlsx" da data/raw/BoE/**
     (7 blocchi d'anni per curva) e li concatena in pannelli parquet date x maturita':
         boe_{nominal|real|inflation}_{spot|fwd}_{std|short}.parquet
     - std   : curva intera, passi di 0.5 anni (0.5-25y nelle vintage vecchie, fino a 40y nelle nuove)
     - short : short end fittato SEPARATAMENTE dalla BoE, passi mensili 1-60 mesi
       (i punti sovrapposti, es. 1y, possono differire di poco fra i due: fit diversi -> non si fondono)
  2. Fed/GSW: legge feds200628 (nominale), feds200805 (TIPS), feds200533 (Kim-Wright) e salva
     i raw in parquet. Espone gsw_zero_panel() che valuta la curva NSS dai parametri BETA/TAU
     a QUALUNQUE maturita' -- piu' flessibile delle colonne SVENY/TIPSY a maturita' intere.
  3. NSS: nss_yield() (formula Svensson), fit_nss() (least squares con bound, multi-start,
     warm start, fallback Nelson-Siegel), fit_nss_panel() (driver per-data per l'euro).
  4. Compounding: conversioni annuale/semestrale <-> continuo.

CONVENZIONI DI COMPOUNDING (la scelta difendibile)
  Tutti i pannelli di output sono in capitalizzazione CONTINUA, percento.
  Motivo: in continuo la relazione di Fisher e' esattamente additiva (y_nom = y_real + pi),
  quindi la base di curva  lambda = real + ils - nominal  e' un'identita' pulita solo li'.
    - BoE: gia' continua per dichiarazione della BoE (FAQ: "continuously compounded, annual basis").
      VERIFICA EMPIRICA fatta in fondo al modulo: sui dati 2025, max |nom - real - infl| ~ 1e-6
      sulla griglia comune -- l'identita' di Fisher additiva regge alla precisione numerica,
      il che conferma sia il compounding sia la coerenza interna delle tre curve.
    - GSW: le colonne SVENY/TIPSY vengono CONFRONTATE in-code con la formula NSS valutata sui
      parametri pubblicati (check_gsw_compounding). Se coincidono, sono gia' continue (la
      Svensson produce yield cc); se coincidono dopo trasformazione, il codice lo dice.
      Non ci si fida della memoria: si verifica sul file.
    - ILS Bloomberg (zero-coupon, capitalizzazione ANNUALE): convertire con ann_to_cc prima
      di sommarli a curve cc. A ils=4%: ln(1.04)=3.922% -> 7.8bp di errore evitato, che su una
      base di 16-24bp e' quasi meta' del segnale.
  Gli YTM locali (per il fitting euro e per le basi ISIN): convenzioni per mercato in
  YTM_COMP; convertire con ytm_to_cc(mkt, y) PRIMA di confrontare con checchessia.

NOTA revisione BoE 2017 (dal FAQ BoE): nel 2017 la procedura della curva REALE e' passata dal
fitting su prezzi nominal-clean a real-clean, con revisione dei dati 2015-2017. La serie reale
UK ha quindi una rottura DI MISURA in quella finestra: BOE_REAL_REVISION qui sotto serve per
dummizzarla nelle analisi. Non e' un evento di mercato.

USO
    python curves.py            # costruisce tutto e stampa il riepilogo con i sanity check
    from curves import load_panel, gsw_zero_panel, fit_nss_panel, ann_to_cc
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- percorsi
try:
    from config import DATA, CACHE
except Exception:                                   # uso fuori progetto / test
    DATA = Path(__file__).resolve().parent / "data"
    CACHE = DATA / "cache"
RAW = Path(os.environ.get("IB_RAW", DATA / "raw"))  # override per test: IB_RAW=/percorso
CACHE.mkdir(parents=True, exist_ok=True)

BOE_REAL_REVISION = (pd.Timestamp("2015-01-01"), pd.Timestamp("2017-12-31"))

# ----------------------------------------------------------------------------- compounding
def ann_to_cc(y):
    """Zero annualmente composto (percento) -> continuo (percento). Per gli ILS Bloomberg."""
    return 100.0 * np.log1p(np.asarray(y, float) / 100.0)

def cc_to_ann(y):
    return 100.0 * np.expm1(np.asarray(y, float) / 100.0)

def comp_to_cc(y, m: int):
    """Composto m volte l'anno (percento) -> continuo (percento). m=2 per convenzione semestrale."""
    return 100.0 * m * np.log1p(np.asarray(y, float) / (100.0 * m))

# Convenzione di quotazione degli YTM per mercato (per YLD_YTM_MID e per il fitting):
#   IT/FR/DE/annuale (ICMA); UK gilt e US Treasury semestrale.
YTM_COMP = {"IT": 1, "FR": 1, "FR_CPI": 1, "DE": 1, "UK": 2, "US": 2}

def ytm_to_cc(mkt: str, y):
    return comp_to_cc(y, YTM_COMP[mkt])

# ----------------------------------------------------------------------------- BoE parser
_CURVE_TOKEN = {"nominal": "nominal", "real": "real", "inflation": "inflation"}

def _sheet_matches(name: str, measure: str, segment: str) -> bool:
    s = name.lower()
    if measure == "spot":
        ok = "spot" in s
    else:
        ok = ("fwd" in s) or ("forward" in s)
    if segment == "short":
        return ok and "short" in s
    return ok and "short" not in s

def _parse_boe_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Layout BoE: riga con 'years:' in col 0 = maturita'; righe con data in col 0 = osservazioni.
    Le righe 'Refresh' e i festivi tutti-NaN vengono eliminati dalla coercizione."""
    raw = xls.parse(sheet, header=None)
    hdr = None
    for i in range(min(15, len(raw))):
        if str(raw.iat[i, 0]).strip().lower().startswith("year"):
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"riga 'years:' non trovata in {xls.io} [{sheet}]")
    mats = pd.to_numeric(raw.iloc[hdr, 1:], errors="coerce")
    keep = mats.notna().values
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    rows = dates.notna().values
    vals = raw.iloc[rows, 1:].loc[:, keep]
    vals = vals.apply(pd.to_numeric, errors="coerce")
    vals.index = pd.DatetimeIndex(dates[rows])
    vals.columns = mats[mats.notna()].astype(float).round(6).values
    return vals.dropna(how="all")

def build_boe(curve: str, measure: str = "spot", segment: str = "std",
              save: bool = True) -> pd.DataFrame:
    """Concatena tutti i blocchi d'anni di una curva BoE in un pannello date x maturita' (cc, %)."""
    token = _CURVE_TOKEN[curve]
    files = sorted(p for p in (RAW / "BoE").rglob("*.xlsx")
                   if token in p.name.lower() and not p.name.startswith("~"))
    if not files:
        raise FileNotFoundError(f"nessun file BoE per '{curve}' sotto {RAW/'BoE'}")
    panels = []
    for f in files:
        xls = pd.ExcelFile(f)
        try:
            sheet = next(s for s in xls.sheet_names if _sheet_matches(s, measure, segment))
        except StopIteration:
            continue                                   # vintage senza quel foglio
        panels.append(_parse_boe_sheet(xls, sheet))
    out = pd.concat(panels).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.reindex(sorted(out.columns), axis=1)
    if save:
        out.to_parquet(CACHE / f"boe_{curve}_{measure}_{segment}.parquet")
    return out

# ----------------------------------------------------------------------------- Fed / GSW
_FEDS_FILE = {"nominal": "feds200628.csv", "tips": "feds200805.csv", "kw": "feds200533.csv"}

def _read_feds_csv(path: Path) -> pd.DataFrame:
    """I CSV della Fed hanno alcune righe descrittive prima dell'header 'Date,...'."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    hdr = next(i for i, l in enumerate(lines)
               if l.split(",")[0].strip().strip('"').lower() == "date")
    df = pd.read_csv(path, skiprows=hdr, encoding="utf-8-sig")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").apply(pd.to_numeric, errors="coerce").sort_index()
    return df

def build_gsw(kind: str = "nominal", save: bool = True) -> pd.DataFrame:
    df = _read_feds_csv(RAW / "Fed" / _FEDS_FILE[kind])
    if save:
        df.to_parquet(CACHE / f"gsw_{kind}_raw.parquet")
    return df

# ------------------------------------------------------------------------------- NSS
def nss_yield(tau, b0, b1, b2, b3, t1, t2):
    """Svensson (1994), yield zero-coupon in capitalizzazione continua.
    Parametrizzazione identica a GSW (BETA0..BETA3, TAU1, TAU2)."""
    tau = np.asarray(tau, float)
    x1 = tau / t1
    x2 = tau / t2
    f1 = (1.0 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
    return b0 + b1 * f1 + b2 * f2 + b3 * f3

def gsw_zero_panel(kind: str, grid, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Valuta la curva GSW dai parametri pubblicati su una griglia arbitraria di maturita'."""
    if df is None:
        df = build_gsw(kind, save=False)
    cols = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    P = df[cols].dropna()
    out = {float(t): nss_yield(t, P["BETA0"].values, P["BETA1"].values, P["BETA2"].values,
                               P["BETA3"].values, P["TAU1"].values, P["TAU2"].values)
           for t in grid}
    return pd.DataFrame(out, index=P.index)

def check_gsw_compounding(df: pd.DataFrame, kind: str) -> str:
    """Confronta le colonne pubblicate (SVENY/TIPSY) con la formula NSS sui parametri.
    Se coincidono -> gia' continue. Verifica sul file, non sulla memoria."""
    pref = "SVENY" if kind == "nominal" else "TIPSY"
    cands = [c for c in df.columns if re.fullmatch(pref + r"\d{2}", c)]
    if not cands:
        return "colonne yield non trovate: solo parametri; usare gsw_zero_panel (cc per costruzione)"
    c = cands[len(cands) // 2]
    t = float(c[len(pref):])
    P = df[["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2", c]].dropna()
    model = nss_yield(t, *(P[k].values for k in
                           ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]))
    d_cc = np.nanmax(np.abs(model - P[c].values))
    d_ann = np.nanmax(np.abs(cc_to_ann(model) - P[c].values))
    if d_cc < 1e-3:
        return f"{c}: coincide con la formula (max diff {d_cc:.2e}) -> yield CONTINUI"
    if d_ann < 1e-3:
        return f"{c}: coincide dopo cc->ann (max diff {d_ann:.2e}) -> yield ANNUALI, convertire"
    return f"{c}: non riconciliato (cc {d_cc:.3g}, ann {d_ann:.3g}) -> ispezionare"

# ------------------------------------------------------------------ fitting NSS (euro)
CANONICAL_STARTS = [
    np.array([4.0, -1.0, 0.0, 0.0, 1.5, 10.0]),
    np.array([3.0, 0.0, -2.0, 2.0, 0.8, 5.0]),
    np.array([2.0, 1.5, 1.0, -1.0, 2.5, 12.0]),
]
# bound larghi: l'euro ha avuto tassi negativi -> nessun vincolo di positivita' sui livelli
NSS_BOUNDS = (np.array([-5.0, -20.0, -50.0, -50.0, 0.05, 1.0]),
              np.array([15.0, 20.0, 50.0, 50.0, 8.0, 30.0]))

def fit_nss(mats, y_cc, weights=None, x0=None, ns_only: bool = False):
    """Fit ai minimi quadrati degli yield cc. Equivalente all'obiettivo GSW (errori di prezzo
    pesati 1/duration ~ errori di yield, come GSW stessi dichiarano). Multi-start + warm start.
    ns_only=True fissa beta3=0 (Nelson-Siegel): da usare quando i bond alla data sono pochi."""
    from scipy.optimize import least_squares
    mats = np.asarray(mats, float)
    y = np.asarray(y_cc, float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)

    def resid(p):
        b0, b1, b2, b3, t1, t2 = p
        if ns_only:
            b3 = 0.0
        return w * (nss_yield(mats, b0, b1, b2, b3, t1, t2) - y)

    best = None
    starts = ([np.asarray(x0, float)] if x0 is not None else []) + CANONICAL_STARTS
    for s in starts:
        try:
            r = least_squares(resid, s, bounds=NSS_BOUNDS, method="trf", max_nfev=4000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            continue
    return best

def fit_nss_panel(bonds: pd.DataFrame, min_bonds_nss: int = 8,
                  min_bonds: int = 4) -> pd.DataFrame:
    """Driver per-data. bonds: colonne [date, tau, y_cc] (+ opzionale weight).
    Ritorna parametri per data + diagnostica (n bond, RMSE bp, modello usato).
    Warm start dal giorno precedente; NS quando n < min_bonds_nss; salta se n < min_bonds."""
    rows, prev = [], None
    for dt, g in bonds.groupby("date"):
        g = g.dropna(subset=["tau", "y_cc"])
        n = len(g)
        if n < min_bonds:
            continue
        ns = n < min_bonds_nss
        w = g["weight"].values if "weight" in g else None
        r = fit_nss(g["tau"].values, g["y_cc"].values, weights=w, x0=prev, ns_only=ns)
        if r is None:
            continue
        prev = r.x
        fit = nss_yield(g["tau"].values, *(list(r.x[:3]) + [0.0 if ns else r.x[3]] + list(r.x[4:])))
        rmse = float(np.sqrt(np.mean((fit - g["y_cc"].values) ** 2))) * 100.0   # bp
        rows.append([dt, *r.x, n, rmse, "NS" if ns else "NSS"])
    return pd.DataFrame(rows, columns=["date", "b0", "b1", "b2", "b3", "t1", "t2",
                                       "n_bonds", "rmse_bp", "model"]).set_index("date")



# ------------------------------------------------------------- Bundesbank (Germania)
# La Bundesbank pubblica DAL 1997 i parametri Svensson giornalieri della curva dei titoli
# federali quotati (metodo documentato: Monthly Report ott-1997, Schich; BIS Paper 25) e i
# rendimenti PAR derivati a scadenze fisse (serie ZAR, cedola annuale). E' la curva nominale
# tedesca gold standard: stessa famiglia Svensson di GSW, istituzionale, replicabile.
#
# ATTENZIONE ALLE ETICHETTE: la tabella di corrispondenza ufficiale (WT3202-07 -> chiavi
# BBSIS ZST.B1/B2/B3[/B4]/T1/T2) e' AMBIGUA sull'accoppiamento etichetta->parametro.
# Questo parser NON si fida delle etichette: identifica i ruoli ripricando i PAR yield
# pubblicati (ZAR) sotto tutte le mappature plausibili e sotto entrambe le ipotesi di
# compounding degli zeri (continuo / annuale), e sceglie la combinazione a RMSE minimo.
# Il verdetto (mappatura, compounding, RMSE in bp) viene stampato e salvato: se nessuna
# combinazione riprezza bene, il parser si RIFIUTA con diagnostica, non tira a indovinare.
BUBA_FLOW = "BBSIS"
_BUBA_PARAM_KEY = "D.I.ZST.{p}.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A"
_BUBA_PAR_KEY = "D.I.ZAR.ZI.EUR.S1311.B.A604.R{m}XX.R.A.A._Z._Z.A"
# I SEI parametri Svensson della Bundesbank: B0,B1,B2,B3,T1,T2 (NON B4 -- non esiste).
# Le etichette del portale sono esplicite: "Parameter Beta0..Beta3", "Parameter Tau1/Tau2".
BUBA_PARAM_LABELS = ["B0", "B1", "B2", "B3", "T1", "T2"]
BUBA_PAR_TENORS = [2, 5, 10, 30]
RAW_BUBA = None  # risolto a runtime: RAW/"Bundesbank"


def download_bundesbank(start: str = "1997-08-01") -> None:
    """Scarica via API SDMX i sei parametri + i par yield di controllo in RAW/Bundesbank.
    Non richiede il terminale Bloomberg. Se l'API cambia formato, scaricare a mano dal
    portale (time series database, flow BBSIS) le serie ZST.B*/T* e ZAR R02/05/10/30XX."""
    import requests
    out = RAW / "Bundesbank"
    out.mkdir(parents=True, exist_ok=True)
    keys = [_BUBA_PARAM_KEY.format(p=p) for p in BUBA_PARAM_LABELS]
    keys += [_BUBA_PAR_KEY.format(m=f"{m:02d}") for m in BUBA_PAR_TENORS]
    base = "https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
    for k in keys:
        dest = out / f"BBSIS_{k.replace('.', '_')}.csv"
        ok = False
        for params in ({"format": "csv", "lang": "en"}, {"format": "csv"}, {}):
            try:
                r = requests.get(base + k, params=params,
                                 headers={"Accept": "text/csv"}, timeout=60)
                if r.ok and any(ch.isdigit() for ch in r.text[:2000]):
                    dest.write_text(r.text, encoding="utf-8")
                    ok = True
                    break
            except Exception:
                continue
        print(f"  {'ok ' if ok else 'FALLITO'} {k}")
        if not ok:
            print("    -> scaricare a mano dal portale Bundesbank (flow BBSIS) in "
                  f"{out}")


def _read_buba_csv(path) -> pd.Series:
    """Lettore tollerante: API (virgola come separatore, punto decimale) e portale (punto
    e virgola come separatore, VIRGOLA decimale, righe di metadati). Il separatore viene
    rilevato riga per riga, cosi' il decimale non viene mai spezzato."""
    vals, idx = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if ";" in line:
            parts, dec = line.split(";"), ","
        else:
            parts, dec = line.split(","), "."
        if not parts:
            continue
        d = pd.to_datetime(parts[0].strip().strip('"'), format="%Y-%m-%d", errors="coerce")
        if pd.isna(d):
            continue
        for cell in parts[1:]:
            s = cell.strip().strip('"')
            if dec == ",":
                s = s.replace(".", "").replace(",", ".")
            try:
                v = float(s)
            except ValueError:
                continue
            idx.append(d)
            vals.append(v)
            break
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


def _buba_scan(raw_dir) -> tuple[dict, dict]:
    """Associa ogni CSV alla sua serie cercando la CHIAVE nel nome file o nel contenuto."""
    import re
    params, pars = {}, {}
    for f in sorted(Path(raw_dir).glob("*.csv")):
        head = f.name + " " + Path(f).read_text(encoding="utf-8", errors="replace")[:4000]
        mp = re.search(r"ZST[._](B[0-3]|T[12])", head)
        mz = re.search(r"ZAR.*?R(\d{2})XX", head, re.S)
        if mp:
            params[mp.group(1)] = _read_buba_csv(f)
        elif mz:
            pars[float(int(mz.group(1)))] = _read_buba_csv(f)
    return params, pars


def _par_from_zero(z_pct: np.ndarray, mats: np.ndarray, comp: str) -> np.ndarray:
    """Par yield (cedola annuale) da una curva zero. z_pct: matrice date x anni interi
    1..M. comp: 'cc' (df=exp(-z t)) o 'ann' (df=(1+z)^-t). Ritorna percento."""
    t = np.arange(1, z_pct.shape[1] + 1, dtype=float)
    df = np.exp(-z_pct / 100.0 * t) if comp == "cc" else (1.0 + z_pct / 100.0) ** (-t)
    out = np.empty((z_pct.shape[0], len(mats)))
    for j, m in enumerate(mats):
        k = int(m)
        out[:, j] = 100.0 * (1.0 - df[:, k - 1]) / df[:, :k].sum(axis=1)
    return out


def build_bundesbank(save: bool = True, n_check: int = 250):
    """Carica i parametri, identifica ruoli e compounding col repricing dei par yield,
    salva buba_params.parquet (colonne b0..t2 nei ruoli GIUSTI) + buba_meta.parquet."""
    raw_dir = RAW / "Bundesbank"
    params, pars = _buba_scan(raw_dir)
    if len(params) < 6:
        raise FileNotFoundError(
            f"parametri Bundesbank incompleti in {raw_dir}: trovate {sorted(params)} "
            f"(servono 6 serie ZST). Lancia download_bundesbank() o scarica dal portale.")
    if not pars:
        raise FileNotFoundError(
            f"nessuna serie par ZAR in {raw_dir}: serve almeno R10XX per il check.")

    labels = sorted(params)
    P = pd.DataFrame(params).dropna()
    Y = pd.DataFrame(pars).dropna()
    common = P.index.intersection(Y.index)
    step = max(1, len(common) // n_check)
    sample = common[::step]

    # Etichette Bundesbank ESPLICITE (portale: "Parameter Beta0..Beta3, Tau1, Tau2"):
    # B0->b0, B1->b1, B2->b2, B3->b3, T1->t1, T2->t2. Mappatura CERTA. Il repricing dei
    # par ZAR resta come check del solo compounding (cc vs ann) e come sanity del segno.
    need = {"B0", "B1", "B2", "B3", "T1", "T2"}
    if not need.issubset(set(labels)):
        raise ValueError(f"etichette Bundesbank inattese: {sorted(labels)} (servono {sorted(need)})")
    cands = [{"b0": "B0", "b1": "B1", "b2": "B2", "b3": "B3", "t1": "T1", "t2": "T2"}]

    mats = np.array(sorted(pars))
    results = []
    for mi, mp in enumerate(cands):
        cols = {r: P[mp[r]].loc[sample].values for r in ("b0", "b1", "b2", "b3", "t1", "t2")}
        if (cols["t1"] <= 0).mean() > 0.01 or (cols["t2"] <= 0).mean() > 0.01:
            continue                       # i tau devono essere positivi
        grid = np.arange(1.0, mats.max() + 1.0)
        z = np.column_stack([nss_yield(t, cols["b0"], cols["b1"], cols["b2"],
                                       cols["b3"], cols["t1"], cols["t2"]) for t in grid])
        for comp in ("cc", "ann"):
            model = _par_from_zero(z, mats, comp)
            obs = Y[sorted(pars)].loc[sample].values
            rmse = float(np.sqrt(np.nanmean((model - obs) ** 2))) * 100.0   # bp
            results.append((rmse, mi, comp, mp))
    if not results:
        raise ValueError("nessuna mappatura ammissibile (tau non positivi ovunque)")
    results.sort()
    best = results[0]
    print("  identificazione parametri Bundesbank (repricing par ZAR, "
          f"{len(sample)} date, tenor {list(map(int, mats))}):")
    for rmse, mi, comp, mp in results[:4]:
        tag = " <-- SCELTA" if (rmse, mi, comp) == best[:3] else ""
        print(f"    mappatura#{mi} comp={comp}: RMSE {rmse:7.2f} bp{tag}")
    if best[0] > 5.0:
        raise ValueError(f"miglior repricing RMSE {best[0]:.2f} bp > 5: qualcosa non torna "
                         "-- ispezionare i file scaricati prima di usare la curva.")
    _, _, comp, mp = best
    out = pd.DataFrame({r: P[mp[r]] for r in ("b0", "b1", "b2", "b3", "t1", "t2")}).dropna()
    meta = pd.DataFrame([{"comp": comp, "rmse_bp": best[0],
                          **{f"lab_{r}": mp[r] for r in mp}}])
    if save:
        out.to_parquet(CACHE / "buba_params.parquet")
        meta.to_parquet(CACHE / "buba_meta.parquet")
    return out, comp


def buba_zero_panel(grid, params: pd.DataFrame | None = None,
                    comp: str | None = None) -> pd.DataFrame:
    """Curva zero tedesca dai parametri Bundesbank, in CC percento (convertita se il
    verdetto del check e' 'ann')."""
    if params is None:
        params = pd.read_parquet(CACHE / "buba_params.parquet")
        comp = pd.read_parquet(CACHE / "buba_meta.parquet")["comp"].iloc[0]
    out = {float(t): nss_yield(t, params["b0"].values, params["b1"].values,
                               params["b2"].values, params["b3"].values,
                               params["t1"].values, params["t2"].values) for t in grid}
    panel = pd.DataFrame(out, index=params.index)
    return ann_to_cc(panel) if comp == "ann" else panel


# ------------------------------------------------------------------------------ utilita'
def load_panel(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"{name}.parquet")

def interp_maturities(panel: pd.DataFrame, taus) -> pd.DataFrame:
    """Interpola linearmente in maturita' (per riga) sulla griglia richiesta; solo interno."""
    cols = sorted(set(map(float, panel.columns)) | set(map(float, taus)))
    wide = panel.reindex(columns=cols)
    wide = wide.interpolate(axis=1, method="index", limit_area="inside")
    return wide[list(map(float, taus))]

def fisher_check(nom: pd.DataFrame, real: pd.DataFrame, infl: pd.DataFrame) -> float:
    """max |nominale - reale - inflazione| sulla griglia comune: deve essere ~0 se tutto e' cc."""
    cols = sorted(set(nom.columns) & set(real.columns) & set(infl.columns))
    idx = nom.index.intersection(real.index).intersection(infl.index)
    d = (nom.loc[idx, cols] - real.loc[idx, cols] - infl.loc[idx, cols]).abs()
    return float(np.nanmax(d.values))

# ------------------------------------------------------------------------------ driver
def build_all(verbose: bool = True) -> None:
    P = print if verbose else (lambda *a, **k: None)
    P("=== BoE ===")
    boe = {}
    for curve in ("nominal", "real", "inflation"):
        for measure in ("spot", "fwd"):
            for segment in ("std", "short"):
                try:
                    df = build_boe(curve, measure, segment)
                    boe[(curve, measure, segment)] = df
                    P(f"  boe_{curve}_{measure}_{segment:5s}: {df.shape[0]:6d} date x "
                      f"{df.shape[1]:3d} mat | {df.index.min().date()} -> {df.index.max().date()}")
                except FileNotFoundError as e:
                    P(f"  boe_{curve}_{measure}_{segment}: MANCANTE ({e})")
    if all(k in boe for k in [("nominal", "spot", "std"), ("real", "spot", "std"),
                              ("inflation", "spot", "std")]):
        dev = fisher_check(boe[("nominal", "spot", "std")], boe[("real", "spot", "std")],
                           boe[("inflation", "spot", "std")])
        P(f"  Fisher additiva (nom - real - infl), max |dev| = {dev:.2e} punti pct"
          f"  -> {'OK: curve cc e coerenti' if dev < 1e-3 else 'ATTENZIONE: ispezionare'}")

    P("=== Fed / GSW ===")
    for kind in ("nominal", "tips", "kw"):
        try:
            df = build_gsw(kind)
            P(f"  gsw_{kind}: {df.shape[0]} date x {df.shape[1]} col | "
              f"{df.index.min().date()} -> {df.index.max().date()}")
            if kind in ("nominal", "tips"):
                P(f"    compounding: {check_gsw_compounding(df, kind)}")
        except FileNotFoundError:
            P(f"  gsw_{kind}: file mancante sotto {RAW/'Fed'}")

if __name__ == "__main__":
    build_all()
