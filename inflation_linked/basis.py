"""basis - flussi del linker, IRR, matching dei nominali, i tre metodi della base, floor.

FEDELTA'. Le convenzioni di calcolo sono REPLICATE dall'originale (btp.py, classe BTPei):
  - coefficiente di indicizzazione: CI = round(int(x*1e6)/1e6, 5) -- troncamento a 6 decimali
    poi arrotondamento a 5, con max(x,1) se floor=True. E' la convenzione MEF, quirk incluso.
  - primo flusso: -(prezzo_clean * CI + rateo_indicizzato), con CI al REGOLAMENTO (T+n,
    festivi di mercato) calcolato sugli ACTUAL (non sulla proiezione), formula MEF giornaliera.
  - rateo: cpn/100 * frazione * (settle-last)/(next-last), con i rami del primo periodo
    irregolare ('Short First', 'Long First', doppi irregolari via FIRST/SECOND_CPN_DT).
  - cedole future: CI dalla proiezione ('CPI SA' via InflationEngine.reference_index),
    cedola arrotondata al centesimo; a scadenza cedola + max(nominale*CI, 100) (par floor).
  - date cedola aggiustate al business day SEGUENTE col calendario del paese; l'originale
    usa per il regolamento il calendario esteso (festivi extra) e per l'aggiustamento cedole
    quello base: quirk preservato (settle_cal vs cpn_cal).
  - IRR: tempo (d - trade_date)/365.25, composto ANNUALE, radice di sum cf/(1+y)^t; dal
    PENULTIMO stacco in poi rendimento monetario semplice ACT/365 (ramo speciale originale).
    Solver: brentq con bracket espandibile (stessa radice di fsolve, robusto e piu' veloce).
  - matching nominali: lower/upper per scadenza, filtro First Settle < data, interpolazione
    degli YTM pesata per distanza in giorni ('interp', la misura dell'originale) oppure il
    singolo piu' vicino ('nearest', l'oggetto FLL tradabile, tolleranza MAX_MISMATCH_DAYS).

I TRE METODI
  B  (bond vs bond)  : lambda = TotalYtM - y_nominale(match), in bp. 'interp' o 'nearest'.
  C-semplice         : e' il B con 'interp' (l'originale). Mantenuto come nome per chiarezza.
  C-esatto           : i flussi nominali sintetici del linker vengono SCONTATI sulla curva
                       zero nominale (cc, %) -> prezzo sintetico -> IRR sugli STESSI flussi;
                       lambda = IRR_osservato - IRR_sintetico. L'effetto cedola si cancella
                       esattamente e il mismatch di scadenza e' zero per costruzione.
                       Identita' di controllo (testata): su curva piatta y_cc, l'IRR
                       sintetico = e^{y_cc} - 1 per QUALUNQUE insieme di flussi.

DEVIAZIONI DICHIARATE (tutte spente in modalita' regressione):
  - ytm_convention='local' (default) sottrae YLD_YTM_MID cosi' com'e' (convenzione locale:
    annuale IT/FR/DE, SEMESTRALE UK/US). 'annual' converte UK/US a composto annuale prima
    della differenza: corretto per il cross-market, ma non e' cio' che faceva l'originale.
  - settle_days per mercato: IT/FR/DE=2 (originale IT=2), UK=1, US=1.
  - exclude_tail: opzionale per il paper (ultimi 12m linker / 6m nominale); default False.
  - la gamba nominale usa lo YTM Bloomberg (convenzione street, regolamento proprio) contro
    un IRR calcolato dal trade date: micro-sfasatura di convenzione EREDITATA dall'originale.

value_floor: port esatto del pricing originale (floorlet Black su ratio, vol lognormale dai
log-rendimenti mensili 3 anni con lambda=1, fixing il 15 del mese precedente lo stacco,
primo stacco escluso se il fixing e' passato). E' la valutazione del floor in bp che serve
alla decomposizione base = floor + residuo.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

try:
    from config import MARKETS, YEARS_FORWARD, CACHE, REGRESSION_TOL_BP
except Exception:
    MARKETS, YEARS_FORWARD, CACHE, REGRESSION_TOL_BP = {}, list(range(1, 11)), None, 0.01

SETTLE_DAYS = {"IT": 2, "FR": 2, "FR_CPI": 2, "DE": 2, "UK": 1, "US": 1}
MAX_MISMATCH_DAYS = 183


# ------------------------------------------------------------------ convenzioni base
def _trunc6_round5(x: float) -> float:
    """Troncamento alla 6a cifra decimale + arrotondamento alla 5a, in aritmetica
    DECIMALE half-up: e' la convenzione MEF alla lettera.
    Perche' non basta round(): sui casi di confine il float sbaglia in modo sistematico.
    Es. 1.033175 -> MEF da' 1.03318, ma il float e' memorizzato come 1.03317499999...
    e round(...,5) restituisce 1.03317 (errore di -1e-05)."""
    d = Decimal(repr(float(x)))
    d6 = d.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    return float(d6.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))


def ci_round(x: float, floor: bool, mode: str = "mef") -> float:
    """CI ufficiale. 'mef' = troncamento a 6 decimali poi arrotondamento a 5 in decimale
    (convenzione MEF, Italia). 'round5' = arrotondamento a 5 decimali: convenzione
    dell'index ratio per UK DMO, US Treasury, AFT, Finanzagentur."""
    v = max(x, 1.0) if floor else x
    if mode == "mef":
        return _trunc6_round5(v)
    return float(Decimal(repr(float(v))).quantize(Decimal("0.00001"),
                                                  rounding=ROUND_HALF_UP))


def rebase_base_cpi(ref: "pd.DataFrame", cpi_series: "pd.Series",
                    lag: int = 3, tol: float = 0.02, verbose: bool = True):
    """Corregge il BASE_CPI dei linker il cui indice di riferimento all'emissione e'
    'congelato' in una base Eurostat vecchia rispetto alla serie CPI corrente.

    Eurostat ribasa l'IAPC periodicamente (2005->2015 a mar-2016, 2015->2025 a mar-2026).
    Per i bond VIVI al ribasamento Bloomberg aggiorna sia il BASE_CPI sia la serie; per i
    bond SCADUTI prima di un ribasamento il BASE_CPI resta nella vecchia base, mentre la
    serie scaricata oggi e' nella base nuova -> il CI (rapporto IR/base) e' sfasato del
    fattore di ribasamento. Il CI e' un rapporto: dev'essere invariante alla base, quindi
    numeratore e denominatore VANNO nella stessa base.

    Rilevazione auto-adattante: si confronta BASE_CPI con l'indice IR interpolato (formula
    MEF) alla data di godimento (START_ACC_DT). Se il rapporto BASE_CPI/IR(godimento) e'
    ~1 il bond e' coerente (nessuna azione); se se ne discosta oltre 'tol', il BASE_CPI e'
    in base vecchia e viene riportato alla base corrente moltiplicandolo per la chiave
    IR(godimento)/BASE_CPI. La chiave risultante e' identica per tutti i bond della stessa
    base vecchia: e' la verifica interna della correzione.

    Ritorna (serie base_cpi corretta, dataframe diagnostico). NON scrive nulla.
    """
    import pandas as _pd
    s = cpi_series.copy()
    s.index = _pd.to_datetime(s.index) + _pd.offsets.MonthEnd(0)

    def _acc_date(row):
        for c in ("START_ACC_DT", "ISSUE_DT", "FIRST_SETTLE_DT", "start_acc_dt"):
            if c in row.index and _pd.notna(row[c]):
                try:
                    return _pd.Timestamp(row[c]).date()
                except Exception:
                    pass
        return None

    base_col = "base_cpi_final" if "base_cpi_final" in ref.columns else "BASE_CPI"
    out, diag = {}, []
    for isin, r in ref.iterrows():
        base = r.get(base_col)
        try:
            base = float(base)
        except Exception:
            out[isin] = base
            continue
        d = _acc_date(r)
        if d is None or base <= 0:
            out[isin] = base
            diag.append((isin, base, None, None, base, "no-date"))
            continue
        try:
            ir = mef_reference(s, d, lag=lag, interpolate=True)
        except Exception:
            out[isin] = base
            diag.append((isin, base, None, None, base, "no-index"))
            continue
        ratio = base / ir if ir else None
        # candidato al ribasamento solo se lo scostamento e' SOSTANZIALE (>~15%): i
        # ribasamenti IAPC danno chiavi ~0.78 (2015->2025) o ~0.66 (2005->2025 doppio),
        # cioe' scostamenti >20%. Uno scostamento piccolo (<10%) e' rumore di
        # interpolazione/settlement, NON un cambio base -> si lascia il base_cpi.
        if ratio and abs(ratio - 1.0) > max(tol, 0.15):
            key = ir / base                       # chiave di ribasamento (base_new/base_old)
            base_corr = base * key                # == ir alla data di godimento
            out[isin] = base_corr
            diag.append((isin, base, ir, key, base_corr, "REBASED"))
        else:
            out[isin] = base
            diag.append((isin, base, ir, 1.0, base, "ok"))

    dd = _pd.DataFrame(diag, columns=["isin", "base_orig", "ir_accrual",
                                      "rebase_key", "base_corr", "status"]).set_index("isin")
    if verbose:
        reb = dd[dd["status"] == "REBASED"]
        if len(reb):
            keys = reb["rebase_key"].round(6)
            print(f"  ribasamento IAPC: {len(reb)} linker corretti "
                  f"(BASE_CPI congelato in base vecchia).")
            print(f"    chiavi di ribasamento applicate: {sorted(keys.unique())}")
            print(f"    (identiche = correzione consistente; ISIN: {list(reb.index)})")
        else:
            print("  ribasamento IAPC: nessun linker da correggere (tutti coerenti).")
    return _pd.Series(out, name="base_cpi_final"), dd


def settlement(trade: date, n_days: int, hol: dict) -> date:
    d, remaining = trade, n_days
    while remaining:
        d = d + relativedelta(days=1)
        if d.weekday() <= 4 and d not in hol:
            remaining -= 1
    return d


def next_business_day(d: date, hol: dict) -> date:
    while d.weekday() >= 5 or d in hol:
        d = d + relativedelta(days=1)
    return d


def mef_reference(series: pd.Series, ref_date: date, lag: int = 3,
                  interpolate: bool = True) -> float:
    """Formula MEF su una serie EOM (usata per il regolamento sugli ACTUAL)."""
    rd = pd.Timestamp(ref_date)
    m_l = rd.replace(day=1) - relativedelta(months=lag) + pd.offsets.MonthEnd(0)
    if not interpolate:
        return float(series.loc[m_l])
    m_l1 = rd.replace(day=1) - relativedelta(months=lag - 1) + pd.offsets.MonthEnd(0)
    dim = (rd + pd.offsets.MonthEnd(0))
    dim = ((rd + relativedelta(months=1)).replace(day=1) - relativedelta(days=1)).day
    w = (rd.day - 1) / dim
    return float(series.loc[m_l]) + w * (float(series.loc[m_l1]) - float(series.loc[m_l]))


# ------------------------------------------------------------------ il bond
_IRREG_DOUBLE = ["Long First and Second", "Short First and Second",
                 "Short First Long Second", "Long First Short Second"]


@dataclass
class LinkerBond:
    isin: str
    mkt: str
    cpn_rate: float                 # percento
    cpn_freq: int
    start_acc: date
    maturity: date
    base_cpi: float
    first_cpn_period_type: str = "Normal"
    first_cpn: date | None = None
    second_cpn: date | None = None
    lag: int = 3
    interpolate: bool = True        # False per i gilt old-style (lag 8)
    style: str = "rc"               # 'rc' real-clean | 'nc' nominal-clean (gilt old-style:
                                    # inflazione maturata DENTRO il prezzo, rateo in moneta)
    ci_mode: str = "mef"            # 'mef' Italia (quirk regressione) | 'round5' altri
    cpn_dp: int = 2                 # arrotondamento cedola: 2 originale IT; DMO 6 new / 4 old
    ex_div_bdays: int = 0           # gilt: 7 business day ex-dividend (entrambi gli stili)
    nominal: float = 100.0
    _sched: list = field(default=None, repr=False)

    @classmethod
    def from_ref(cls, isin: str, row: pd.Series) -> "LinkerBond":
        """Costruisce dal ref_linker.parquet di bbg.py (colonne bdp in maiuscolo)."""
        def _d(x):
            return pd.Timestamp(x).date() if pd.notna(x) else None
        mkt = row["mkt"]
        seg = row.get("segment")
        old_style = (mkt == "UK" and seg == "old")
        return cls(isin=isin, mkt=mkt, cpn_rate=float(row["CPN"]),
                   cpn_freq=int(float(row["CPN_FREQ"])), start_acc=_d(row["START_ACC_DT"]),
                   maturity=_d(row["MATURITY"]),
                   base_cpi=float(row.get("base_cpi_final", row.get("BASE_CPI"))),
                   first_cpn_period_type=str(row.get("FIRST_CPN_PERIOD_TYP", "Normal")),
                   first_cpn=_d(row.get("FIRST_CPN_DT")),
                   second_cpn=_d(row.get("SECOND_CPN_DT")),
                   lag=8 if old_style else 3, interpolate=not old_style,
                   style="nc" if old_style else "rc",
                   ci_mode="mef" if mkt == "IT" else "round5",
                   cpn_dp=(4 if old_style else 6) if mkt == "UK" else 2,
                   ex_div_bdays=7 if mkt == "UK" else 0)

    @property
    def cpn_dates(self) -> list:
        """Schedule cedolare: port esatto di generate_cpn_dates."""
        if self._sched is not None:
            return self._sched
        out, step = [], 12 // self.cpn_freq
        t = self.first_cpn_period_type
        if t == "Normal":
            cur = self.start_acc + relativedelta(months=step)
        elif t in ("Short First", "Long First"):
            cur = self.first_cpn
        elif t in _IRREG_DOUBLE:
            out.append(self.first_cpn)
            cur = self.second_cpn
        else:
            cur = self.start_acc + relativedelta(months=step)
        while cur <= self.maturity:
            out.append(cur)
            cur = cur + relativedelta(months=step)
        self._sched = out
        return out

    # -------------------------------------------------------------- flussi
    def cashflows(self, asof: date, engine, price_clean: float,
                  settle_cal: dict, cpn_cal: dict,
                  is_total: bool = True, floor: bool = True) -> pd.DataFrame:
        """Port di compute_cashflows_ytm: prima riga -(dirty), poi cedole proiettate."""
        future = [d for d in self.cpn_dates if d > asof]
        if not future:
            raise ValueError(f"{self.isin}: nessuna cedola futura ad asof {asof}")
        last = max([d for d in self.cpn_dates if d <= asof], default=None)

        settle = settlement(asof, SETTLE_DAYS.get(self.mkt, 2), settle_cal)

        # ex-dividend (gilt, entrambi gli stili): dal 7o business day prima dello stacco
        # il compratore NON riceve la cedola: rateo negativo e cedola esclusa dai flussi.
        exdiv = False
        if self.ex_div_bdays:
            xd, k = future[0], self.ex_div_bdays
            while k:
                xd = xd - relativedelta(days=1)
                if xd.weekday() <= 4 and xd not in cpn_cal:
                    k -= 1
            exdiv = settle >= xd

        actual = engine._visible(asof)

        # frazione di cedola del periodo corrente (rami irregolari, port esatto)
        t = self.first_cpn_period_type
        if t in ("Short First", "Long First") and last is None:
            frac = (self.first_cpn - self.start_acc).days / 365
        elif t in _IRREG_DOUBLE:
            if last is None:
                frac = (self.first_cpn - self.start_acc).days / 365
            elif last == self.first_cpn:
                frac = (self.second_cpn - self.first_cpn).days / 365
            else:
                frac = 1 / self.cpn_freq
        else:
            frac = 1 / self.cpn_freq

        nxt = future[0]
        if last is None:
            num, den = (settle - self.start_acc).days, (nxt - self.start_acc).days
        else:
            num, den = (settle - last).days, (nxt - last).days
        if exdiv:
            num = -(nxt - settle).days          # rateo negativo nel periodo ex-dividend
        acc = self.cpn_rate / 100 * frac * num / den

        if self.style == "nc":
            # OLD-STYLE (nominal-clean): l'inflazione maturata e' gia' NEL prezzo quotato;
            # il rateo e' in moneta sulla cedola nota: RPI(m-8 dello stacco), pubblicato.
            ref_nxt = engine.reference_index(nxt, asof, self.lag, self.interpolate, "total")
            ci_nxt = ci_round(ref_nxt / self.base_cpi, floor, self.ci_mode)
            cfs = [-(price_clean + acc * ci_nxt * self.nominal)]
        else:
            # REAL-CLEAN (default): dirty = clean x CI(settle) + rateo reale x CI(settle)
            ref_settle = mef_reference(actual, settle, self.lag, self.interpolate)
            ci0 = ci_round(ref_settle / self.base_cpi, floor, self.ci_mode)
            cfs = [-(price_clean * ci0 + ci0 * acc * self.nominal)]

        if exdiv:
            future = future[1:]
            if not future:
                raise ValueError(f"{self.isin}: ex-dividend oltre l'ultima cedola")

        kind = "total" if is_total else "real"
        for cpn_date in future:
            ref = engine.reference_index(cpn_date, asof, self.lag, self.interpolate, kind)
            ci = ci_round(ref / self.base_cpi, floor, self.ci_mode)
            if last is None and t in _IRREG_DOUBLE:
                frac_k = (self.second_cpn - self.first_cpn).days / 365
            else:
                frac_k = 1 / self.cpn_freq
            ced = round(self.cpn_rate / 100 * frac_k * self.nominal * ci, self.cpn_dp)
            cfs.append(ced + max(self.nominal * ci, 100.0) if cpn_date == self.maturity else ced)

        dates = [settle] + [next_business_day(d, cpn_cal) for d in future]
        return pd.DataFrame({"Cashflows": cfs}, index=dates)

    # -------------------------------------------------------------- IRR
    def irr(self, cf: pd.DataFrame, asof: date) -> float:
        """Port di get_single_ytm: annuale, t=(d-asof)/365.25; ramo money-market dal
        penultimo stacco (ACT/365). Ritorna PERCENTO."""
        tt = np.array([(d - asof).days / 365.25 for d in cf.index])
        c = cf["Cashflows"].values.astype(float)
        if len(self.cpn_dates) >= 2 and asof >= self.cpn_dates[-2]:
            return (abs(c[-1] / c[0]) - 1) * 365 / (cf.index[-1] - cf.index[0]).days * 100
        return _solve_irr(c, tt) * 100


def _solve_irr(c: np.ndarray, t: np.ndarray) -> float:
    f = lambda y: float(np.sum(c / (1 + y) ** t))
    lo, hi = -0.60, 1.00
    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0 and tries < 6:          # bracket espandibile
        lo, hi = lo * 0.5 - 0.15, hi * 2
        flo, fhi = f(lo), f(hi)
        tries += 1
    if flo * fhi > 0:
        raise ValueError("IRR: bracket non trovato")
    return brentq(f, lo, hi, xtol=1e-12)


# ------------------------------------------------------------------ matching nominali
def match_nominals(nom_ref: pd.DataFrame, target_maturity: date, asof: date) -> dict:
    """Port di find_closest_cusips_per_date + tolleranza per il 'nearest'.
    nom_ref: index=isin, colonne almeno ['MATURITY','FIRST_SETTLE_DT','AMT_OUTSTANDING']."""
    df = nom_ref.copy()
    df = df[pd.to_datetime(df["FIRST_SETTLE_DT"]) < pd.Timestamp(asof)]
    df = df[pd.to_datetime(df["MATURITY"]) > pd.Timestamp(asof)]
    tm = pd.Timestamp(target_maturity)
    df["mat"] = pd.to_datetime(df["MATURITY"])
    lower = df[df["mat"] < tm].copy()
    upper = df[df["mat"] > tm].copy()
    exact = df[df["mat"] == tm]
    out = {"lower": None, "upper": None, "nearest": None}
    if len(exact):
        e = exact.sort_values("AMT_OUTSTANDING", ascending=False).index[0]
        out.update(lower=e, upper=e, nearest=e, dist_nearest=0)
        return out
    if len(lower):
        lower["d"] = (tm - lower["mat"]).dt.days
        out["lower"] = lower.sort_values(["d", "AMT_OUTSTANDING"],
                                         ascending=[True, False]).index[0]
    if len(upper):
        upper["d"] = (upper["mat"] - tm).dt.days
        out["upper"] = upper.sort_values(["d", "AMT_OUTSTANDING"],
                                         ascending=[True, False]).index[0]
    dl = int(lower.loc[out["lower"], "d"]) if out["lower"] else None
    du = int(upper.loc[out["upper"], "d"]) if out["upper"] else None
    if dl is not None and du is not None:
        out["nearest"], out["dist_nearest"] = (out["lower"], dl) if dl <= du else (out["upper"], du)
    elif dl is not None:
        out["nearest"], out["dist_nearest"] = out["lower"], dl
    elif du is not None:
        out["nearest"], out["dist_nearest"] = out["upper"], du
    return out


def ytm_semi_to_annual(y: float) -> float:
    return ((1 + y / 200.0) ** 2 - 1) * 100.0


def nominal_leg(match: dict, ytm_row: pd.Series, mkt: str,
                ytm_convention: str = "annual") -> float:
    """YTM della gamba nominale (percento), metodo 'nearest' = FLL/Kita-Tortorice
    matched-maturity: il singolo nominale piu' vicino per scadenza, mismatch in giorni,
    cap MAX_MISMATCH_DAYS. Convenzione 'annual' (default): il Total YtM del linker e'
    un'IRR in composizione ANNUALE, quindi i nominali UK/US -- che Bloomberg quota in
    composizione SEMESTRALE -- vanno convertiti (ytm_semi_to_annual), altrimenti la base
    incorpora ~4bp di puro artefatto di convenzione a rendimenti del 4%. IT/FR/DE quotano
    gia' annuale: nessuna conversione. 'local' = nessuna conversione (solo diagnostica)."""
    if match["nearest"] is None or match.get("dist_nearest", 1e9) > MAX_MISMATCH_DAYS:
        return np.nan
    v = ytm_row.get(match["nearest"], np.nan)
    if pd.isna(v):
        return np.nan
    return ytm_semi_to_annual(float(v)) if (ytm_convention == "annual"
                                            and mkt in ("UK", "US")) else float(v)


# ------------------------------------------------------------------ metodo C-esatto
def synthetic_irr(bond: LinkerBond, cf: pd.DataFrame, asof: date,
                  zero_row: pd.Series) -> float:
    """Sconta i flussi FUTURI sulla curva zero nominale (cc, %, colonne=maturita' in anni)
    AL REGOLAMENTO (il prezzo sintetico dirty vive a t0, come quello osservato) e risolve
    l'IRR sugli STESSI flussi. Percento. Base tempo identica all'IRR (ACT/365.25).
    Identita' esatta (testata): su curva piatta y_cc, IRR sintetico = e^{y_cc} - 1 per
    qualunque insieme di flussi -- scontare ad asof invece che al regolamento la rompe
    di ~0.25-0.5 bp, bias sistematico che questa versione elimina."""
    t0 = (cf.index[0] - asof).days / 365.25
    tt_abs = np.array([(d - asof).days / 365.25 for d in cf.index[1:]])
    taus = tt_abs - t0                                   # tempo dal regolamento
    grid = np.array(sorted(float(c) for c in zero_row.dropna().index))
    vals = zero_row.dropna().reindex(sorted(zero_row.dropna().index)).values.astype(float)
    z = np.interp(taus, grid, vals)                      # lineare in maturita', cc %
    c_fut = cf["Cashflows"].values[1:].astype(float)
    p_syn = float(np.sum(c_fut * np.exp(-z / 100.0 * taus)))
    c2 = np.concatenate([[-p_syn], c_fut])
    tt = np.concatenate([[t0], tt_abs])
    # coerenza col ramo speciale dell'IRR osservato
    if len(bond.cpn_dates) >= 2 and asof >= bond.cpn_dates[-2]:
        return (abs(c2[-1] / c2[0]) - 1) * 365 / (cf.index[-1] - cf.index[0]).days * 100
    return _solve_irr(c2, tt) * 100


# ------------------------------------------------------------------ floor (port)
def value_floor(bond: LinkerBond, asof: date, engine) -> float:
    """Port esatto del pricing originale del floor (Black su ratio, vol lognormale dai
    log-rendimenti mensili 3y con lambda=1, fixing 15 del mese precedente). Ritorna bp."""
    data = engine._visible(asof)
    prev_eom = pd.Timestamp(asof.replace(day=1) - relativedelta(days=1))
    win = data[(data.index >= prev_eom - relativedelta(years=3)) & (data.index <= prev_eom)]
    lr = np.log(win / win.shift(1)).dropna()
    mu = float(lr.mean())
    sd = float(np.sqrt(((lr - mu) ** 2).mean()))

    future = [d for d in bond.cpn_dates if d > asof]
    if future and asof > (future[0] - relativedelta(months=1)).replace(day=15):
        future = future[1:]

    def floorlet(ratio: float, vol: float, tenor: float) -> float:
        d1 = (np.log(ratio) + vol**2 * tenor / 2) / (vol * np.sqrt(tenor))
        d2 = d1 - vol * np.sqrt(tenor)
        return norm.cdf(-d2) - ratio * norm.cdf(-d1)

    tot = 0.0
    for cpn_date in future:
        fixing = (cpn_date - relativedelta(months=1)).replace(day=15)
        tenor = (fixing - asof).days / 365
        if tenor <= 0:
            continue
        ref = engine.reference_index(cpn_date, asof, bond.lag, bond.interpolate, "total")
        ratio = ref / bond.base_cpi
        vol = ratio * np.sqrt(np.exp(sd**2) - 1)
        fl = floorlet(ratio, vol, tenor)
        w = bond.nominal + bond.cpn_rate / (100 * bond.cpn_freq) * bond.nominal \
            if cpn_date == bond.maturity else bond.cpn_rate / (100 * bond.cpn_freq) * bond.nominal
        tot += fl * w
    return tot / bond.nominal * 10000


# ------------------------------------------------------------------ orchestrazione
def bond_basis_row(bond: LinkerBond, asof: date, engine, price_clean: float,
                   nom_ref: pd.DataFrame, ytm_row: pd.Series,
                   settle_cal: dict, cpn_cal: dict,
                   zero_row: pd.Series | None = None,
                   floor: bool = True, ytm_convention: str = "annual") -> dict:
    """Le misure per (bond, data): 'nearest' (FLL/KT matched-maturity senza STRIPS) e,
    se zero_row e' fornita, il C-esatto (IRR osservato - IRR sintetico sulla curva).
    Ritorna dict in bp/percento."""
    cf = bond.cashflows(asof, engine, price_clean, settle_cal, cpn_cal, True, floor)
    y_tot = bond.irr(cf, asof)
    match = match_nominals(nom_ref, bond.maturity, asof)
    y_nom = nominal_leg(match, ytm_row, bond.mkt, ytm_convention)
    out = {"total_ytm": y_tot,
           "basis_nearest": (y_tot - y_nom) * 100 if pd.notna(y_nom) else np.nan,
           "ynom_nearest": y_nom,
           "mismatch_days": match.get("dist_nearest", np.nan)}
    if zero_row is not None:
        y_syn = synthetic_irr(bond, cf, asof, zero_row)
        out["basis_c_exact"] = (y_tot - y_syn) * 100
        out["ysyn"] = y_syn
    return out


