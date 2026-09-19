"""inflation - proiezione dell'indice e stagionalita': la matematica di cpi.py, con cache.

FEDELTA'. Questo modulo REPLICA LETTERALMENTE l'algoritmo del progetto originale (cpi.py):
stessi anchor, stessi rami per-mercato, stessa aritmetica di riga (months_first_year,
index_adj, indices_to_zero, slice iloc), stessa stagionalita' vincolata. Non e' una
riscrittura "pulita" della matematica: e' la stessa matematica con tre differenze dichiarate:

  1. I/O: niente Bloomberg dentro. L'engine riceve la serie CPI (EOM) e il pannello ILS
     (date x tenor, composto annuale, percento) dalla cache di bbg.py.
  2. CACHE. La tabella proiettata dipende solo da (data, tipo), NON dal bond: si calcola
     una volta per data e si condivide fra tutti i bond. La stagionalita' dipende solo
     dall'ultimo dato CPI: si ricalcola quando esce un nuovo print, non 242.000 volte.
  3. pub_lag_days (default None = comportamento originale). L'originale decide se il print
     del mese m-1 e' disponibile guardando se prev_eom sta nella serie: in un backtest con
     la serie storica completa questo e' VERO anche nei primi giorni del mese, quando nella
     realta' il print non era ancora uscito (look-ahead di giorni/settimane). None replica
     l'originale ESATTAMENTE (necessario per il test di regressione su btpei_basis.xlsx);
     un intero (es. 14) maschera i print con eom + lag > asof, per la versione honest del
     paper. La differenza fra le due modalita' e' essa stessa un numero da riportare.

COLONNA CANONICA: 'CPI SA' -- e' quella che il consumatore originale legge (btp.py:1604).

STAGIONALITA' ('risk', il default dell'originale): minimizza sum_i (mu + s_{i mod 12} - r_i)^2
con vincolo sum(s)=0, dove r sono i log-rendimenti mensili nella finestra e mu la loro media.
QUIRK PRESERVATO: s e' indicizzato per POSIZIONE nella serie (i mod 12), e la mappa
posizione->mese di calendario e' data dai mesi delle prime 12 osservazioni. Cambiare questo
"a favore di pulizia" cambierebbe i numeri. Le due varianti lambda (drop/no-drop degli
estremi, pesi 0.95^anni) sono incluse per gli studi di stagionalita'.

RIFERIMENTO GIORNALIERO (formula MEF, identica per BTPei/TIPS/OATei/gilt new-style):
  ref(d; L) = I(m-L) + (day(d)-1)/gg_mese(d) * (I(m-L+1) - I(m-L)),  L=3
Gilt old-style: L=8 SENZA interpolazione (ref = I(m-8)) -- interpolate=False.

DEVIAZIONE DICHIARATA (irrilevante per la regressione): nelle stime override 1y/2y
l'originale usa l'anno del giorno di RUN (variabile globale `today`) per decidere quale
anno di calendario sostituire; qui si usa l'anno di asof. Default: override spenti.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from scipy.optimize import fsolve, minimize

warnings.filterwarnings("ignore")

try:
    from config import YEARS_FORWARD
except Exception:
    YEARS_FORWARD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]


def _eom(d: date, months_back: int = 0) -> date:
    """Fine mese del mese `months_back` mesi prima di quello di d (0 = mese di d... prev)."""
    return (pd.Timestamp(d) - relativedelta(months=months_back - 1)).replace(day=1).date() \
        - relativedelta(days=1) if months_back >= 1 else \
        (pd.Timestamp(d) + relativedelta(day=31)).date()


class InflationEngine:
    """Un engine per (mercato, serie CPI, pannello ILS). Vedi docstring di modulo."""

    def __init__(self, ticker: str, cpi_series: pd.Series, ils_panel: pd.DataFrame,
                 years_history: int = 10, season_weight: float = 1.0,
                 seasonality_method: str = "risk",
                 estimate_1y: float | None = None, estimate_2y: float | None = None,
                 pub_lag_days: int | None = None) -> None:
        self.ticker = ticker                      # 'CPTFEMU','UKRPI','CPURNSA','FRCPXTOB','ITCPIUNR'
        s = cpi_series.dropna().copy()
        s.index = pd.to_datetime(s.index)
        # normalizza a fine mese (i print Bloomberg possono arrivare a date infra-mese)
        s.index = s.index + MonthEnd(0)
        self.cpi = s[~s.index.duplicated(keep="last")].sort_index()
        p = ils_panel.copy()
        p.index = pd.to_datetime(p.index)
        # come l'originale (get_table): griglia business + interpolazione lineare nel tempo
        full = pd.bdate_range(p.index.min(), p.index.max())
        self._ils = p.reindex(p.index.union(full)).interpolate(method="time").reindex(full)
        self.years_history = years_history
        self.season_weight = season_weight
        self.seasonality_method = seasonality_method
        self.estimate_1y, self.estimate_2y = estimate_1y, estimate_2y
        self.pub_lag_days = pub_lag_days
        self._seas_cache: dict = {}
        self._proj_cache: dict = {}

    # ---------------------------------------------------------------- dati visibili
    def _visible(self, asof: date) -> pd.Series:
        if self.pub_lag_days is None:
            return self.cpi
        cut = pd.Timestamp(asof) - pd.Timedelta(days=self.pub_lag_days)
        return self.cpi[self.cpi.index <= cut]

    # ---------------------------------------------------------------- stagionalita'
    def seasonality(self, asof: date, method: str | None = None) -> pd.Series:
        """Pattern 1..12 a somma zero. Cache per (metodo, prev_eom, finestra)."""
        method = method or self.seasonality_method
        prev_eom = pd.Timestamp(_eom(asof, 1))
        key = (method, prev_eom, self.years_history, self.pub_lag_days)
        if key in self._seas_cache:
            return self._seas_cache[key]
        data = self._visible(asof)
        lo = prev_eom - relativedelta(years=self.years_history)
        win = data[(data.index > lo) & (data.index <= prev_eom)] if prev_eom in data.index \
            else data[(data.index >= lo) & (data.index <= prev_eom)]
        r = np.log(win / win.shift(1)).dropna()

        if method == "risk":
            lr, mu = r.values, float(r.mean())

            def objective(sv: np.ndarray) -> float:
                pos = np.arange(len(lr)) % 12
                return float(np.sum((mu + sv[pos] - lr) ** 2))

            res = minimize(objective, np.zeros(12),
                           constraints={"type": "eq", "fun": lambda sv: float(np.sum(sv))})
            months = r.index.month
            out = pd.Series(res.x, index=months[:12], name="Seasonality").sort_index()

        elif method in ("log_drop_lambda", "log_no_drop_lambda"):
            lam, vals = 0.95, []
            for month in range(1, 13):
                g = r[r.index.month == month].copy()
                if method == "log_drop_lambda" and len(g) > 2:
                    g = g.drop([g.idxmax(), g.idxmin()])
                yrs = g.index.max().year - g.index.year
                vals.append(float(np.average(g.values, weights=lam ** yrs)))
            aux = pd.Series(vals, index=np.arange(1, 13))
            out = (aux - aux.sum() / 12).rename("Seasonality")
        else:
            raise ValueError(f"metodo stagionalita' ignoto: {method}")

        self._seas_cache[key] = out
        return out

    # ---------------------------------------------------------------- ILS alla data
    def _ils_row(self, asof: date) -> pd.Series:
        ts = pd.Timestamp(asof)
        if ts in self._ils.index:
            row = self._ils.loc[ts]
        else:                                       # weekend/festivo: ultimo disponibile
            prior = self._ils.index[self._ils.index <= ts]
            if not len(prior):
                raise ValueError(f"nessun dato ILS a/prima di {asof}")
            row = self._ils.loc[prior.max()]
        row = row.dropna()
        row.index = [int(t) for t in row.index]
        return row[[t for t in row.index if float(t).is_integer()]]

    # ---------------------------------------------------------------- proiezione TOTAL
    def projection(self, asof: date, kind: str = "total") -> pd.DataFrame:
        key = (pd.Timestamp(asof), kind)
        if key in self._proj_cache:
            return self._proj_cache[key]
        out = self._project_total(asof) if kind == "total" else self._project_real(asof)
        self._proj_cache[key] = out
        return out

    def _project_total(self, asof: date) -> pd.DataFrame:
        data = self._visible(asof)
        tk = self.ticker
        ppprev = pd.Timestamp(_eom(asof, 3))
        pprev = pd.Timestamp(_eom(asof, 2))
        prev = pd.Timestamp(_eom(asof, 1))
        seas = self.seasonality(asof)
        bool_val = prev in data.index

        # precondizione dell'algoritmo originale: i print m-3 e m-2 devono essere visibili.
        # Con i lag di pubblicazione reali (14-25 gg) e' sempre vero; solo bool_val (m-1)
        # commuta. Un pub_lag_days piu' grande viola la struttura: errore parlante.
        for req in (ppprev, pprev):
            if req not in data.index:
                raise ValueError(
                    f"print CPI {req.date()} non visibile ad asof {asof} "
                    f"(pub_lag_days={self.pub_lag_days}): l'algoritmo richiede m-3 e m-2 "
                    f"pubblicati; riduci il lag o sposta asof")

        # --- anchor iniziale e ancore ILS (ramo UKRPI: t-2; altri: t-3) --------------
        start = pprev if tk == "UKRPI" else ppprev
        if start not in data.index:
            raise ValueError(f"CPI mancante a {start.date()} per asof {asof}")
        nsa = pd.Series({start: float(data.loc[start])})
        if bool_val:
            nsa.loc[prev] = float(data.loc[prev])
        elif tk != "UKRPI":
            nsa.loc[pprev] = float(data.loc[pprev])

        ils = self._ils_row(asof)
        yfwd = sorted(int(t) for t in ils.index)
        if tk == "UKRPI":
            for y in yfwd:
                nsa.loc[pprev + relativedelta(years=y)] = \
                    float(data.loc[pprev]) * (1 + ils.loc[y] / 100) ** y
        else:
            for y in yfwd:
                nsa.loc[ppprev + relativedelta(years=y)] = \
                    float(data.loc[ppprev]) * (1 + ils.loc[y] / 100) ** y

        # --- riempimento geometrico mensile (esponente in MESI, come l'originale) ----
        nsa = nsa.sort_index()
        idx = nsa.index if tk == "UKRPI" else nsa.index[1:]
        fills = {}
        for i in range(len(idx) - 1):
            a, b = idx[i], idx[i + 1]
            span = (b.year - a.year) * 12 + (b.month - a.month)
            for m in pd.date_range(a, b, freq="ME"):
                if m in nsa.index:
                    continue
                k = (m.year - a.year) * 12 + (m.month - a.month)
                fills[m] = float(nsa.loc[a]) * (float(nsa.loc[b]) / float(nsa.loc[a])) ** (k / span)
        nsa = pd.concat([nsa, pd.Series(fills)]).sort_index()

        sa = pd.DataFrame({"CPI NSA": nsa})
        sa["Seasonality"] = pd.Series(sa.index.month, index=sa.index).map(seas.to_dict()) \
            * self.season_weight
        sa["Seasonality Add"] = 0.0

        # --- tabellina (bool_val, UKRPI) -> months_first_year, index_adj -------------
        if bool_val:
            mfy, adj = (11, 1) if tk == "UKRPI" else (10, 1)
        else:
            mfy, adj = (12, 0) if tk == "UKRPI" else (11, 1)

        adjustment = -float(sa.iloc[1 + adj: 1 + mfy + adj]["Seasonality"].cumsum().iloc[-1]) / mfy
        idx_zero = list((np.array(yfwd) - 1) * 12 + mfy + adj)

        col_add = sa.columns.get_loc("Seasonality Add")
        sa.iloc[1 + adj: mfy + adj, col_add] = sa.iloc[1 + adj: mfy + adj]["Seasonality"].cumsum()
        sa.iloc[mfy + 1 + adj:, col_add] = sa.iloc[mfy + 1 + adj:]["Seasonality"].cumsum()

        sa["CPI SA"] = 0.0
        col_sa = sa.columns.get_loc("CPI SA")
        col_nsa = sa.columns.get_loc("CPI NSA")
        sa.iloc[0, col_sa] = sa.iloc[0, col_nsa]
        if not (tk == "UKRPI" and not bool_val):
            sa.iloc[1, col_sa] = sa.iloc[1, col_nsa]
        sa.loc[sa.index[idx_zero], "CPI SA"] = sa.loc[sa.index[idx_zero], "CPI NSA"]

        if tk == "UKRPI":
            if bool_val:
                sa = sa.drop(sa.index[0])
        else:
            sa = sa.drop(sa.index[0])

        # --- stub del primo anno: NSA * (1 + SeasAdd + mesi * adjustment) ------------
        first = sa.index[0]
        mdiff = np.array([relativedelta(d, first).months for d in sa.index[1:mfy]])
        sa.iloc[1:mfy, col_sa] = sa.iloc[1:mfy, col_nsa].values * \
            (1 + sa.iloc[1:mfy, col_add].values + mdiff * adjustment)

        # --- blocchi fra ancore: NSA * (1 + SeasAdd) ----------------------------------
        tot = 0
        for i in range(len(yfwd) - 1):
            dy = yfwd[i + 1] - yfwd[i]
            lo, hi = mfy + 1 + 12 * tot, mfy + 12 * (tot + dy)
            sa.iloc[lo:hi, col_sa] = sa.iloc[lo:hi, col_nsa].values * \
                (1 + sa.iloc[lo:hi, col_add].values)
            tot += dy

        # --- override analista (default: spenti) --------------------------------------
        if self.estimate_1y is not None:
            sa["CPI Ratio"] = sa["CPI SA"] / sa["CPI SA"].shift(1)

            def solve_rate(target: float) -> float:
                def f(rm: float) -> float:
                    prod = 1.0
                    for m in seas.index:
                        prod *= (1 + rm + seas.loc[m])
                    return (1 + target) - prod
                return float(fsolve(f, 0.001)[0])

            def replace_year(yr: int, rate: float) -> None:
                mask = (sa.index >= pd.Timestamp(yr, 1, 1)) & (sa.index <= pd.Timestamp(yr, 12, 31))
                gf = 1 + rate + sa.loc[mask, "Seasonality"]
                sa.loc[mask, "CPI SA"] = float(sa["CPI SA"].shift(1).loc[mask].iloc[0]) * gf.cumprod()

            y1 = pd.Timestamp(asof).year + 1
            replace_year(y1, solve_rate(self.estimate_1y))
            if self.estimate_2y is not None:
                replace_year(y1 + 1, solve_rate(self.estimate_2y))
                tail = sa.index >= pd.Timestamp(y1 + 2, 1, 1)
            else:
                tail = sa.index >= pd.Timestamp(y1 + 1, 1, 1)
            sa.loc[tail, "CPI SA"] = sa["CPI SA"].shift(1).loc[tail] * sa.loc[tail, "CPI Ratio"]
            sa = sa.drop(columns=["CPI Ratio"])

        # --- timbro finale degli actual m-3 / m-2 (come l'originale) ------------------
        for d in (ppprev, pprev):
            sa.loc[d] = [float(data.loc[d]), float(seas.loc[d.month]), 0.0, float(data.loc[d])]
        return sa.sort_index()

    # ---------------------------------------------------------------- proiezione REAL
    def _project_real(self, asof: date) -> pd.DataFrame:
        """Indice congelato all'ultimo actual, stagionalita' mantenuta dentro l'anno e
        azzerata a ogni anniversario (per il Real YtM / breakeven)."""
        data = self._visible(asof)
        ppprev = pd.Timestamp(_eom(asof, 3))
        pprev = pd.Timestamp(_eom(asof, 2))
        prev = pd.Timestamp(_eom(asof, 1))
        seas = self.seasonality(asof)
        bool_val = prev in data.index
        n_years = max(int(t) for t in self._ils_row(asof).index)

        start = prev if bool_val else pprev
        base = float(data.loc[start])
        n = n_years * 12 - (2 if bool_val else 1)
        idx = pd.DatetimeIndex([start]).append(
            pd.date_range(start + relativedelta(months=1), periods=n, freq="ME"))
        sa = pd.DataFrame({"CPI NSA": base}, index=idx)
        sa["Seasonality"] = pd.Series(sa.index.month, index=sa.index).map(seas.to_dict()) \
            * self.season_weight
        sa["Seasonality Add"] = 0.0
        idx_zero = np.arange(0, n_years * 12, 12)
        idx_zero = idx_zero[idx_zero < len(sa)]
        sa.iloc[idx_zero, sa.columns.get_loc("Seasonality")] = 0.0
        col_add = sa.columns.get_loc("Seasonality Add")
        for j in range(n_years):
            lo, hi = 1 + 12 * j, 12 * (j + 1)
            if lo >= len(sa):
                break
            sa.iloc[lo:hi, col_add] = sa.iloc[lo:hi]["Seasonality"].cumsum()
        sa["CPI SA"] = 0.0
        col_sa = sa.columns.get_loc("CPI SA")
        sa.iloc[idx_zero, col_sa] = sa.iloc[idx_zero, sa.columns.get_loc("CPI NSA")]
        for j in range(n_years):
            lo, hi = 1 + 12 * j, 12 * (j + 1)
            if lo >= len(sa):
                break
            sa.iloc[lo:hi, col_sa] = float(sa.iloc[12 * j, col_sa]) * \
                (1 + sa.iloc[lo:hi, col_add].values)
        for d in (ppprev, pprev):
            if d in data.index:
                sa.loc[d] = [float(data.loc[d]), float(seas.loc[d.month]), 0.0, float(data.loc[d])]
        return sa.sort_index()

    # ---------------------------------------------------------------- indice di riferimento
    def _index_series(self, asof: date, kind: str) -> pd.Series:
        """Actuals storici + proiezione: gli actual coprono i mesi precedenti all'inizio
        della tabella (servono ai lag lunghi, es. 8 mesi dei gilt old-style); sul tratto
        sovrapposto i timbri della proiezione coincidono con gli actual per costruzione."""
        tab = self.projection(asof, kind)["CPI SA"]
        vis = self._visible(asof)
        return pd.concat([vis[vis.index < tab.index[0]], tab])

    def reference_index(self, ref_date: date, asof: date, lag: int = 3,
                        interpolate: bool = True, kind: str = "total") -> float:
        """Formula MEF sul 'CPI SA' della proiezione (con fallback sugli actual storici
        per i mesi anteriori). lag=3 interpolato (BTPei/TIPS/OATei/gilt new-style);
        lag=8 interpolate=False per i gilt old-style."""
        tab = self._index_series(asof, kind)
        rd = pd.Timestamp(ref_date)
        m_l = (rd.replace(day=1) - relativedelta(months=lag) + MonthEnd(0))
        if not interpolate:
            return float(tab.loc[m_l])
        m_l1 = (rd.replace(day=1) - relativedelta(months=lag - 1) + MonthEnd(0))
        dim = (rd + MonthEnd(0)).day
        w = (rd.day - 1) / dim
        return float(tab.loc[m_l]) + w * (float(tab.loc[m_l1]) - float(tab.loc[m_l]))

    def index_ratio(self, ref_date: date, asof: date, base_cpi: float,
                    lag: int = 3, interpolate: bool = True, floor: bool = False,
                    kind: str = "total") -> float:
        ci = self.reference_index(ref_date, asof, lag, interpolate, kind) / base_cpi
        return max(ci, 1.0) if floor else ci
