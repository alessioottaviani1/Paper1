"""
13 - STRATEGIA CCT-BTP: indici SW ed EW secondo Paper 1, eq. (4) e (5).

REPLICA FEDELE. Le due ponderazioni sono quelle di Paper 1 (Ottaviani), che riprende
Duarte et al. (2007) per l'equal-weighted e Rebonato-Ronzani (2021) per il signal-weighted:

    r_t^EW = (1/K_t) * sum_k r_{k,t}

    r_t^SW = sum_k [ |b_{k,0} - theta_{t_k}| / sum_j |b_{j,0} - theta_{t_j}| ] * r_{k,t}

dove b_{k,0} e' la base ALLA DATA DI ENTRATA del trade k, theta_{t_k} e' la media a finestra
ESPANDENTE della base cross-sezionale dall'inizio del campione fino a t_k, e r_{k,t} e' il
rendimento con segno del trade, con la DIREZIONE FISSATA ALL'ENTRATA. Il valore assoluto fa
si' che il peso rifletta l'ampiezza della dislocazione e non il suo segno, che r_{k,t} porta
gia'. La logica di Rebonato-Ronzani: una base piu' lontana dal proprio equilibrio storico e'
un'opportunita' piu' ricca e merita un'allocazione maggiore, condizionatamente alla convergenza.

TRE PUNTI IN CUI E' FACILE SBAGLIARE, e che qui sono rispettati:
  1. il peso usa la base ALL'ENTRATA, non quella corrente: e' fissato una volta e non si
     aggiorna, altrimenti si sta inseguendo il segnale invece di allocare su di esso;
  2. il peso e' la DEVIAZIONE da theta, non il livello: una base di 20 bp quando l'equilibrio
     e' 15 e' un'opportunita' piccola, non grande;
  3. theta e' espandente e calcolata solo su dati PASSATI: nessun look-ahead.

In Paper 1 il SW e' il BASELINE e l'EW la robustezza, perche' il SW domina su Sharpe e MPPM
in tutte e tre le strategie. Qui si riportano entrambi, con il SW per primo.

SOGLIE. Primaria: entrata a 10 bp, uscita a 0. Dieci bp e' circa il 60mo percentile della base
CCTeu e sta sopra l'RMSE della curva (5.9 bp): a 5 bp si scambierebbe errore di stima per
segnale. Minimo 6 mesi a scadenza all'entrata, come Paper 1. Griglia 5/15/20/30 come robustezza.

NORMALIZZAZIONE PER RISCHIO (differenza necessaria rispetto a Paper 1). Le tre strategie di
Paper 1 sono spread trade su strumenti omogenei, quindi i trade sono gia' comparabili. Qui no:
una base di 10 bp vale 0.1 punti di prezzo su un CCT a un anno e 0.6 su uno a sei anni. Sono
la stessa opportunita' in RENDIMENTO ma sei volte diverse in P&L, e un indice che le media a
peso uguale e' dominato in varianza dai trade lunghi -- da cui una curtosi molto superiore a
quella di Paper 1. Ogni trade viene quindi scalato per la propria DV01 (approssimata dalla
duration del CCT all'entrata), cosi' che tutti contribuiscano rischio comparabile. La versione
NON scalata resta riportata, perche' e' la replica letterale di Paper 1 e la differenza fra
le due e' informativa.

RENDIMENTI. Mensili, in percentuale del nozionale: la base in prezzo e' gia' espressa in punti
per 100, quindi la sua variazione E' un rendimento percentuale. Media e deviazione standard
annualizzate, Sharpe come loro rapporto, t di Newey-West, e le stesse colonne della Tabella
A.IV di Paper 1 (skew, curtosi in eccesso, %Neg, AC(1)) per confrontabilita' diretta.

Output: PROC/strategy_returns.csv, PROC/strategy_trades.csv + results/13_strategy.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

ENTRY_MAIN, EXIT_MAIN = 10.0, 0.0
GRID = [5.0, 10.0, 15.0, 20.0, 30.0]
MIN_TAU_ENTRY = 0.5
RA_GRID = [2, 3, 4]

def nw_t(x, lags=None):
    """t di Newey-West sulla media, come in Paper 1."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 12: return np.nan
    if lags is None: lags = int(np.floor(4*(n/100)**(2/9)))
    e = x - x.mean(); g0 = float(np.dot(e, e))/n; s = g0
    for l in range(1, lags+1):
        g = float(np.dot(e[l:], e[:-l]))/n
        s += 2*(1 - l/(lags+1))*g
    return np.nan if s <= 0 else x.mean()/np.sqrt(s/n)

def find_trades(B, entry, exit_):
    """
    Identifica i trade come in Duarte et al.: ogni trade e' un fondo separato che apre quando
    la base supera la soglia d'entrata e chiude quando rientra oltre quella d'uscita o il
    titolo scade. theta e' la media espandente della base cross-sezionale, solo dati passati.
    """
    B = B.sort_values("date").copy()
    # theta_t: media espandente della base cross-sezionale (in bp), shiftata di un giorno
    daily_x = B.groupby("date")["basis3_y"].mean()
    theta = daily_x.expanding().mean().shift(1)

    trades, legs = [], []
    for isin, g in B.groupby("CCT_ISIN"):
        g = g.sort_values("date").set_index("date")
        y, p = g["basis3_y"], g["basis3_p"]
        tau = g["tau_cct"] if "tau_cct" in g.columns else pd.Series(np.nan, index=g.index)
        direction, b0, t0, tid = 0, np.nan, None, None
        for d in g.index:
            s = y.get(d, np.nan)
            if direction == 0:
                if np.isfinite(s) and abs(s) >= entry and (not np.isfinite(tau.get(d, np.nan))
                                                           or tau.get(d, np.nan) >= MIN_TAU_ENTRY):
                    # base positiva in bp = CCT rende di piu' = economico -> si COMPRA il CCT
                    direction = int(np.sign(s)); b0 = float(s); t0 = d
                    tau0 = tau.get(d, np.nan)
                    tau0 = tau0 if np.isfinite(tau0) else 3.0
                    tid = f"{isin}_{d.date()}"
            else:
                closed = (not np.isfinite(s)) or (direction > 0 and s <= exit_) or \
                         (direction < 0 and s >= -exit_)
                if closed:
                    # La gamba del giorno di USCITA va registrata PRIMA di chiudere: e' il
                    # giorno in cui la base attraversa lo zero, cioe' dove sta il grosso del
                    # guadagno. Saltarla toglie sistematicamente il movimento finale -- se la
                    # base passa da +12 a -3 in un giorno, si perde l'intero salto -- e
                    # produce un bias verso il basso che puo' rendere negativa una strategia
                    # che per costruzione chiude sempre in profitto sui trade a convergenza.
                    if np.isfinite(p.get(d, np.nan)):
                        legs.append({"date": d, "trade": tid, "isin": isin, "dir": direction,
                                     "b0": b0, "theta0": float(theta.get(t0, np.nan)),
                                     "p": float(p.get(d, np.nan)),
                                     "dv01": max(float(tau0), 0.25)})
                    trades.append({"trade": tid, "isin": isin, "entry": t0, "exit": d,
                                   "dir": direction, "b0": b0, "b1": float(s) if np.isfinite(s) else np.nan,
                                   "theta0": float(theta.get(t0, np.nan)),
                                   "days": (d - t0).days,
                                   "exit_reason": "convergenza" if np.isfinite(s) else "dati"})
                    direction = 0; continue
            if direction != 0:
                legs.append({"date": d, "trade": tid, "isin": isin, "dir": direction,
                             "b0": b0, "theta0": float(theta.get(t0, np.nan)),
                             "p": float(p.get(d, np.nan)),
                             "dv01": max(float(tau0), 0.25)})
        # chiusura a SCADENZA: se il ciclo finisce con posizione aperta, il trade esiste
        # comunque e va registrato, altrimenti il conteggio e' sottostimato e manca l'uscita
        # per scadenza, che in Paper 1 e' una colonna della tabella trade-level.
        if direction != 0 and t0 is not None:
            trades.append({"trade": tid, "isin": isin, "entry": t0, "exit": g.index[-1],
                           "dir": direction, "b0": b0, "b1": float(y.get(g.index[-1], np.nan)),
                           "theta0": float(theta.get(t0, np.nan)),
                           "days": (g.index[-1]-t0).days, "exit_reason": "scadenza"})
    T = pd.DataFrame(trades)
    Lg = pd.DataFrame(legs)
    if Lg.empty: return T, Lg
    # rendimento giornaliero del trade: comprando il CCT economico si guadagna quando il suo
    # prezzo risale verso il fair value, cioe' quando basis3_p AUMENTA.
    Lg = Lg.sort_values(["trade","date"])
    Lg["r_raw"] = Lg.groupby("trade")["p"].diff() * Lg["dir"]
    # scalato per DV01: ogni trade porta rischio comparabile. Il fattore 3.0 riporta la scala
    # a quella di un trade a tre anni, cosi' i livelli restano leggibili in punti per 100.
    Lg["r"] = Lg["r_raw"] * (3.0 / Lg["dv01"])
    # peso SW: |b_0 - theta_0|, FISSO all'entrata
    Lg["w_sw"] = (Lg["b0"] - Lg["theta0"]).abs()
    return T, Lg.dropna(subset=["r"])

def weight_diag(Lg):
    """
    Concentrazione dei pesi SW. Con pochi trade attivi e dislocazioni molto disperse
    l'indice signal-weighted puo' ridursi al rendimento di UN SOLO trade: il numero
    efficace di posizioni e' 1/Herfindahl dei pesi normalizzati. Se e' vicino a 1, il SW
    non e' un indice ma una scommessa singola, e confrontarne lo Sharpe con l'EW -- che
    diversifica su tutti i trade attivi -- non ha senso.
    """
    rows = []
    for d, x in Lg.groupby("date"):
        w = np.nan_to_num(x["w_sw"].values)
        if w.sum() <= 0: continue
        w = w/w.sum()
        rows.append({"date": d, "k": len(w), "eff_n": 1.0/np.sum(w**2), "wmax": w.max()})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

def indices(Lg):
    """Aggrega i trade in indici EW e SW, normalizzando i pesi fra i trade attivi ogni giorno."""
    ew = Lg.groupby("date")["r"].mean()
    def sw_day(x):
        w = x["w_sw"].values
        if not np.isfinite(w).any() or np.nansum(w) <= 0: return np.nanmean(x["r"].values)
        w = np.nan_to_num(w); return float(np.sum(w*x["r"].values)/np.sum(w))
    sw = Lg.groupby("date").apply(sw_day)
    k  = Lg.groupby("date").size()
    return ew, sw, k

def monthly(r):
    return r.resample("ME").sum()

def stats(rm, freq=12):
    if len(rm) < 12: return {}
    m, s = rm.mean()*freq, rm.std()*np.sqrt(freq)
    return {"n": len(rm), "mean": m, "t": nw_t(rm.values)*np.sqrt(1), "std": s,
            "min": rm.min(), "max": rm.max(), "skew": rm.skew(), "kurt": rm.kurtosis(),
            "neg": (rm < 0).mean()*100, "ac1": rm.autocorr(1),
            "sharpe": m/s if s > 0 else np.nan}

def mppm(rm, ra, freq=12):
    x = 1.0 + rm/100.0
    x = x[x > 0]
    if len(x) < 12: return np.nan
    if abs(ra-1.0) < 1e-9: return float(np.mean(np.log(x))*freq)
    return float((1.0/(1.0-ra))*np.log(np.mean(x**(1.0-ra)))*freq)

def vol_managed(rm, w=12):
    v = rm.rolling(w).var().shift(1); c = rm.var()
    return (rm*(c/v)).replace([np.inf,-np.inf], np.nan).dropna()

if __name__ == "__main__":
    print("== 13 strategia ==")
    B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
    ONLY_CCTEU = True   # solo i CCTeu sono un arbitraggio replicabile (IRS Euribor di mercato):
                        # i loro Sharpe sono REALIZZABILI. I CCT-BOT (model-based, nessuno swap
                        # contro BOT) darebbero Sharpe TEORICI non monetizzabili -> esclusi.
    if ONLY_CCTEU:
        B = B[B.regime == "CCTeu"].copy()
        print(f"  [ONLY_CCTEU] strategia sui soli CCTeu (arbitraggio realizzabile): {len(B):,} oss.")
    L=[]; P=L.append
    P("=== 13 STRATEGIA CCT-BTP: indici SW ed EW (Paper 1, eq. 4-5) ===")
    P(f"entrata |base| >= {ENTRY_MAIN:.0f} bp | uscita a {EXIT_MAIN:.0f} | "
      f"min {MIN_TAU_ENTRY*12:.0f} mesi a scadenza")
    P("peso SW = |base all'entrata - media espandente cross-sezionale|, FISSO dall'entrata")

    keep = {}
    for reg in (["CCTeu"] if ONLY_CCTEU else ["TUTTI", "CCT-BOT", "CCTeu"]):
        g = B if reg == "TUTTI" else B[B.regime == reg]
        if len(g) < 1000: continue
        T, Lg = find_trades(g, ENTRY_MAIN, EXIT_MAIN)
        if Lg.empty: P(f"\n{reg}: nessun trade"); continue
        ew, sw, k = indices(Lg)
        WD = weight_diag(Lg)
        P(f"\n{'='*88}\n{reg}: {len(T):,} trade | durata mediana {T.days.median():.0f} gg | "
          f"trade aperti: mediana {k.median():.0f}, max {k.max()}\n{'='*88}")
        if not WD.empty:
            P(f"  CONCENTRAZIONE DEI PESI SW: trade attivi mediana {WD.k.median():.0f}, "
              f"numero EFFICACE (1/Herfindahl) mediana {WD.eff_n.median():.1f}")
            P(f"    peso massimo di un singolo trade: mediana {WD.wmax.median():.0%}, "
              f"p90 {WD.wmax.quantile(.9):.0%}, max {WD.wmax.max():.0%}")
            P(f"    giorni in cui un trade pesa oltre il 50%: {(WD.wmax>0.5).mean():.0%}")
            P( "    [se il numero efficace e' vicino a 1, il SW non e' un indice ma una")
            P( "     scommessa singola: confrontarne lo Sharpe con l'EW non e' informativo]")
        P(f"  {'schema':>9}{'n':>5}{'Mean':>8}{'t-NW':>7}{'StdDev':>8}{'Min':>7}{'Max':>7}"
          f"{'Skew':>7}{'Kurt':>7}{'%Neg':>7}{'AC(1)':>7}{'Sharpe':>8}")
        Lg_raw = Lg.copy(); Lg_raw["r"] = Lg_raw["r_raw"]
        ew_raw, sw_raw, _ = indices(Lg_raw)
        # SW con pesi troncati al 95mo percentile: stessa logica di Rebonato-Ronzani ma
        # senza che una singola dislocazione estrema catturi l'intero indice.
        Lg_cap = Lg.copy()
        cap = Lg_cap["w_sw"].quantile(0.95)
        Lg_cap["w_sw"] = Lg_cap["w_sw"].clip(upper=cap)
        _, sw_cap, _ = indices(Lg_cap)
        for lab, r in [("SW", sw), ("EW", ew), ("SW-cap95", sw_cap),
                       ("SW-noDV01", sw_raw), ("EW-noDV01", ew_raw)]:
            rm = monthly(r); st = stats(rm)
            if not st: continue
            keep[f"{reg}_{lab}"] = rm
            P(f"  {lab:>9}{st['n']:>5}{st['mean']:>8.2f}{st['t']:>7.2f}{st['std']:>8.2f}"
              f"{st['min']:>7.2f}{st['max']:>7.2f}{st['skew']:>7.2f}{st['kurt']:>7.2f}"
              f"{st['neg']:>7.1f}{st['ac1']:>7.2f}{st['sharpe']:>8.2f}")
        P(f"  [Paper 1 EW: BTP Italia 0.79 | CDS-Bond 1.46 | iTraxx Skew 1.30]")
        P( "  [noDV01 = replica letterale di Paper 1, senza normalizzare i trade per rischio.")
        P( "   SW-cap95 = pesi troncati al 95mo percentile: se lo Sharpe sale sensibilmente,")
        P( "   il SW puro era dominato da poche dislocazioni estreme.]")
        if "exit_reason" in T.columns:
            P(f"  uscite: {T.exit_reason.value_counts().to_dict()}")
        kr = stats(monthly(sw)).get("kurt", np.nan)
        P(f"  curtosi in eccesso SW: {kr:.1f}  [Paper 1: 1.9-5.3. Se molto piu' alta, pochi")
        P( "   episodi dominano e va riportata anche la versione winsorizzata]")
        for lab, r in [("SW", sw), ("EW", ew)]:
            rm = monthly(r)
            if len(rm) < 12: continue
            P(f"  MPPM {lab} (avversione {RA_GRID}): " + "  ".join(f"{mppm(rm,a):+.2f}" for a in RA_GRID))
            vm = vol_managed(rm)
            if len(vm) > 12:
                s0, s1 = stats(rm).get("sharpe", np.nan), stats(vm).get("sharpe", np.nan)
                P(f"  vol-managed {lab}: Sharpe {s1:+.2f} vs {s0:+.2f} "
                  f"({'migliora' if s1 > s0 else 'PEGGIORA -> la strategia rende quando la vol e alta'})")
        # --- VERIFICA DI COSTRUZIONE: i trade a convergenza DEVONO chiudere in profitto.
        # Si entra a |base| >= soglia e si esce quando rientra oltre zero: la base si e'
        # mossa a favore, quindi il P&L cumulato del trade non puo' essere negativo. Se lo
        # e', c'e' un errore nel P&L, non nella strategia. E' il controllo piu' stringente
        # dell'intero script perche' segue dalla logica dell'arbitraggio, non dai dati.
        pnl_tr = Lg.groupby("trade")["r_raw"].sum()
        Tv = T.set_index("trade").join(pnl_tr.rename("pnl")) if "trade" in T.columns else pd.DataFrame()
        if not Tv.empty and Tv.pnl.notna().any():
            P("\n  VERIFICA: i trade chiudono in profitto?")
            for reason, gg in Tv.dropna(subset=["pnl"]).groupby("exit_reason"):
                pos = (gg.pnl > 0).mean()
                P(f"    {reason:12s}: {len(gg):>4} trade | in profitto {pos:>6.1%} | "
                  f"P&L mediano {gg.pnl.median():+.3f} | peggiore {gg.pnl.min():+.3f}")
            conv = Tv[(Tv.exit_reason=="convergenza") & Tv.pnl.notna()]
            bad = conv[conv.pnl < -1e-9]
            if len(bad):
                P(f"    [!] {len(bad)} trade a convergenza chiudono in PERDITA: non e' possibile")
                P( "        per costruzione. Esempi (trade, b0, b1, P&L):")
                for t, r in bad.nsmallest(5, "pnl").iterrows():
                    P(f"        {str(t)[:28]:30s} b0={r.b0:+7.1f} b1={r.get('b1',np.nan):+7.1f} pnl={r.pnl:+.3f}")
            else:
                P( "    [OK] tutti i trade a convergenza chiudono in profitto, come deve essere.")
            P(f"    P&L totale: {Tv.pnl.sum():+.1f} punti | per trade {Tv.pnl.mean():+.3f}")
        if reg == "TUTTI":
            (Tv if not Tv.empty else T).to_csv(PROC/"strategy_trades.csv")

    P(f"\n{'='*88}\nROBUSTEZZA ALLE SOGLIE (tutti i CCT)\n{'='*88}")
    P(f"  {'entrata':>9}{'n trade':>9}{'Sharpe SW':>11}{'Sharpe EW':>11}{'Mean SW':>10}{'mesi':>7}")
    for e in GRID:
        T, Lg = find_trades(B, e, EXIT_MAIN)
        if Lg.empty: continue
        ew, sw, _ = indices(Lg)
        se, ss = stats(monthly(ew)), stats(monthly(sw))
        if not ss: continue
        P(f"  {e:>7.0f}bp{len(T):>9,}{ss['sharpe']:>11.2f}{se.get('sharpe',np.nan):>11.2f}"
          f"{ss['mean']:>10.2f}{ss['n']:>7}")
    P("  [se lo Sharpe e' stabile sulla griglia, la soglia non e' scelta ex post]")

    if keep:
        pd.DataFrame(keep).to_csv(PROC/"strategy_returns.csv")
        P(f"\n[saved] {PROC/'strategy_returns.csv'}, {PROC/'strategy_trades.csv'}")
    save_txt("13_strategy.txt", L); print("\n".join(L))
