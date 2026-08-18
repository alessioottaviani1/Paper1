"""
14 - LA STORIA ECONOMICA HA UNA STORIA? Batteria di predizioni numerate.

PERCHE' QUESTO SCRIPT. Fino a 13 il progetto documenta un FATTO e identifica due canali
sopravvissuti. Non e' ancora una storia economica: Paper 1 non descrive tre anomalie, prende
una teoria (Duffie, Brunnermeier-Pedersen, Mitchell-Pulvino), ne deriva CINQUE predizioni
numerate, e le testa -- sfruttando l'eterogeneita' della base di detentori come fonte di
identificazione. Qui si fa lo stesso con la lettura che i risultati suggeriscono, PRIMA di
portarla a Rebonato.

LA STORIA DA TESTARE: "stesso bilancio, due regimi".
I CCT sono detenuti dalle banche italiane per il matching di bilancio (attivo a tasso
variabile contro passivo a tasso variabile). Questo genera DUE effetti dallo stesso bilancio:
in calma la domanda strutturale li rende CARI; nello stress le stesse banche sono il settore
vincolato e vendono cio' che hanno in quantita', rendendoli ECONOMICI. La domanda e il
dissesto vengono dalla medesima fonte -- ed e' questo che distingue la storia da uno
slow-moving capital generico, che predice solo allargamento sotto stress, non un'ASIMMETRIA
DI SEGNO fra regimi.

CINQUE PREDIZIONI, ciascuna falsificabile:

  P1  ASIMMETRIA DI SEGNO. L'effetto della dimensione sulla base deve essere POSITIVO in
      calma (le banche detengono i titoli grandi -> domanda -> caro) e NEGATIVO nello stress
      (vendono gli stessi titoli). E' la predizione decisiva: un canale di sola liquidita'
      da' segno negativo in entrambi i regimi, uno di sola domanda positivo in entrambi.
      Test: stima separata nei due regimi, non interazione -- l'interazione nasconde se il
      punto di inversione cada DENTRO o FUORI dal campione osservato.

  P2  DETENZIONI. La base deve rispondere alle detenzioni bancarie EFFETTIVE di titoli
      pubblici, non solo a proxy di stress. Quando le banche riducono il portafoglio
      sovrano, il CCT si sconta. Dati: API BCE (02c), gia' scaricati.

  P3  HAIRCUT. Se il vantaggio di collaterale spiega la ricchezza in calma, questa deve
      CRESCERE con la scadenza: i titoli a tasso variabile prendono l'haircut della fascia
      0-1 anno indipendentemente dalla scadenza, quindi il vantaggio rispetto al BTP cresce
      col tenor. Test: profilo per fascia nei periodi calmi.

  P4  CONCENTRAZIONE DELLO SCONTO. Nello stress lo sconto deve concentrarsi dove le banche
      detengono di piu': non solo emissioni grandi ma anche scadenze compatibili col
      matching di bilancio (il passivo bancario e' a breve). Test: sconto per fascia x regime.

  P5  DISTINZIONE DA PAPER 1. Se il canale e' la CLIENTELA BANCARIA e non il capitale lento
      degli arbitraggisti levered, la base CCT deve co-muoversi con le tre strategie di
      Paper 1 MENO di quanto quelle facciano fra loro. E' l'estensione diretta della fonte
      di identificazione di Paper 1 (chi detiene lo strumento). Richiede le serie di
      mispricing di Paper 1: se non presenti, lo script lo dichiara.

Output: results/14_story_tests.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

CALM_Q, STRESS_Q = 0.33, 0.67          # terzili di stress, fissati ex ante

if __name__ == "__main__":
    print("== 14 test della storia ==")
    L=[]; P=L.append
    P("=== 14 LA STORIA HA UNA STORIA? BATTERIA DI PREDIZIONI ===")
    M = pd.read_csv(PROC/"mechanism_panel.csv", parse_dates=["date"])
    B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
    try:
        X = pd.read_csv(PROC/"ecb_series.csv", index_col=0, parse_dates=True)
    except Exception:
        X = None

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        P("[statsmodels assente]"); save_txt("14_story_tests.txt", L); raise SystemExit

    def run(d, f, keys, lab):
        try:
            r = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["CCT_ISIN"]})
            P(f"  {lab:34s} " + "  ".join(
                f"{k}={r.params.get(k,np.nan):+.4f}[{r.tvalues.get(k,np.nan):+.2f}]" for k in keys)
              + f"   n {int(r.nobs):,}")
            return {k: (r.params.get(k,np.nan), r.tvalues.get(k,np.nan)) for k in keys}
        except Exception as e:
            P(f"  {lab:34s} fallita ({str(e)[:45]})"); return {}

    d = M.dropna(subset=["basis_p","sov_swap","logamt"]).copy()
    d["mon"]=d.date.dt.month.astype(str); d["yr"]=d.date.dt.year.astype(str)
    lo, hi = d.sov_swap.quantile(CALM_Q), d.sov_swap.quantile(STRESS_Q)
    P(f"\nregimi per terzili di stress sovrano: calma <= {lo:.3f}%, stress >= {hi:.3f}%")
    P(f"  range osservato dello stress: [{d.sov_swap.min():.3f}, {d.sov_swap.max():.3f}]")

    P("\n" + "="*76)
    P("P1  ASIMMETRIA DI SEGNO -- la predizione decisiva")
    P("    atteso: dimensione POSITIVA in calma, NEGATIVA nello stress")
    P("    (liquidita' pura -> negativa in entrambi; domanda pura -> positiva in entrambi)")
    P("="*76)
    res = {}
    for lab, sub in [("CALMA  (terzile basso)", d[d.sov_swap <= lo]),
                     ("MEDIO", d[(d.sov_swap > lo) & (d.sov_swap < hi)]),
                     ("STRESS (terzile alto)", d[d.sov_swap >= hi])]:
        if len(sub) < 200: continue
        w = sub.copy(); w["logamt_c"] = w.logamt - w.logamt.mean()
        res[lab] = run(w, "basis_p ~ logamt_c + tau + I(tau**2) + C(mon)", ["logamt_c"],
                       f"  {lab}")
    if "CALMA  (terzile basso)" in res and "STRESS (terzile alto)" in res:
        bc = res["CALMA  (terzile basso)"]["logamt_c"]; bs = res["STRESS (terzile alto)"]["logamt_c"]
        ok = (bc[0] > 0 and bc[1] > 1.6) and (bs[0] < 0 and bs[1] < -1.6)
        P(f"\n  VERDETTO P1: {'CONFERMATA' if ok else 'NON confermata'}")
        P( "    la storia richiede ENTRAMBI i segni. Se in calma l'effetto e' nullo, la meta'")
        P( "    'domanda' non regge e la ricchezza in regime normale resta senza meccanismo.")

    P("\n" + "="*76)
    P("P2  DETENZIONI BANCARIE EFFETTIVE (API BCE)")
    P("    atteso: banche riducono il portafoglio sovrano -> CCT si sconta -> coef POSITIVO")
    P("="*76)
    if X is not None and "mfi_govt_holdings_it" in X.columns:
        H = X[["mfi_govt_holdings_it"]].copy()
        if "mfi_total_assets_it" in X.columns:
            H["share"] = X.mfi_govt_holdings_it / X.mfi_total_assets_it * 100
        H["d_hold"] = np.log(X.mfi_govt_holdings_it).diff(12) * 100     # variazione annua %
        Hm = H.resample("ME").last(); Hm["ym"] = Hm.index.to_period("M")
        dd = d.copy(); dd["ym"] = dd.date.dt.to_period("M")
        dd = dd.merge(Hm[["d_hold","share","ym"]], on="ym", how="left").dropna(subset=["d_hold"])
        if len(dd) > 300:
            run(dd, "basis_p ~ d_hold + tau + I(tau**2) + C(mon)", ["d_hold"], "  variazione detenzioni")
            run(dd, "basis_p ~ d_hold + sov_swap + tau + I(tau**2) + C(mon)",
                ["d_hold","sov_swap"], "  + stress sovrano")
            run(dd, "basis_p ~ d_hold + sov_swap + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
                ["d_hold","sov_swap"], "  + eff. fissi CCT")
            ddd = dd.copy(); ddd["logamt_c"]=ddd.logamt-ddd.logamt.mean()
            ddd["dh_c"]=ddd.d_hold-ddd.d_hold.mean()
            run(ddd, "basis_p ~ dh_c*logamt_c + sov_swap + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
                ["dh_c","dh_c:logamt_c"], "  interazione con dimensione")
            P( "    [se le detenzioni contano AL NETTO dello stress sovrano, il canale clientela")
            P( "     e' diretto e non un riflesso del sovrano: e' il test che rende la storia")
            P( "     un meccanismo invece di una correlazione]")
        else: P("  dati insufficienti dopo il merge")
    else:
        P("  [ecb_series.csv assente o senza mfi_govt_holdings_it: lanciare 02c]")

    P("\n" + "="*76)
    P("P3  HAIRCUT -- la ricchezza in calma cresce con la scadenza?")
    P("    i titoli a tasso variabile prendono l'haircut 0-1y a QUALSIASI scadenza, quindi")
    P("    il vantaggio sul BTP CRESCE col tenor: atteso ricchezza crescente in scadenza")
    P("="*76)
    Bc = B.merge(d[["CCT_ISIN","date","sov_swap"]].drop_duplicates(), on=["CCT_ISIN","date"], how="left")
    calm = Bc[Bc.sov_swap <= lo]
    P(f"  {'fascia':>10}{'base bp':>10}{'base prezzo':>13}{'n':>10}")
    prof = []
    for a,b in [(0.5,1.5),(1.5,3),(3,4.5),(4.5,8)]:
        w = calm[(calm.tau_cct>=a)&(calm.tau_cct<b)]
        if len(w) < 200: continue
        prof.append((a, w.basis3_y.median()))
        P(f"  {f'{a}-{b}y':>10}{w.basis3_y.median():>10.1f}{w.basis3_p.median():>13.3f}{len(w):>10,}")
    if len(prof) >= 3:
        rich_short, rich_long = -prof[0][1], -prof[-1][1]     # ricchezza = -base in bp
        P(f"\n  ricchezza corto {rich_short:+.1f} bp vs lungo {rich_long:+.1f} bp")
        P(f"  VERDETTO P3: {'compatibile con haircut' if rich_long > rich_short else 'INCOMPATIBILE con haircut'}")
        P( "    se la ricchezza CALA con la scadenza, il canale collaterale non spiega il")
        P( "    regime calmo, e il profilo indica invece domanda di QUASI-CASSA sul breve.")

    P("\n" + "="*76)
    P("P4  DOVE SI CONCENTRA LO SCONTO NELLO STRESS")
    P("="*76)
    stress = Bc[Bc.sov_swap >= hi]
    P(f"  {'fascia':>10}{'calma bp':>11}{'stress bp':>11}{'differenza':>12}")
    for a,b in [(0.5,1.5),(1.5,3),(3,4.5),(4.5,8)]:
        wc = calm[(calm.tau_cct>=a)&(calm.tau_cct<b)]; ws = stress[(stress.tau_cct>=a)&(stress.tau_cct<b)]
        if len(wc) < 200 or len(ws) < 200: continue
        P(f"  {f'{a}-{b}y':>10}{wc.basis3_y.median():>11.1f}{ws.basis3_y.median():>11.1f}"
          f"{ws.basis3_y.median()-wc.basis3_y.median():>12.1f}")
    P( "    [il passivo bancario e' a breve: se il matching guida la detenzione, lo sconto")
    P( "     da stress deve concentrarsi sulle scadenze che le banche usano davvero]")

    P("\n" + "="*76)
    P("P5  DISTINZIONE DA PAPER 1 (capitale lento degli arbitraggisti)")
    P("="*76)
    P("  Richiede le serie di mispricing di Paper 1 (BTP Italia, CDS-Bond, iTraxx Skew).")
    P("  Test: correlazione delle variazioni mensili fra la base CCT e ciascuna delle tre,")
    P("  contro la correlazione media FRA le tre. Se la base CCT co-muove SENSIBILMENTE")
    P("  MENO, la clientela e' distinta e la storia bancaria e' identificata per esclusione;")
    P("  se co-muove come le altre, il canale e' il capitale lento generico e la storia")
    P("  bancaria e' superflua. E' l'estensione diretta della fonte di identificazione di")
    P("  Paper 1 -- l'eterogeneita' della base di detentori -- ed e' il test che vale di piu':")
    P("  usa dati che gia' esistono e distingue le due letture in modo netto.")
    P("  -> copiare le serie in PROC/paper1_mispricing.csv (colonne: date, btpitalia, cdsbond, itraxx)")

    P("\n" + "="*76)
    P("COME LEGGERE LA BATTERIA")
    P("="*76)
    P("  P1 confermata + P2 confermata  -> la storia regge: si puo' portare a Rebonato.")
    P("  P1 a meta' (solo stress)       -> meta' storia: lo sconto da stress e' documentato,")
    P("                                    la ricchezza in calma resta senza meccanismo.")
    P("                                    Serve un canale alternativo (P3) o si dichiara.")
    P("  P2 non confermata              -> il canale bancario e' inferito dalla dimensione,")
    P("                                    non misurato: e' correlazione, non meccanismo.")
    P("  P5 e' il discriminante rispetto a Paper 1 e va fatto prima di proporre il paper.")
    save_txt("14_story_tests.txt", L); print("\n".join(L))
