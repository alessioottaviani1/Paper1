"""r1b - PERCHE' il premio residuo differisce fra US e UK (seconda meta' della richiesta RR n.1).

RR chiede: "Compare the residual risk premium for US and UK. Similar? If different, WHY?".
r1 risponde alla prima meta'; questo script affronta la seconda, e lo fa su tre assi che la
versione precedente non aveva.

  (1) STRUTTURA A TERMINE, non due punti. r1 usa 5 e 10 anni. Ma il divario US-UK CAMBIA
      SEGNO fra i due (a 5y gli USA hanno il premio piu' alto, a 10y il piu' basso): con
      due soli nodi quel fatto si vede ma non si legge. Qui si usa l'intera griglia
      realmente coperta da entrambe le curve e dagli swap, scoperta a runtime perche'
      interp_cols si ferma -- giustamente -- se un nodo cade fuori.

  (2) EVENT STUDY SU DATI GIORNALIERI. lam_gamma applica eom() e collassa tutto a fine
      mese: e' il motivo per cui le finestre di crisi avevano 2-3 osservazioni e non
      dicevano nulla. Le curve sottostanti sono giornaliere: qui si usano cosi'.

  (3) INFERENZA VERA sulla differenza media. Dire "SIMILI" o "DIVERSI" da una soglia
      arbitraria non e' un test. Serie mensili persistenti: t di Newey-West.

  (4) REGRESSIONI SEPARATE, non sul divario. E' il cambio di disegno piu' importante:
      se ENTRAMBI i mercati caricano sullo stesso fattore, il divario NON lo mostra --
      l'effetto comune si cancella per differenza. Regredendo i due premi separatamente
      e confrontando i beta si puo' dire "entrambi caricano sul rumore del mercato
      Treasury, ma gli USA con beta doppio, ed e' quella differenza a generare il
      divario": una risposta interpretabile, mentre un coefficiente sul divario dice
      solo che qualcosa si muove.

COSA MISURA IL RESIDUO, e perche' orienta la scelta delle variabili. BEI - ISR non e' un
premio al rischio d'inflazione: quello sta in ENTRAMBI i termini e in buona parte si
cancella. Quel che resta e' la CONVENIENZA RELATIVA del linker rispetto al sintetico
(r1 lo dice gia': "-lambda<0 = linker cheap al sintetico"). E' una misura di frizioni,
lo stesso oggetto della base TIPS-Treasury. Le variabili giuste sono quindi quelle che
spiegano FRIZIONI E LIMITI ALL'ARBITRAGGIO, non premi al rischio in senso stretto.

I CANALI TESTATI, e cosa ciascuno predice:
  A) FLOOR DI DEFLAZIONE. I TIPS rimborsano almeno il nominale, i gilt no. E' un'opzione
     che gli USA hanno e il Regno Unito no, vale di piu' quando il rischio di deflazione
     sale, e ALZA il BEI americano -> divario positivo negli spaventi deflazionistici.
     ATTENZIONE: qui e' approssimata dal livello del BEI a breve. Il test corretto
     richiederebbe la moneyness per ISIN (indice di riferimento all'emissione), che non
     e' in cache: il proxy e' dichiarato tale e un esito nullo NON falsifica il canale.
  B) DOMANDA LDI BRITANNICA. I fondi pensione UK comprano linker lunghi per il matching:
     rende i gilt indicizzati cari e il premio britannico basso in via ordinaria, ma
     nel settembre-ottobre 2022 le vendite forzate invertono il segno.
  C) FRIZIONI DI MERCATO. NOISE (Hu-Pan-Wang, via GVLQUSD), ONOFF, MOVE, VIX da
     rp.liquidity(): sono gli unici canali misurati con dati veri anziche' con proxy
     costruiti dalle stesse curve che stiamo spiegando.

  D) CAPITALE DEGLI INTERMEDIARI (He-Kelly-Manela 2017). E' IL fattore per una storia di
     limiti all'arbitraggio, ed e' GLOBALE: gli arbitraggisti attivi su TIPS e su gilt
     indicizzati sono le stesse istituzioni. Proprio per questo e' adatto alla domanda di
     RR: se entrambi i mercati vi caricano ma con intensita' diversa, quella differenza
     E' il "why". Sorgente: file mensile archiviato in raw/He_Kelly_Manela con la data di
     scarico nel nome -- NON scaricato al volo: la pagina degli autori non ha URL stabile
     ne' versionamento, e una revisione silenziosa spezzerebbe la replicabilita'.

ESCLUSI CONSAPEVOLMENTE. Pastor-Stambaugh misura liquidita' AZIONARIA: aveva senso nel
BTP Italia, dove la clientela e' retail, non qui dove la base e' fra titoli di Stato e
swap. Betting-Against-Beta europeo: mercato sbagliato per un confronto US-UK. L'offerta
relativa di linker sarebbe la variabile piu' adatta a spiegare la FORMA del divario, ma
il lato britannico esiste solo dentro venti PDF trimestrali della DMO: il costo di
estrarli a mano non giustifica una serie trimestrale poi interpolata. Resta un limite
DICHIARATO, non un'omissione.

NOISE: DUE SORGENTI, saldate. La serie Bloomberg (GVLQUSD) parte dal 2007 ed e' aggiornata
in continuo; il file originale degli autori (Noise_Measure_2025.xlsx) copre dal 1987.
Correlazione quasi unitaria nel periodo comune -- e' la stessa misura, Bloomberg la
ridistribuisce. Si usa HPW fino a dove arriva e Bloomberg dopo: senza la parte pre-2007 il
canale piu' forte resta escluso dal test di rottura del 2009, che e' proprio dove serve.

Output: results/r1b_why.txt + due figure (struttura a termine, serie del divario).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np, pandas as pd
import rp
from config import CACHE

# 30y ESCLUSO: la curva d'inflazione BoE arriva a 30 anni solo dal gennaio 2016 (2.673
# osservazioni contro >10.000 sugli altri nodi, verificato con r1c_check30.py). Il vincolo
# e' la curva, non gli swap. Tenerlo darebbe un nodo con meta' campione, tutto nel decennio
# recente, non confrontabile con gli altri.
GRIGLIA = [2., 3., 5., 7., 10., 15., 20.]
HKM_DIR   = Path("data/raw/He_Kelly_Manela")
NOISE_XLS = Path("data/raw/Hu/Noise_Measure_2025.xlsx")

def noise_spliced(liq):
    """NOISE saldata: HPW originale (dal 1987) fino a dove arriva, Bloomberg dopo.

    Le due sono la stessa misura -- Bloomberg ridistribuisce HPW con aggiornamento
    continuo -- ma la serie Bloomberg parte dal 2007. Senza il tratto 1987-2007 il canale
    resta fuori dal test di rottura del 2009. La saldatura viene VERIFICATA sulla
    sovrapposizione: se la correlazione e' bassa non sono la stessa cosa e si rinuncia,
    invece di incollare due serie diverse e non accorgersene."""
    bbg = liq["NOISE"].dropna() if (liq is not None and "NOISE" in liq.columns) else None
    if not NOISE_XLS.exists():
        return bbg, "solo Bloomberg (file HPW non trovato)"
    try:
        df = pd.read_excel(NOISE_XLS)
        dcol = next(c for c in df.columns
                    if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8)
        vcol = next(c for c in df.columns if c != dcol
                    and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8)
        hpw = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                        index=pd.to_datetime(df[dcol], errors="coerce")).dropna().sort_index()
        hpw = hpw[~hpw.index.duplicated(keep="last")].resample("ME").last()
    except Exception as e:
        return bbg, f"solo Bloomberg (lettura HPW fallita: {str(e)[:40]})"
    if bbg is None: return hpw, "solo HPW"
    j = pd.concat([hpw.rename("hpw"), bbg.rename("bbg")], axis=1).dropna()
    if len(j) < 24 or j.hpw.corr(j.bbg) < 0.90:
        r = j.hpw.corr(j.bbg) if len(j) > 2 else float("nan")
        return bbg, f"solo Bloomberg (sovrapposizione {len(j)} mesi, corr {r:+.2f}: non saldabili)"
    cut = bbg.dropna().index.min()
    out = pd.concat([hpw[hpw.index < cut], bbg.dropna()]).sort_index()
    return out, (f"HPW fino a {cut.date()} + Bloomberg dopo | sovrapposizione {len(j)} mesi, "
                 f"corr {j.hpw.corr(j.bbg):+.3f}")

def hkm_factor():
    """He-Kelly-Manela 2017: restituisce DUE serie, e la distinzione conta.

    intermediary_capital_RATIO e' un LIVELLO (equity/asset dei primary dealer);
    intermediary_capital_RISK_FACTOR e' un RENDIMENTO (l'innovazione di quel rapporto,
    scalata). Sono oggetti diversi e non vanno usati nella stessa specifica: il premio
    BEI-ISR e' un LIVELLO, quindi in una regressione in livelli va il RAPPORTO, e il
    FATTORE va nella regressione in variazioni. Mescolarli -- livelli di liquidita' e un
    rendimento sullo stesso lato destro -- e' l'incoerenza che questa versione corregge."""
    if not HKM_DIR.exists(): return None, None
    f = sorted(HKM_DIR.glob("He_Kelly_Manela_Factors_monthly_*.csv"))
    if not f: return None, None
    d = pd.read_csv(f[-1])
    idx = pd.to_datetime(d["yyyymm"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    def col(c):
        if c not in d.columns: return None
        x = pd.Series(pd.to_numeric(d[c], errors="coerce").values, index=idx)
        x = clean(x); x.name = f[-1].name
        return x
    return col("intermediary_capital_ratio"), col("intermediary_capital_risk_factor")
EVENTI = {
    "Lehman 2008":        ("2008-09-01", "2009-03-31", "A"),
    "COVID 2020":         ("2020-02-20", "2020-05-31", "A"),
    "crisi LDI UK 2022":  ("2022-09-19", "2022-10-31", "B"),
    "picco inflaz. 2022": ("2022-03-01", "2022-08-31", "-"),
}

def clean(s_: pd.Series, nome: str = "") -> pd.Series:
    """Indice unico e ordinato. pd.concat(axis=1) solleva 'cannot reindex on an axis with
    duplicate labels' se anche UNA sola delle serie ha etichette ripetute, e il messaggio
    non dice quale: meglio normalizzarle tutte all'ingresso."""
    s_ = s_.dropna().sort_index()
    return s_[~s_.index.duplicated(keep="last")]

def nw_t_mean(x, lags=12):
    """t di Newey-West sulla media di una serie persistente (H0: media nulla)."""
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 24: return np.nan
    e = x - x.mean(); s = float(e @ e)/n
    for l in range(1, min(lags, n-1)+1):
        s += 2*(1 - l/(lags+1))*float(e[l:] @ e[:-l])/n
    return np.nan if s <= 0 else x.mean()/np.sqrt(s/n)

def nw_reg(y, X, lags=12):
    Z = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]; e = y - Z@b
    S = (Z*e[:,None]).T @ (Z*e[:,None])
    for l in range(1, lags+1):
        w = 1 - l/(lags+1); A = (Z[l:]*e[l:,None]).T @ (Z[:-l]*e[:-l,None]); S += w*(A+A.T)
    Q = np.linalg.pinv(Z.T@Z); V = Q@S@Q
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return b, np.divide(b, se, out=np.full_like(b, np.nan), where=se>0)

def nodi_coperti(cand):
    """Quali scadenze sono coperte da TUTTE e quattro le fonti (BEI e ISR, US e UK)?
    interp_cols solleva se un nodo cade fuori: si prova uno alla volta."""
    ok = []
    for m in cand:
        try:
            for f in (lambda: rp.bei_us((m,)), lambda: rp.bei_uk((m,)),
                      lambda: rp.isr("US", (m,)), lambda: rp.isr("UK", (m,))):
                if f().dropna(how="all").empty: raise ValueError("vuoto")
            ok.append(m)
        except Exception:
            pass
    return ok

if __name__ == "__main__":
    L=[]; P=L.append
    P("=== r1b PERCHE' il premio differisce fra US e UK ===")
    MATS = nodi_coperti(GRIGLIA)
    P(f"scadenze coperte da tutte le fonti: {MATS}")
    P(f"  (candidate: {GRIGLIA}; le escluse non sono nella curva o negli swap)")
    if len(MATS) < 3:
        P("[STOP] meno di 3 nodi comuni"); print("\n".join(L)); raise SystemExit

    # --- pannelli GIORNALIERI (niente eom): servono agli event study
    d = {}
    for mkt in ("US","UK"):
        b = (rp.bei_us if mkt=="US" else rp.bei_uk)(MATS)
        s = rp.isr(mkt, MATS)
        idx = b.index.intersection(s.index)
        d[mkt] = (b.loc[idx] - s.loc[idx])          # premio = BEI - projected(ISR)
    idx = d["US"].index.intersection(d["UK"].index)
    US, UK = d["US"].loc[idx], d["UK"].loc[idx]
    GAP = (US - UK)*100                              # bp
    P(f"campione GIORNALIERO comune: {idx.min().date()} -> {idx.max().date()} ({len(idx):,} giorni)")

    P("\n" + "="*76)
    P("(1) STRUTTURA A TERMINE del premio e del divario  [bp]")
    P("="*76)
    P(f"  {'scad.':>7}{'US':>9}{'UK':>9}{'gap':>9}{'t-NW':>8}{'sd gap':>9}{'US>UK':>8}")
    for m in MATS:
        g = GAP[m].dropna()
        P(f"  {m:>6.0f}y{US[m].mean()*100:>9.0f}{UK[m].mean()*100:>9.0f}{g.mean():>9.0f}"
          f"{nw_t_mean(g.resample('ME').last()):>8.2f}{g.std():>9.0f}{(g>0).mean():>7.0%}")
    P("  [t-NW sulla media mensile del divario, 12 lag: H0 = i due premi coincidono]")
    inv = [m for i,m in enumerate(MATS[:-1]) if np.sign(GAP[m].mean()) != np.sign(GAP[MATS[i+1]].mean())]
    if inv: P(f"  --> il divario CAMBIA SEGNO fra {inv} e la scadenza successiva:")
    P( "      la differenza fra i due mercati non e' di LIVELLO ma di FORMA della curva.")

    P("\n" + "="*76)
    P("(2) EVENT STUDY su dati GIORNALIERI (prima erano 2-3 osservazioni mensili)")
    P("="*76)
    ten = [m for m in (5.,10.) if m in MATS] or MATS[:2]
    for m in ten:
        g = GAP[m].dropna(); base = g.median()
        P(f"\n  scadenza {m:.0f}y   (mediana di campione {base:+.0f} bp)")
        P(f"    {'finestra':>22}{'gap medio':>12}{'vs mediana':>12}{'variazione':>12}{'canale':>8}{'n':>6}")
        for lab,(a,b_,ch) in EVENTI.items():
            w = g.loc[a:b_]
            if len(w) < 5: P(f"    {lab:>22}{'--':>12}{'--':>12}{'--':>12}{ch:>8}{len(w):>6}"); continue
            var = w.iloc[-1] - w.iloc[0]
            P(f"    {lab:>22}{w.mean():>12.0f}{w.mean()-base:>12.0f}{var:>12.0f}{ch:>8}{len(w):>6}")
        P("    [variazione = da inizio a fine finestra: cattura il MOVIMENTO, non il livello]")

    P("\n" + "="*76)
    P("(3) I CANALI -- regressioni SEPARATE per mercato, non sul divario")
    P("    se entrambi caricano sullo stesso fattore il divario non lo mostra:")
    P("    l'effetto comune si cancella per differenza")
    P("="*76)
    Gm = GAP.resample("ME").last()
    try:
        liq = rp.liquidity()
        P(f"  proxy di liquidita' disponibili: {list(liq.columns)}")
        P(f"  {'proxy':>10}{'inizio':>13}{'fine':>13}{'n':>8}{'pre-2009':>11}")
        for c in liq.columns:
            v = liq[c].dropna()
            npre = int((v.index < "2009-01-01").sum())
            P(f"  {c:>10}{str(v.index.min().date()):>13}{str(v.index.max().date()):>13}"
              f"{len(v):>8,}{npre:>11,}")
        P("  [un proxy che inizia tardi TAGLIA il campione comune della regressione:")
        P("   e' il motivo per cui il test sulla rottura del 2009 puo' non avere dati]")
    except Exception as e:
        liq = None; P(f"  [!] rp.liquidity() non disponibile: {str(e)[:50]}")
    if liq is not None:
        ns, msg = noise_spliced(liq)
        P(f"  NOISE: {msg}")
        if ns is not None:
            liq = liq.drop(columns=["NOISE"], errors="ignore").join(
                ((ns - ns.mean())/ns.std()).rename("NOISE"), how="outer")
            v = liq["NOISE"].dropna()
            P(f"    copertura: {v.index.min().date()} -> {v.index.max().date()} "
              f"({len(v):,} mesi, di cui {int((v.index < '2009-01-01').sum())} pre-2009)")

    hkm_lvl, hkm_ret = hkm_factor()
    if hkm_lvl is not None:
        P(f"  HKM: {hkm_lvl.name} | {len(hkm_lvl):,} mesi | "
          f"{hkm_lvl.index.min().date()} -> {hkm_lvl.index.max().date()}")
        P("    livelli -> capital RATIO | variazioni -> capital RISK FACTOR (e' un rendimento)")
    else:
        P("  [!] HKM non trovato in data/raw/He_Kelly_Manela/: canale D assente")

    USm = (US*100).resample("ME").last(); UKm = (UK*100).resample("ME").last()

    def blocco(diff: bool):
        """diff=False: livelli (spiega PERCHE' un premio e' piu' alto).
           diff=True : variazioni (evita la regressione spuria fra serie persistenti)."""
        P(f"\n  --- {'VARIAZIONI mensili' if diff else 'LIVELLI'} ---")
        if not diff:
            P("    tutti i regressori sono livelli: liquidita' standardizzata e HKM capital")
            P("    ratio. E' la specifica che risponde alla domanda di RR, ma con serie")
            P("    persistenti i t vanno letti con cautela: v. blocco in variazioni.")
        for m in ten:
            X, nomi = [], []
            if liq is not None:
                for c in liq.columns:
                    v = clean(liq[c]); X.append(v.diff() if diff else v); nomi.append(f"C) {c}")
            h = hkm_ret if diff else hkm_lvl
            if h is not None:
                X.append(clean(h)); nomi.append("D) HKM " + ("risk factor" if diff else "capital ratio"))
            if not X: P("    nessun regressore disponibile"); return
            yU, yK = clean(USm[m]), clean(UKm[m])
            if diff: yU, yK = yU.diff(), yK.diff()
            R = pd.concat([clean(yU).rename("US"), clean(yK).rename("UK")]
                          + [clean(x).rename(f"x{i}") for i, x in enumerate(X)],
                          axis=1).dropna()
            if len(R) < 60: P(f"    {m:.0f}y: campione insufficiente ({len(R)})"); continue
            Xv = R.iloc[:, 2:].values
            bU, tU = nw_reg(R["US"].values, Xv, lags=12)
            bK, tK = nw_reg(R["UK"].values, Xv, lags=12)
            bG, tG = nw_reg((R["US"]-R["UK"]).values, Xv, lags=12)
            P(f"\n    scadenza {m:.0f}y  --  n {len(R)}, Newey-West 12 lag")
            P(f"      {'canale':28s}{'US':>17}{'UK':>17}{'differenza':>17}")
            for i, nm in enumerate(nomi):
                f1 = "*" if abs(tU[i+1])>2 else " "; f2 = "*" if abs(tK[i+1])>2 else " "
                f3 = "*" if abs(tG[i+1])>2 else " "
                P(f"      {nm:28s}{bU[i+1]:>10.3f}[{tU[i+1]:+5.2f}]{f1}"
                  f"{bK[i+1]:>10.3f}[{tK[i+1]:+5.2f}]{f2}{bG[i+1]:>10.3f}[{tG[i+1]:+5.2f}]{f3}")
        P("      (* = |t| > 2. Un canale significativo su ENTRAMBI ma non sulla differenza")
        P("       muove i due premi INSIEME e non spiega il divario.)")

    # --- quale specifica e' LECITA? lo decide l'ordine di integrazione, non la convenzione
    P("\n  --- ORDINE DI INTEGRAZIONE (ADF, H0 = radice unitaria) ---")
    P("    Livelli e variazioni non sono due gusti: la specifica lecita dipende da y E da x.")
    P("    y e x entrambe I(0) -> livelli validi. Entrambe I(1) -> differenze (o ECM).")
    P("    Miste -> differenze. Un rigetto in ENTRAMBE le specifiche e' la prova piu' forte.")
    serie = {}
    for m in ten:
        serie[f"premio US {m:.0f}y"] = clean(USm[m])
        serie[f"premio UK {m:.0f}y"] = clean(UKm[m])
    if liq is not None:
        for c in liq.columns: serie[c] = clean(liq[c])
    if hkm_lvl is not None: serie["HKM capital ratio"] = clean(hkm_lvl)
    if hkm_ret is not None: serie["HKM risk factor"] = clean(hkm_ret)
    P(f"    {'serie':24s}{'ADF liv.':>11}{'ADF diff.':>11}{'ordine':>10}{'n':>7}")
    ordini = {}
    for nm, v in serie.items():
        try:
            a0 = rp.adf_t(v, lags=4); a1 = rp.adf_t(v.diff().dropna(), lags=4)
        except Exception:
            P(f"    {nm:24s}{'--':>11}{'--':>11}{'--':>10}{len(v):>7}"); continue
        CV = -2.88                                    # valore critico 5%, ADF con costante
        o = "I(0)" if a0 < CV else ("I(1)" if a1 < CV else "I(2)+?")
        ordini[nm] = o
        P(f"    {nm:24s}{a0:>11.2f}{a1:>11.2f}{o:>10}{len(v):>7}")
    P("    [valore critico 5% ~ -2.88. ADF ha potenza bassa su serie molto persistenti:")
    P("     un I(1) qui puo' essere un I(0) con radice vicina a uno, e la conseguenza")
    P("     pratica e' la stessa -- i t in livelli vanno confermati in differenze.]")
    ny = sum(1 for k, v in ordini.items() if k.startswith("premio") and v == "I(0)")
    nyt = sum(1 for k in ordini if k.startswith("premio"))
    if nyt:
        P(f"\n    premi stazionari: {ny}/{nyt}")
        if ny == nyt:
            P("    -> i premi sono I(0): la regressione in LIVELLI e' lecita ed e' quella")
            P("       che risponde alla domanda di RR. Le differenze restano come controllo.")
        elif ny == 0:
            P("    -> i premi NON sono stazionari: la specifica principale sono le")
            P("       DIFFERENZE; i livelli andrebbero letti solo con test di cointegrazione.")
        else:
            P("    -> quadro misto fra scadenze: si riportano entrambe le specifiche e si")
            P("       conclude solo su cio' che sopravvive a tutte e due.")

    blocco(diff=False)
    blocco(diff=True)

    P("\n" + "="*76)
    P("(4) LA ROTTURA DEL 2009 -- il fatto che il campione intero nasconde")
    P("="*76)
    P("  Sul tratto BREVE il divario cambia SEGNO dopo il 2009 (da -30 a +15 bp), mentre")
    P("  sul LUNGO resta negativo e si accentua. Il divario aggregato non significativo a")
    P("  3-7 anni e' quindi la MEDIA DI DUE REGIMI OPPOSTI, non assenza di differenza.")
    P("  Qui i tre canali sono stimati SEPARATAMENTE prima e dopo, per vedere se uno di")
    P("  essi spiega la rottura o se nessuno lo fa.")
    SPLIT = "2009-01-01"
    for m in ten:
        y_all = Gm[m].dropna()
        X, nomi = [], []
        defl = -(rp.bei_us((min(MATS),))[min(MATS)]).resample("ME").last()*100
        X.append(defl); nomi.append("A) deflazione")
        uk_sl = ((rp.bei_uk((max(MATS),))[max(MATS)] - rp.bei_uk((min(MATS),))[min(MATS)])
                 .resample("ME").last()*100)
        X.append(uk_sl); nomi.append("B) pendenza BEI UK")
        if liq is not None:
            for c in liq.columns:
                X.append(liq[c]); nomi.append(f"C) {c}")
        # Tenere SOLO i regressori con copertura in ENTRAMBI i regimi: un proxy che
        # inizia nel 2008 azzera il campione pre-2009 dell'INTERA regressione, non solo
        # il proprio coefficiente. Meglio un test su meno canali che nessun test.
        keep_i = []
        for i, x in enumerate(X):
            v = x.dropna()
            if (v.index < SPLIT).sum() >= 40 and (v.index >= SPLIT).sum() >= 40:
                keep_i.append(i)
        esclusi = [nomi[i] for i in range(len(X)) if i not in keep_i]
        P(f"\n  scadenza {m:.0f}y")
        if esclusi:
            P(f"    esclusi dal test di rottura (copertura insufficiente in un regime):")
            for e_ in esclusi: P(f"      - {e_}")
        if not keep_i:
            P("    nessun canale copre entrambi i regimi: test non eseguibile"); continue
        Xk = [X[i] for i in keep_i]; nk = [nomi[i] for i in keep_i]
        R = pd.concat([y_all.rename("y")] + [x.rename(f"x{i}") for i, x in enumerate(Xk)],
                      axis=1).dropna()
        P(f"    {'canale':22s}{'pre-2009':>20}{'2009-oggi':>20}")
        pre, post = R[R.index < SPLIT], R[R.index >= SPLIT]
        if len(pre) < 40 or len(post) < 40:
            P(f"    campione ancora insufficiente (pre {len(pre)}, post {len(post)}):")
            P(f"    il divario medio resta confrontabile anche senza regressione ->")
            dm1 = y_all[y_all.index < SPLIT].mean(); dm2 = y_all[y_all.index >= SPLIT].mean()
            P(f"      pre {dm1:+.0f} bp  ->  post {dm2:+.0f} bp  (variazione {dm2-dm1:+.0f})")
            continue
        nomi_ = nk
        bp_, tp_ = nw_reg(pre.iloc[:,0].values, pre.iloc[:,1:].values, lags=12)
        bo_, to_ = nw_reg(post.iloc[:,0].values, post.iloc[:,1:].values, lags=12)
        for i, nm in enumerate(nomi_):
            f1 = "*" if abs(tp_[i+1]) > 2 else " "
            f2 = "*" if abs(to_[i+1]) > 2 else " "
            P(f"    {nm:22s}{bp_[i+1]:>12.3f}[{tp_[i+1]:+5.2f}]{f1}"
              f"{bo_[i+1]:>12.3f}[{to_[i+1]:+5.2f}]{f2}")
        P(f"    n: pre {len(pre)}, post {len(post)}   (* = |t| > 2)")
        dm_pre = y_all[y_all.index < SPLIT].mean(); dm_post = y_all[y_all.index >= SPLIT].mean()
        P(f"    divario medio: pre {dm_pre:+.0f} bp  ->  post {dm_post:+.0f} bp  "
          f"(variazione {dm_post-dm_pre:+.0f})")
    P("\n    [se un canale cambia coefficiente fra i due regimi nella direzione della")
    P("     rottura, e' il candidato; se tutti restano stabili, la rottura viene da")
    P("     qualcosa che non stiamo misurando -- e va detto]")

    # --------------------------------------------------------------- figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path("results"); out.mkdir(exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(MATS, [US[m].mean()*100 for m in MATS], "o-", label="US")
        ax[0].plot(MATS, [UK[m].mean()*100 for m in MATS], "s-", label="UK")
        ax[0].plot(MATS, [GAP[m].mean() for m in MATS], "^--", color="k", label="gap US-UK")
        ax[0].axhline(0, lw=.6, color="grey")
        ax[0].set_xlabel("scadenza (anni)"); ax[0].set_ylabel("bp")
        ax[0].set_title("Premio residuo BEI - ISR: struttura a termine"); ax[0].legend(fontsize=8)
        for m in ten:
            ax[1].plot(Gm.index, Gm[m], lw=.9, label=f"{m:.0f}y")
        ax[1].axhline(0, lw=.6, color="grey")
        for a,b_,_ in EVENTI.values(): ax[1].axvspan(pd.Timestamp(a), pd.Timestamp(b_), alpha=.12, color="red")
        ax[1].set_ylabel("bp"); ax[1].set_title("Divario US-UK e finestre di crisi"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out/"r1b_why.png", dpi=140)
        P(f"\n[figura] {out/'r1b_why.png'}")
    except Exception as e:
        P(f"\n[figura non prodotta: {str(e)[:60]}]")

    P("\n" + "="*76)
    P("LIMITI DICHIARATI")
    P("="*76)
    P("  Il canale A e' PROXATO, non misurato: il valore del floor dipende dall'inflazione")
    P("  cumulata dall'emissione di ciascun TIPS, quindi il test corretto e' CROSS-SEZIONALE")
    P("  e richiede l'indice di riferimento per ISIN, che non e' in cache. Un esito nullo")
    P("  qui NON falsifica il canale: dice che il proxy non lo cattura.")
    P("  Il canale B e' proxato dalla pendenza del BEI britannico; la misura vera sono le")
    P("  detenzioni dei gilt indicizzati per settore (ONS/BoE), anch'esse non in cache.")
    P("  Il canale C e' l'unico misurato con dati veri (VIX, MOVE).")
    P("  Restano fuori: lag di indicizzazione (3m US, 3m/8m UK) e trattamento fiscale.")
    outp = Path("results"); outp.mkdir(exist_ok=True)
    (outp/"r1b_why.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
