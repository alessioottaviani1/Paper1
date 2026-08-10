"""33 - ESPERIMENTI NATURALI: la clientela cambia per decreto, la segmentazione la segue?

PERCHE' QUESTO E' IL PEZZO PIU' PREZIOSO CHE MANCA AL PAPER.
La storia del paper e' "un oggetto, due clientele". Oggi e' sostenuta da un ORDINAMENTO
CROSS-SECTION: il Giappone e' il piu' integrato, la sterlina il meno, e le due curve dentro
la stessa valuta differiscono. Un referee puo' attribuire qualunque ordinamento cross-section
a differenze istituzionali generiche, ed e' esattamente l'obiezione che la Sezione 6.6 e' stata
scritta per rispondere -- ma la 31 ha mostrato che in VARIAZIONI quel contrasto non e'
significativo (USD p=0.82, EUR 0.22, GBP 0.066, JPY 0.040 con segno invertito). L'identificazione
bandiera e' piu' fragile di quanto il documento affermi.

Esiste una fonte di identificazione molto piu' forte, gia' nei dati, mai usata: DUE CAMBI
ESOGENI DELLA CLIENTELA, ciascuno datato e non scelto da noi.

  (A) YIELD CURVE CONTROL. La Banca del Giappone introduce il controllo della curva il
      21 settembre 2016 e lo abbandona il 19 marzo 2024. Per sette anni e mezzo un attore
      unico pinna la curva JGB a una funzione di reazione dichiarata. Il paper AFFERMA che
      il JGB e' la curva piu' integrata "dove la BoJ conduce YCC". Se e' vero, l'integrazione
      deve COMPARIRE nel 2016 e SPARIRE nel 2024. Se invece e' una proprieta' permanente del
      mercato giapponese, non si muove -- e la spiegazione YCC va ritirata.

  (B) LA CRISI LDI. Fra il 23 settembre e il 14 ottobre 2022 i fondi LDI britannici subiscono
      richieste di margine, la Banca d'Inghilterra interviene, e la leva del settore viene
      ridotta strutturalmente dopo. Il paper attribuisce il co-movimento negativo degli swap
      in sterlina alla copertura di duration delle casse pensione (Klingler-Sundaresan). Se e'
      vero, la segmentazione degli SWAP deve INDEBOLIRSI dopo l'ottobre 2022, e quella dei
      GILT no o molto meno.

Queste NON sono correlazioni fra mercati: sono variazioni nel tempo DENTRO un mercato,
generate da decisioni di politica monetaria e da una crisi di margine, entrambe plausibilmente
esogene rispetto al prezzo relativo della convessita'. E' la differenza fra "i mercati con
clientela vincolata sono piu' segmentati" e "quando la clientela cambia, la segmentazione
cambia con lei". La seconda e' un'affermazione causale che si puo' difendere.

IL PLACEBO E' PARTE DEL TEST, NON UN'AGGIUNTA. Le stesse date di rottura vengono applicate a
dollaro ed euro, dove nulla di rilevante e' accaduto in quei mesi. Se la segmentazione si muove
anche li', ci si sta accorgendo di un cambio di regime globale (volatilita', inflazione,
politica monetaria sincronizzata) e non della clientela. Il test e' che il Giappone si muova
alle date giapponesi, la sterlina alla data britannica, e gli altri no.

PANNELLI
  [1] YCC: C3 (livelli e variazioni) per JPY swap e JGB nei tre regimi pre/durante/post
  [2] YCC: la gara di previsione (32) nei tre regimi -- l'inversione giapponese e' YCC?
  [3] LDI: C3 per GBP swap e gilt prima e dopo ottobre 2022, e il contrasto fra i due
  [4] PLACEBO: le stesse date su USD/EUR e le loro curve governative
  [5] Inferenza: block bootstrap appaiato sulla DIFFERENZA fra regimi, con le stesse
      convenzioni della 31 (blocchi di 12 mesi, ricampionamento congiunto)

REGOLA DI DECISIONE, fissata prima di guardare l'output. Se il Giappone si muove alle date
giapponesi e il placebo tace, questo diventa la Sezione 2 del paper e la 6.6 scende a corollario.
Se il Giappone non si muove, la spiegazione YCC va ritirata dal testo -- e' oggi asserita, non
testata. Se la sterlina si muove ma il Giappone no (o viceversa), si riporta l'asimmetria e si
dichiara che un solo esperimento identifica.

Output: results/33_natural_experiments.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_vols

print("== 33 natural experiments ==")
rng = np.random.default_rng(SEED)
L = []; P = L.append
P("=== 33 ESPERIMENTI NATURALI: YCC e LDI come spostamenti esogeni della clientela ===")
P("")

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
W   = pd.read_csv(PROC/"vols_monthly.csv",  index_col=0, parse_dates=True)
mid = pd.read_csv(PROC/"mids_daily.csv",    index_col=0, parse_dates=True)

# ------------------------------------------------------------------ date, dichiarate
YCC_IN  = pd.Timestamp("2016-09-30")   # BoJ introduce lo YCC, 21 settembre 2016
YCC_OUT = pd.Timestamp("2024-03-31")   # BoJ abbandona lo YCC, 19 marzo 2024
LDI     = pd.Timestamp("2022-10-31")   # crisi LDI: 23 set - 14 ott 2022, poi deleveraging
BLOCK, B = 12, 5000
P(f"[0] date di rottura (esogene, non stimate): YCC {YCC_IN.date()} -> {YCC_OUT.date()} ; "
  f"LDI {LDI.date()}")
P("    nota: lo YCC e' stato allentato per gradi (dicembre 2022, luglio e ottobre 2023) prima")
P("    dell'uscita formale. L'allentamento lavora CONTRO di noi: sposta parte del regime YCC")
P("    verso il non-YCC e attenua il contrasto misurato. Nessuna data e' scelta dai dati.")
P("")

def iv(mkt, exp="3M", ten="10Y"):
    c = f"{IVMAP[mkt]}_{exp}_{ten}_NORM"
    return W[c] if c in W.columns else None

def c3(mkt, a=None, b=None, mode="delta"):
    """Correlazione fra sigma_BE e sigma_IV nella finestra [a,b]. Ritorna (rho, n)."""
    v = iv(mkt)
    if v is None or mkt not in sbe.columns: return np.nan, 0
    j = pd.concat([sbe[mkt], v], axis=1).dropna()
    if a is not None: j = j[j.index >= a]
    if b is not None: j = j[j.index <= b]
    if mode == "delta": j = j.diff().dropna()
    if len(j) < 24: return np.nan, len(j)
    return j.iloc[:,0].corr(j.iloc[:,1]), len(j)

def boot_diff(x1, y1, x2, y2, b=B):
    """Distribuzione bootstrap a blocchi della differenza rho(finestra1) - rho(finestra2).
    Le due finestre sono disgiunte, quindi si ricampionano indipendentemente."""
    def one(x, y):
        n = len(x); nb = max(int(np.ceil(n/BLOCK)), 1)
        pool = np.arange(0, max(n-BLOCK, 0)+1)
        st = rng.choice(pool, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, s+BLOCK) for s in st])[:n]
        xs, ys = x[idx], y[idx]
        if np.std(xs) == 0 or np.std(ys) == 0: return np.nan
        return np.corrcoef(xs, ys)[0,1]
    out = np.array([one(x1,y1) - one(x2,y2) for _ in range(b)])
    return out[np.isfinite(out)]

def pair_series(mkt, a=None, b=None, mode="delta"):
    v = iv(mkt)
    j = pd.concat([sbe[mkt], v], axis=1).dropna()
    if a is not None: j = j[j.index >= a]
    if b is not None: j = j[j.index <= b]
    if mode == "delta": j = j.diff().dropna()
    return j.iloc[:,0].values, j.iloc[:,1].values

def regime_row(mkt, mode):
    pre  = c3(mkt, None,    YCC_IN,  mode)
    dur  = c3(mkt, YCC_IN,  YCC_OUT, mode)
    post = c3(mkt, YCC_OUT, None,    mode)
    return pre, dur, post

# ------------------------------------------------------------------ [1] YCC e la segmentazione
P("[1] YCC: la curva JGB e' integrata PERCHE' la BoJ la pinna?")
P("    predizione: rho ALTA nel regime YCC, PIU' BASSA prima e dopo. Il JGB deve muoversi")
P("    piu' dello swap in yen, perche' e' la curva su cui lo YCC opera.")
for mode in ("liv", "delta"):
    P(f"    -- {'LIVELLI' if mode=='liv' else 'VARIAZIONI'} --")
    P(f"{'mercato':9}{'pre-YCC':>10}{'n':>5}{'YCC':>10}{'n':>5}{'post-YCC':>11}{'n':>5}"
      f"{'YCC-pre':>10}{'p boot':>9}")
    for mkt in ("JPY", "JPgovt"):
        if mkt not in sbe.columns: continue
        (r0,n0), (r1,n1), (r2,n2) = regime_row(mkt, mode)
        if not np.isfinite(r0) or not np.isfinite(r1):
            P(f"{mkt:9}  campione insufficiente"); continue
        x1,y1 = pair_series(mkt, YCC_IN, YCC_OUT, mode)
        x0,y0 = pair_series(mkt, None,   YCC_IN,  mode)
        bd = boot_diff(x1,y1,x0,y0)
        obs = r1 - r0
        p = float(np.mean(np.abs(bd - bd.mean()) >= abs(obs))) if len(bd) else np.nan
        P(f"{mkt:9}{r0:+10.2f}{n0:5d}{r1:+10.2f}{n1:5d}{r2:+11.2f}{n2:5d}{obs:+10.2f}{p:9.3f}")
P("")

# ------------------------------------------------------------------ [2] YCC e la previsione
P("[2] YCC: l'inversione giapponese della gara di previsione e' un fenomeno YCC?")
P("    la 32 trova che in Giappone la CURVA prevede la volatilita' realizzata (b 0.14 [4.6],")
P("    R2 0.30 sullo swap; 0.10 [4.8] sul JGB) mentre l'opzione no. Se e' la firma della")
P("    clientela YCC, la pendenza deve essere alta nel regime e crollare fuori.")
HOR = 63
def rv_fwd(mkt):
    lg = MK[mkt][0][1] if mkt in MK else None
    if lg is None or lg not in mid.columns: return None
    dy = (mid[lg]/100.0).dropna().diff()
    return (dy.shift(-1).rolling(HOR).std().shift(-(HOR-1))*np.sqrt(252)*1e4).resample("ME").last()

def mz(mkt, a, b):
    """Pendenze di Mincer-Zarnowitz separate nella finestra. Ritorna (b_BE, b_IV, n)."""
    f = rv_fwd(mkt); v = iv(mkt)
    if f is None or v is None: return np.nan, np.nan, 0
    j = pd.concat([f.rename("RV"), sbe[mkt].rename("BE"), v.rename("IV")], axis=1).dropna()
    if a is not None: j = j[j.index >= a]
    if b is not None: j = j[j.index <= b]
    if len(j) < 24: return np.nan, np.nan, len(j)
    o = lambda c: np.linalg.lstsq(np.column_stack([np.ones(len(j)), j[c].values]),
                                  j.RV.values, rcond=None)[0][1]
    return o("BE"), o("IV"), len(j)

P(f"{'mercato':9}{'regime':12}{'b_BE':>9}{'b_IV':>9}{'n':>6}")
for mkt in ("JPY", "JPgovt"):
    if mkt not in sbe.columns: continue
    for lab, a, b in (("pre-YCC", None, YCC_IN), ("YCC", YCC_IN, YCC_OUT),
                      ("post-YCC", YCC_OUT, None)):
        bb, bi, n = mz(mkt, a, b)
        P(f"{mkt:9}{lab:12}{bb:9.2f}{bi:9.2f}{n:6d}")
P("    lettura: se b_BE e' alta solo dentro il regime, la curva giapponese prevede la")
P("    volatilita' perche' un attore unico la fissa -- non per una proprieta' del mercato.")
P("")

# ------------------------------------------------------------------ [3] LDI
P("[3] LDI: il co-movimento negativo in sterlina si indebolisce dopo il deleveraging?")
P("    predizione: lo SWAP si muove (e' dove le casse pensione coprono la duration), il GILT")
P("    molto meno. Il contrasto swap-meno-gilt deve ATTENUARSI dopo ottobre 2022.")
for mode in ("liv", "delta"):
    P(f"    -- {'LIVELLI' if mode=='liv' else 'VARIAZIONI'} --")
    P(f"{'mercato':9}{'pre-LDI':>10}{'n':>5}{'post-LDI':>11}{'n':>5}{'diff':>9}{'p boot':>9}")
    for mkt in ("GBP", "UKgovt"):
        if mkt not in sbe.columns: continue
        (r0,n0), (r1,n1) = c3(mkt, None, LDI, mode), c3(mkt, LDI, None, mode)
        if not np.isfinite(r0) or not np.isfinite(r1):
            P(f"{mkt:9}  campione insufficiente (post-LDI e' corto: dichiararlo)"); continue
        x1,y1 = pair_series(mkt, LDI, None, mode)
        x0,y0 = pair_series(mkt, None, LDI, mode)
        bd = boot_diff(x1,y1,x0,y0); obs = r1 - r0
        p = float(np.mean(np.abs(bd - bd.mean()) >= abs(obs))) if len(bd) else np.nan
        P(f"{mkt:9}{r0:+10.2f}{n0:5d}{r1:+11.2f}{n1:5d}{obs:+9.2f}{p:9.3f}")
P("    CAVEAT DA SCRIVERE NEL PAPER: il campione post-LDI e' di poche decine di mesi. Questo")
P("    esperimento e' sotto-potenziato per costruzione e va riportato come suggestivo, non")
P("    come prova. Lo YCC, con sette anni e mezzo di regime, non ha questo problema.")
P("")

# ------------------------------------------------------------------ [4] PLACEBO
P("[4] PLACEBO: le stesse date su mercati dove non e' successo nulla")
P("    se dollaro ed euro si muovono alle date giapponesi o britanniche, stiamo misurando un")
P("    cambio di regime globale (volatilita', inflazione, politica sincronizzata) e non la")
P("    clientela. Il test e' che QUI non succeda niente.")
P(f"{'mercato':9}{'rottura':10}{'prima':>9}{'dopo':>9}{'diff':>9}{'p boot':>9}")
for lab, dt in (("YCC-in", YCC_IN), ("YCC-out", YCC_OUT), ("LDI", LDI)):
    for mkt in ("USDswap", "USTgovt", "EUR", "DEgovt"):
        if mkt not in sbe.columns: continue
        (r0,n0), (r1,n1) = c3(mkt, None, dt, "delta"), c3(mkt, dt, None, "delta")
        if not np.isfinite(r0) or not np.isfinite(r1): continue
        x1,y1 = pair_series(mkt, dt, None, "delta")
        x0,y0 = pair_series(mkt, None, dt, "delta")
        bd = boot_diff(x1,y1,x0,y0); obs = r1 - r0
        p = float(np.mean(np.abs(bd - bd.mean()) >= abs(obs))) if len(bd) else np.nan
        P(f"{mkt:9}{lab:10}{r0:+9.2f}{r1:+9.2f}{obs:+9.2f}{p:9.3f}")
P("")

# ------------------------------------------------------------------ [5] sintesi
P("[5] SINTESI: il quadro che il paper puo' rivendicare")
P("    Il criterio non e' che ogni singolo p sia sotto 0.05 -- con quattro test e campioni")
P("    di poche decine di mesi non lo sarebbe comunque. Il criterio e' la CONFIGURAZIONE:")
P("      (a) il Giappone si muove alle date giapponesi, e il JGB piu' dello swap in yen;")
P("      (b) la sterlina si muove alla data britannica, e lo swap piu' del gilt;")
P("      (c) il placebo tace su entrambe.")
P("    Se (a)+(c) reggono, il paper ha un esperimento naturale su sette anni e mezzo di")
P("    regime e il meccanismo smette di essere un ordinamento cross-section interpretato.")
P("    Se (a) non regge, la frase 'where the Bank of Japan conducts yield-curve control'")
P("    va tolta dal paper: oggi e' asserita e non testata, ed e' l'unica spiegazione")
P("    causale che il documento offra per l'inversione giapponese.")
save_txt("33_natural_experiments.txt", L); print("\n".join(L))
