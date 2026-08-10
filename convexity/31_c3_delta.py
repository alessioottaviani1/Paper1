"""31 - L'IDENTIFICAZIONE WITHIN-CURRENCY, IN VARIAZIONI E CON INFERENZA APPAIATA.

IL PROBLEMA. La 6.4 stabilisce che il risultato centrale e' immune al premio per il rischio di
volatilita' PRECISAMENTE perche' e' un enunciato sulle VARIAZIONI: un premio spiega un livello,
non l'assenza di correlazione fra le variazioni di due prezzi. Ma la 6.6 -- che il documento
stesso chiama "the paper's cleanest identification" -- presenta la tabella swap-contro-governativo
(GBP -0.61 vs -0.18; JPY +0.65 vs +0.79) IN LIVELLI. E la 6.17 diagnostica il rigetto del modello
strutturale con il rango +0.80 "col negativo del livello C3": di nuovo livelli.

Quindi l'identificazione-bandiera e la diagnosi strutturale poggiano entrambe sull'oggetto che il
paper ha dichiarato NON identificato. Peggio: 03 stampa gia' entrambe le colonne, e in variazioni
il contrasto sterlina si assottiglia. E' il primo punto che un referee attento trova.

COSA MANCA, OLTRE ALLA COLONNA GIUSTA. Anche prendendo le variazioni, la 6.6 confronta due
correlazioni SENZA UN ERRORE STANDARD. "Tre volte piu' forte" non e' un test. E le due
correlazioni sono DIPENDENTI: condividono la stessa serie sigma_IV (una sola superficie swaption
per valuta -- che e' esattamente il pregio del disegno). Serve quindi un test appaiato.

QUESTO SCRIPT FA TRE COSE.
  [1] La tabella 6.6 in LIVELLI e VARIAZIONI affiancate, sul CAMPIONE COMUNE alle tre serie
      (sigma_BE swap, sigma_BE govt, sigma_IV), cosi' il confronto e' genuinamente appaiato.
  [2] Inferenza sulla DIFFERENZA rho_swap - rho_govt:
      (a) t di Williams (modifica di Hotelling) per correlazioni dipendenti che condividono
          una variabile -- il test analitico corretto per questo disegno;
      (b) block bootstrap appaiato: si ricampionano BLOCCHI DI DATE (12 mesi) e si ricalcolano
          ENTRAMBE le correlazioni sullo stesso ricampionamento, preservando l'autocorrelazione
          e la dipendenza fra le due stime.
  [3] Lo stesso contrasto su TUTTE le 16 celle del cubo, cosi' il risultato sterlina non poggia
      su una singola cella scelta.

REGOLA DI DECISIONE, fissata ex ante: se la differenza appaiata in VARIAZIONI e' significativa
in sterlina, la 6.6 e' la migliore identificazione del progetto e va guidata. Se non lo e', la
6.6 va riscritta come enunciato sui livelli CON il caveat del VRP dichiarato accanto, e il peso
della claim si sposta sulla gara di previsione (32).

Output: results/31_c3_delta.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_vols

print("== 31 c3 delta ==")
rng = np.random.default_rng(SEED)
L = []; P = L.append
P("=== 31 IDENTIFICAZIONE WITHIN-CURRENCY: livelli vs variazioni, con test appaiato ===")
P("")

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
W   = pd.read_csv(PROC/"vols_monthly.csv",  index_col=0, parse_dates=True)

# coppie (valuta -> curva swap, curva governativa) contro UNA superficie swaption
PAIRS = [("USD", "USDswap", "USTgovt"),
         ("EUR", "EUR",     "DEgovt"),
         ("GBP", "GBP",     "UKgovt"),
         ("JPY", "JPY",     "JPgovt")]
BLOCK = 12          # blocchi di 12 mesi: cattura l'autocorrelazione mensile del cuneo
B     = 5000

def williams_t(r12, r13, r23, n):
    """t di Williams per r12 vs r13 con la variabile 1 condivisa. df = n-3.
    r12 = corr(IV, BE_swap), r13 = corr(IV, BE_govt), r23 = corr(BE_swap, BE_govt)."""
    if n < 10: return np.nan
    R = (1 - r12**2 - r13**2 - r23**2) + 2*r12*r13*r23
    den = 2*R*(n-1)/(n-3) + ((r12 + r13)**2/4)*(1 - r23)**3
    if den <= 0: return np.nan
    return (r12 - r13)*np.sqrt((n-1)*(1 + r23)/den)

def paired_block_boot(x, y, z, nblk, b=B):
    """Ricampiona BLOCCHI DI RIGHE (le tre serie insieme) e ritorna la distribuzione di
    rho(z,x) - rho(z,y). x = BE swap, y = BE govt, z = IV. Tutte allineate, stesso indice."""
    n = len(z); nb = max(int(np.ceil(n/nblk)), 1)
    starts_pool = np.arange(0, max(n-nblk, 0)+1)
    out = np.empty(b)
    for k in range(b):
        st = rng.choice(starts_pool, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, s+nblk) for s in st])[:n]
        xs, ys, zs = x[idx], y[idx], z[idx]
        if np.std(xs) == 0 or np.std(ys) == 0 or np.std(zs) == 0:
            out[k] = np.nan; continue
        out[k] = np.corrcoef(zs, xs)[0,1] - np.corrcoef(zs, ys)[0,1]
    return out[np.isfinite(out)]

def cell(ccy, exp="3M", ten="10Y"):
    c = f"{ccy}_{exp}_{ten}_NORM"
    return W[c] if c in W.columns else None

# ---------------------------------------------------------------- [1]-[2] cella di riferimento
P("[1] CELLA DI RIFERIMENTO 3Mx10Y (NORM), campione comune alle tre serie")
P(f"{'valuta':7}{'N':>5} | {'LIV swap':>9}{'LIV govt':>9}{'diff':>8} | "
  f"{'DEL swap':>9}{'DEL govt':>9}{'diff':>8}{'[t Will]':>10}{'p boot':>8}")
SUMMARY = {}
for ccy, msw, mgv in PAIRS:
    iv = cell(ccy)
    if iv is None or msw not in sbe.columns or mgv not in sbe.columns:
        P(f"{ccy:7}  serie mancante"); continue
    al = pd.concat([sbe[msw], sbe[mgv], iv], axis=1).dropna()
    al.columns = ["SW", "GV", "IV"]
    if len(al) < 60: P(f"{ccy:7}  campione comune troppo corto ({len(al)})"); continue
    # livelli
    lsw = al.IV.corr(al.SW); lgv = al.IV.corr(al.GV)
    # variazioni
    d = al.diff().dropna(); n = len(d)
    dsw = d.IV.corr(d.SW); dgv = d.IV.corr(d.GV); dbb = d.SW.corr(d.GV)
    tw = williams_t(dsw, dgv, dbb, n)
    bootd = paired_block_boot(d.SW.values, d.GV.values, d.IV.values, BLOCK)
    obs = dsw - dgv
    # p bilaterale: quota di ricampionamenti la cui differenza CENTRATA supera |obs|
    pb = float(np.mean(np.abs(bootd - bootd.mean()) >= abs(obs))) if len(bootd) else np.nan
    SUMMARY[ccy] = dict(n=n, lsw=lsw, lgv=lgv, dsw=dsw, dgv=dgv, dbb=dbb,
                        tw=tw, pb=pb, se=np.std(bootd) if len(bootd) else np.nan)
    P(f"{ccy:7}{n:5d} | {lsw:+9.2f}{lgv:+9.2f}{lsw-lgv:+8.2f} | "
      f"{dsw:+9.2f}{dgv:+9.2f}{obs:+8.2f}{tw:10.2f}{pb:8.3f}")
P("")
P("    corr(BE_swap, BE_govt) in variazioni -- la dipendenza che il test di Williams usa:")
for ccy, s in SUMMARY.items():
    P(f"      {ccy}: {s['dbb']:+.2f}   SE bootstrap della differenza: {s['se']:.3f}")
P("")

# ---------------------------------------------------------------- [3] tutto il cubo
P("[3] LO STESSO CONTRASTO SU TUTTE LE 16 CELLE (4 expiry x 4 tail), in VARIAZIONI")
P("    Se il contrasto sterlina e' un fatto, non deve dipendere dalla cella scelta.")
IV = load_vols(); EXPS = ["3M","6M","1Y","2Y"]; TENS = ["2Y","5Y","10Y","30Y"]
P(f"{'valuta':7}{'celle':>7}{'diff media':>12}{'diff mediana':>14}{'quota diff<0':>14}{'min':>8}{'max':>8}")
CUBE = {}
for ccy, msw, mgv in PAIRS:
    if msw not in sbe.columns or mgv not in sbe.columns: continue
    diffs = []
    for e in EXPS:
        for t in TENS:
            s = IV.get((ccy, e, t, "NORM"))
            if s is None: continue
            al = pd.concat([sbe[msw], sbe[mgv], s.resample("ME").last()], axis=1).dropna()
            al.columns = ["SW","GV","IV"]
            if len(al) < 60: continue
            d = al.diff().dropna()
            diffs.append(d.IV.corr(d.SW) - d.IV.corr(d.GV))
    if not diffs: continue
    a = np.array(diffs); CUBE[ccy] = a
    P(f"{ccy:7}{len(a):7d}{a.mean():+12.2f}{np.median(a):+14.2f}"
      f"{np.mean(a<0):14.2f}{a.min():+8.2f}{a.max():+8.2f}")
P("    lettura: per la sterlina la storia LDI prevede diff < 0 (co-movimento negativo PIU'")
P("    forte negli swap, dove le casse pensione coprono la duration). Per il Giappone prevede")
P("    diff < 0 con segno opposto sui livelli (il JGB e' la curva PIU' integrata: YCC).")
P("    Il segno deve essere sistematico sul cubo, non su una cella.")
P("")

P("=== VERDETTO ===")
for ccy, s in SUMMARY.items():
    liv = s["lsw"] - s["lgv"]; dlt = s["dsw"] - s["dgv"]
    same = "SI" if np.sign(liv) == np.sign(dlt) else "NO"
    sig = "significativa" if (np.isfinite(s["pb"]) and s["pb"] < 0.05) else "NON significativa"
    P(f"  {ccy}: diff livelli {liv:+.2f} | diff variazioni {dlt:+.2f} ({sig}, p={s['pb']:.3f}) "
      f"| stesso segno: {same}")
P("")
P("  Se la differenza in VARIAZIONI e' significativa: la 6.6 e' la migliore identificazione del")
P("  progetto -- si guida con essa e si cita il livello solo come corollario.")
P("  Se NON lo e': la 6.6 va riscritta come enunciato sui LIVELLI, col caveat del VRP dichiarato")
P("  accanto (la 6.3 lo impone), e il peso della claim si sposta sulla gara di previsione (32).")
P("  In nessuno dei due casi la tabella corrente in livelli puo' restare presentata come")
P("  'the cleanest identification' senza qualificazione.")
save_txt("31_c3_delta.txt", L); print("\n".join(L))
