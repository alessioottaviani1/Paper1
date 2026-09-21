"""20 - L'ANOMALIA HA LA TAGLIA DELL'UPLIFT MANCANTE? Offline, nessuna chiamata, nessuna scrittura.

DA DOVE VIENE L'IPOTESI. Il floor e' stato respinto dai dati: il danno sta dove l'index
ratio e' ALTO (1.05-1.20, 21-29% di assurde) e non dove e' basso (0.98-1.02, 0%). Restano
tre fatti, e sono vincoli stretti:
  a) le assurde sono negative al 98%;
  b) peggiorano al crescere dell'index ratio, ma solo fino a ~1.20;
  c) peggiorano al ridursi della vita residua (23% a 1-2 anni, 2% oltre 12).

L'IPOTESI. Bloomberg non rivaluta il capitale per l'index ratio nel calcolo dello Z-spread
di quei titoli in quella finestra. Allora sottostima il rimborso di (IR - 1) in punti di
prezzo, e per riagganciare il prezzo osservato serve uno spread di circa

        errore ~ - (IR - 1) x 10000 / D        bp,   D = duration in anni

negativo sempre (a), proporzionale a IR - 1 (b), e diviso per la duration, quindi enorme
sui titoli corti (c). A IR = 1 l'errore e' ZERO, che e' proprio dove l'ipotesi floor moriva.

PERCHE' QUESTO TEST VALE E LE STORIE PRECEDENTI NO. Non chiede "il segno torna?" ma "il
NUMERO torna?". L'errore previsto e' calcolabile osservazione per osservazione da IR e
dalla vita residua, che abbiamo entrambi. Se l'ipotesi e' giusta, la regressione
dell'anomalia osservata su quella prevista ha pendenza ~1 e R2 alto. Se e' sbagliata, non
c'e' verso che una formula inventata a tavolino azzecchi 5000 numeri.

D ~ ttm e' un'approssimazione: per un linker a cedola bassa la duration sta un po' sotto la
vita residua, quindi il vero errore e' un po' PIU' grande del previsto. Una pendenza
leggermente sopra 1 e' attesa; una pendenza di 0.1 o di 10 no.

LA SECONDA DOMANDA, indipendente dalla prima. Ogni titolo ha il suo periodo rotto, o si
rompono e guariscono tutti insieme? Se ogni titolo entra e esce per conto suo, e' una
proprieta' del titolo (o di IR, che dipende dal titolo). Se le date sono le stesse per
tutti, e' una finestra di metodologia Bloomberg. Le due cose si distinguono guardando le
date di inizio e fine episodio, non ragionandoci sopra.
"""
import numpy as np
import pandas as pd
import bbg
from basis import mef_reference
from config import CACHE, MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO  = "IT"
LAG, INTERP = 3, True
ASSURDA  = 100.0
MIN_VITA = 365
N_MOSTRA = 4          # titoli peggiori di cui stampare la serie giorno per giorno

# ----------------------------------------------------------------- dati
pm = CACHE / f"bbgmatch_{MERCATO}.parquet"
if not pm.exists():
    raise SystemExit(f"manca {pm.name}: lancia prima 17.")
D = pd.read_parquet(pm)
D["date"] = pd.to_datetime(D["date"])
D = D[D["ttm"] > MIN_VITA].copy()

m = MARKETS[MERCATO]
cpi = bbg.load(f"cpi_{m.cpi}").iloc[:, 0]; cpi.index = pd.to_datetime(cpi.index)
ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]
base = pd.to_numeric(ref_l["base_cpi_final"], errors="coerce")

rif = {}
for d in D["date"].drop_duplicates():
    try:
        rif[d] = mef_reference(cpi, d.date(), lag=LAG, interpolate=INTERP)
    except Exception:
        rif[d] = np.nan
D["IR"] = D["date"].map(rif) / D["isin"].map(base)
D["anni"] = D["ttm"] / 365.25
D = D[D["IR"].notna() & (D["IR"] > 0) & (D["anni"] > 0)]
D["male"] = D["interp"].abs() > ASSURDA

# --- il livello "sano" di ciascun titolo: la sua stessa base dove non e' rotta -----
# Non una costante globale: ogni titolo ha il suo livello di base, e sottrarre 25 bp a
# tutti mescolerebbe il segnale vero con l'anomalia da spiegare.
sano = D[~D["male"]].groupby("isin")["interp"].median()
D["sano"] = D["isin"].map(sano)
D = D[D["sano"].notna()]
D["anomalia"] = D["interp"] - D["sano"]
D["atteso"] = -(D["IR"] - 1.0) * 10000.0 / D["anni"]

print(f"=== {MERCATO}: l'anomalia ha la taglia dell'uplift mancante? ===")
print(f"    {len(D)} osservazioni, {int(D['male'].sum())} assurde ({D['male'].mean():.1%})")
print(f"    livello sano per titolo: mediana {sano.median():+.1f} bp "
      f"(da {sano.min():+.1f} a {sano.max():+.1f})\n")

# --- 1. il test quantitativo -------------------------------------------------------
A = D[D["male"]]
print("--- 1. anomalia osservata contro anomalia prevista, sulle sole assurde ---")
if len(A) > 50:
    x, y = A["atteso"].values, A["anomalia"].values
    sl, ic = np.polyfit(x, y, 1)
    rho = np.corrcoef(x, y)[0, 1]
    print(f"    {len(A)} osservazioni")
    print(f"    prevista:  mediana {np.median(x):>9.1f} bp   p5 {np.percentile(x,5):>9.1f}")
    print(f"    osservata: mediana {np.median(y):>9.1f} bp   p5 {np.percentile(y,5):>9.1f}")
    print(f"    regressione osservata ~ prevista:  pendenza {sl:+.3f}   "
          f"intercetta {ic:+.1f} bp   R2 {rho**2:.3f}")
    rap = y / np.where(np.abs(x) > 1e-9, x, np.nan)
    rap = rap[np.isfinite(rap)]
    if len(rap):
        q = np.percentile(rap, [25, 50, 75])
        print(f"    rapporto osservata/prevista: q1 {q[0]:.2f}  mediana {q[1]:.2f}  q3 {q[2]:.2f}")
    print("    Pendenza ~1 e R2 alto = la formula azzecca la TAGLIA, non solo il segno.")
    print("    D~ttm sottostima un po' l'errore, quindi una pendenza appena sopra 1 e' attesa.")
else:
    sl = rho = np.nan
    print("    troppe poche assurde per regredire.")

# --- 2. e sulle osservazioni SANE la formula prevede zero? -------------------------
# Controllo simmetrico, ed e' quello che impedisce di farsi ingannare: una formula che
# spiega le assurde ma predice anomalie enormi anche dove non ce ne sono e' sbagliata.
S = D[~D["male"]]
print(f"\n--- 2. controllo: cosa prevede la formula dove NON c'e' anomalia ---")
if len(S) > 50:
    print(f"    {len(S)} osservazioni sane")
    print(f"    prevista:  mediana {S['atteso'].median():>9.1f} bp   "
          f"p5 {np.percentile(S['atteso'],5):>9.1f}   p95 {np.percentile(S['atteso'],95):>9.1f}")
    print(f"    osservata: mediana {S['anomalia'].median():>9.1f} bp   "
          f"p5 {np.percentile(S['anomalia'],5):>9.1f}   p95 {np.percentile(S['anomalia'],95):>9.1f}")
    grandi = int((S["atteso"] < -ASSURDA).sum())
    print(f"    {grandi} sane ({grandi/len(S):.0%}) dove la formula prevedeva oltre "
          f"{-ASSURDA:.0f} bp di anomalia")
    print("    Se questa quota e' alta, la formula prevede disastri che non accadono: la")
    print("    taglia e' giusta per caso e l'ipotesi non regge.")

# --- 3. episodi: proprieta' del titolo o finestra di metodologia? ------------------
print(f"\n--- 3. quando si rompe e quando guarisce, titolo per titolo ---")
ep = []
for isin, g in D.sort_values("date").groupby("isin"):
    b = g[g["male"]]
    if not len(b):
        continue
    ep.append((isin, len(g), len(b), b["date"].min(), b["date"].max(),
               g.loc[b["date"].idxmin(), "IR"] if len(b) else np.nan,
               g.loc[b["date"].idxmin(), "anni"] if len(b) else np.nan))
E = pd.DataFrame(ep, columns=["isin", "n", "n_male", "da", "a", "IR_inizio", "anni_inizio"])
E = E.sort_values("n_male", ascending=False)
print(f"    {'isin':<14}{'n':>6}{'assurde':>9}  {'primo':<11}{'ultimo':<11}"
      f"{'IR@primo':>10}{'anni@primo':>12}")
for _, r in E.head(12).iterrows():
    print(f"    {r['isin']:<14}{int(r['n']):>6}{int(r['n_male']):>9}  "
          f"{r['da']:%Y-%m-%d} {r['a']:%Y-%m-%d} {r['IR_inizio']:>10.3f}{r['anni_inizio']:>12.1f}")
if len(E) > 2:
    print(f"\n    dispersione delle date di INIZIO: {E['da'].min():%Y-%m} -> {E['da'].max():%Y-%m}")
    print(f"    dispersione delle date di FINE:   {E['a'].min():%Y-%m} -> {E['a'].max():%Y-%m}")
    print(f"    IR alla rottura: mediana {E['IR_inizio'].median():.3f} "
          f"(da {E['IR_inizio'].min():.3f} a {E['IR_inizio'].max():.3f})")
    print("    Date sparse + IR simile alla rottura = proprieta' del TITOLO.")
    print("    Date uguali per tutti = finestra di metodologia BLOOMBERG.")

# --- 4. la serie, giorno per giorno, attorno alla rottura -------------------------
print(f"\n--- 4. i {N_MOSTRA} titoli peggiori, attorno al primo giorno rotto ---")
print("    (salto netto = interruttore; deriva = qualcosa che cresce con IR o col tempo)")
for _, r in E.head(N_MOSTRA).iterrows():
    g = D[D["isin"] == r["isin"]].sort_values("date").reset_index(drop=True)
    k = g.index[g["date"] == r["da"]]
    if not len(k):
        continue
    k = int(k[0])
    lo, hi = max(0, k - 4), min(len(g), k + 5)
    print(f"\n    {r['isin']}   sano {g['sano'].iloc[0]:+.1f} bp")
    print(f"      {'data':<12}{'base':>10}{'anomalia':>11}{'prevista':>11}{'IR':>8}{'anni':>7}")
    for i in range(lo, hi):
        x = g.iloc[i]
        seg = "  <-- primo rotto" if i == k else ""
        print(f"      {x['date']:%Y-%m-%d}{x['interp']:>10.1f}{x['anomalia']:>11.1f}"
              f"{x['atteso']:>11.1f}{x['IR']:>8.3f}{x['anni']:>7.1f}{seg}")

# --- verdetto ---------------------------------------------------------------------
print(f"\n--- verdetto ---")
ok_sl = np.isfinite(sl) and 0.5 < sl < 2.0
ok_r2 = np.isfinite(rho) and rho**2 > 0.5
falsi = (S["atteso"] < -ASSURDA).mean() if len(S) > 50 else np.nan
ok_fp = np.isfinite(falsi) and falsi < 0.10
print(f"    pendenza {sl:+.3f} (attesa ~1)   R2 {rho**2:.3f}   "
      f"falsi allarmi sulle sane {falsi:.0%}" if np.isfinite(sl) else "    non calcolabile")
if ok_sl and ok_r2 and ok_fp:
    print("\n    CONFERMATA. L'anomalia ha la taglia dell'uplift mancante, non solo il segno:")
    print("    in quella finestra lo Z-spread Bloomberg di quei linker e' calcolato senza")
    print("    rivalutare il capitale per l'index ratio. Non e' rumore ed e' correggibile in")
    print("    linea di principio, ma NON va corretta a mano: si dichiara il campo")
    print("    inutilizzabile dove l'errore previsto e' grande, con una soglia scelta")
    print("    sull'errore e non sul risultato.")
elif ok_sl and ok_r2 and not ok_fp:
    print("\n    La taglia torna sulle assurde ma la formula prevede disastri anche dove non")
    print("    ce ne sono. Spiega troppo: c'e' dentro qualcosa che spegne l'errore in certi")
    print("    periodi. Guarda il punto 3 -- se le date sono comuni a tutti i titoli, e' una")
    print("    finestra di metodologia e IR spiega solo QUANTO, non QUANDO.")
else:
    print("\n    NON CONFERMATA: la formula non azzecca la taglia. Il segno e le dipendenze")
    print("    tornavano, il numero no, e il numero e' quel che conta. Smetti di cercare")
    print("    ipotesi a tavolino: il punto 4 stampa le serie vere, e la forma della")
    print("    transizione (salto o deriva) dice piu' di qualunque altro ragionamento.")
