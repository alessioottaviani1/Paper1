"""26 - LA CODA NEGATIVA STA SUL LUNGO, NON SUL CORTO. Offline: solo file gia' prodotti.

IL FATTO CHE HA FATTO NASCERE QUESTO SCRIPT. Il 25 chiedeva se le basi negative fossero
rumore da duration piccola. La risposta e' NO, ed e' il contrario di quel che mi
aspettavo: sulla vita residua 3-10 anni non ce n'e' NESSUNA, l'85% sta oltre i 10 anni, e
il 69% cade nel solo 2010. La tabella per titolo del 25 dice anche chi: IT0003745541, il
2035, con mediane -14 / -33 / -5 nel 2009-2010-2011 mentre ogni altro BTPei sta fra +5 e
+31. Un titolo, un periodo, e il piu' lungo del campione.

Un'ipotesi che spiega solo l'aggregato non serve a niente: -33 bp di mediana annuale su un
titolo, con gli altri otto positivi, e' una differenza di 40-60 bp sullo STESSO emittente e
non e' plausibile come economia. Quindi o e' la nostra macchina su quel titolo, o e' un
prezzo. Questo script separa i due casi senza aggiungere dati.

COME. La base e' z(linker) - z(nominale), e il pannello del 24 ha z_lnk e la base: quindi
la gamba nominale si RICAVA, z_nom = z_lnk - base. Si guardano separate. Se z_lnk e'
normale e la gamba nominale e' alta, il problema e' nei gemelli o nell'interpolazione fra
i due; se e' z_lnk a essere negativo, il problema e' il linker -- e su un linker z_lnk
dipende dai flussi SINTETICI, quindi dall'inflazione proiettata, quindi dal tratto lungo
della curva ILS. Nel 2009-2010 l'ILS a 25-30 anni e' il pezzo piu' sottile che abbiamo.

TRE IPOTESI, E OGNUNA HA IL SUO CONTROLLO
  A. proiezione: l'ILS lungo e' rado o fermo -> i flussi proiettati sono sbagliati e
     z_lnk con loro. Si guarda la copertura dei tenor lunghi in quegli anni.
  B. bracket largo: a 25 anni i due nominali che bracciano possono distare anni, e
     interpolare z linearmente su quella distanza, dove la curva e' curva, introduce bias.
     Si guarda la distribuzione di 'bracket' sulle osservazioni negative contro le sane.
  C. prezzo: il titolo era illiquido e il prezzo fermo. Un prezzo fermo mentre il mercato
     si muove produce escursioni, non un livello NEGATIVO PERSISTENTE, quindi questa
     ipotesi si falsifica guardando se la serie e' piatta a tratti.

Nessuna delle tre viene assunta: si stampano i tre controlli e si guarda quale regge.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE, MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO  = "IT"
MISURA   = "interp"
MIN_VITA = 365
LUNGO    = 10.0      # anni di vita residua oltre cui siamo "sul lungo"
NEG      = -25.0

# ----------------------------------------------------------------- dati
D = pd.read_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
D["anno"] = D["date"].dt.year
D["anni"] = D["ttm"] / 365.25
D = D[D[MISURA].notna() & (D["ttm"] > MIN_VITA)].copy()
# la gamba nominale non e' nel pannello ma si ricava: base = z_lnk - z_nom
D["z_nom"] = D["z_lnk"] - D[MISURA]

ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]
mat = pd.to_datetime(ref_l["MATURITY"], errors="coerce")
m = MARKETS[MERCATO]
px = bbg.load(f"px_mid_{MERCATO}"); px.index = pd.to_datetime(px.index)
ils = bbg.load(f"ils_{m.ils}"); ils.index = pd.to_datetime(ils.index)

print(f"=== {MERCATO}: da dove viene la coda negativa ===")
neg = D[D[MISURA] < NEG]
print(f"    {len(neg)} osservazioni sotto {NEG:.0f} bp su {len(D)}")

# --- 0. chi le produce -------------------------------------------------------------
print(f"\n--- 0. quali titoli, e quanto pesano ---")
q = neg["isin"].value_counts()
print(f"    {'titolo':<16}{'scad':<12}{'n negative':>12}{'quota':>8}"
      f"{'   sue oss. tot':>16}{'   % sue negative':>18}")
for isin, n in q.head(8).items():
    tot = int((D["isin"] == isin).sum())
    sc = mat.get(isin)
    print(f"    {isin:<16}{sc:%Y-%m-%d}  {n:>12}{n/len(neg):>8.0%}{tot:>16}"
          f"{n/tot:>17.1%}" if pd.notna(sc) else
          f"    {isin:<16}{'?':<12}{n:>12}{n/len(neg):>8.0%}{tot:>16}{n/tot:>17.1%}")
capro = q.index[0]
print(f"\n    il principale e' {capro} (scad {mat.get(capro):%Y-%m-%d})")

# --- 1. LE DUE GAMBE, separate: e' il linker o il nominale? -----------------------
print(f"\n--- 1. le due gambe sul titolo {capro}, anno per anno ---")
print("    z(linker) e z(nominale) sono livelli con un significato proprio: uno")
print("    Z-spread sovrano contro swap sta fra qualche bp e alcune centinaia, mai")
print("    negativo di decine. La colonna che esce dal seminato e' la gamba rotta.\n")
S = D[D["isin"] == capro]
print(f"    {'anno':<7}{'n':>6}{'z(lnk)':>10}{'z(nom)':>10}{'base':>9}{'bracket gg':>12}"
      f"{'ttm anni':>10}")
for y, g in S.groupby("anno"):
    print(f"    {y:<7}{len(g):>6}{g['z_lnk'].median():>10.1f}{g['z_nom'].median():>10.1f}"
          f"{g[MISURA].median():>9.1f}{g['bracket'].median():>12.0f}{g['anni'].median():>10.1f}")

print(f"\n    confronto: le stesse gambe su TUTTI gli altri titoli sopra i {LUNGO:.0f} anni")
A = D[(D["isin"] != capro) & (D["anni"] > LUNGO)]
print(f"    {'anno':<7}{'n':>6}{'z(lnk)':>10}{'z(nom)':>10}{'base':>9}{'titoli':>9}")
for y, g in A.groupby("anno"):
    if y > 2015:
        continue
    print(f"    {y:<7}{len(g):>6}{g['z_lnk'].median():>10.1f}{g['z_nom'].median():>10.1f}"
          f"{g[MISURA].median():>9.1f}{g['isin'].nunique():>9}")
print("\n    Se nei 2009-2010 z(nom) del sospetto e' in linea con gli altri e z(lnk) no,")
print("    il difetto e' sul LINKER: flussi sintetici, quindi inflazione proiettata.")

# --- 2. ipotesi A: il tratto lungo della curva ILS --------------------------------
print(f"\n--- 2. ipotesi A: la curva ILS sul lungo, negli anni incriminati ---")
ten = sorted(float(c) for c in ils.columns)
lunghi = [t for t in ten if t >= 15]
print(f"    tenor ILS disponibili: {', '.join(f'{t:g}' for t in ten)}")
print(f"\n    {'anno':<7}{'date':>7}" + "".join(f"{f'{t:g}a':>10}" for t in lunghi)
      + f"{'  max tenor vivo':>18}")
for y, g in ils.groupby(ils.index.year):
    if y < 2006 or y > 2014:
        continue
    cop = "".join(f"{g[str(t) if str(t) in g.columns else t].notna().mean():>10.0%}"
                  if (str(t) in g.columns or t in g.columns) else f"{'-':>10}"
                  for t in lunghi)
    vivi = [t for t in ten if (str(t) in g.columns or t in g.columns)
            and g[str(t) if str(t) in g.columns else t].notna().any()]
    print(f"    {y:<7}{len(g):>7}{cop}{(max(vivi) if vivi else 0):>18.0f}")
print("\n    Un titolo a 25-30 anni proietta l'inflazione fin la'. Se il tenor piu' lungo")
print("    quotato in quegli anni e' piu' corto della scadenza del titolo, la proiezione")
print("    e' un'ESTRAPOLAZIONE, e su 25 anni di cedole l'errore si accumula sul prezzo")
print("    sintetico -- quindi su z(lnk), quindi sulla base. In quel caso non e' un bug:")
print("    e' un limite del dato, e va dichiarato o il titolo va escluso in quel periodo.")

# --- 3. ipotesi B: la larghezza del bracket ---------------------------------------
print(f"\n--- 3. ipotesi B: quanto sono distanti i due nominali che bracciano ---")
D["neg"] = D[MISURA] < NEG
print(f"    {'gruppo':<28}{'n':>9}{'bracket mediano':>18}{'p95':>9}{'mismatch med':>15}")
for et, s in [("negative (oltre 10a)", D[D["neg"] & (D["anni"] > LUNGO)]),
              ("sane (oltre 10a)", D[~D["neg"] & (D["anni"] > LUNGO)]),
              ("sane (sotto 10a)", D[~D["neg"] & (D["anni"] <= LUNGO)])]:
    if len(s):
        print(f"    {et:<28}{len(s):>9}{s['bracket'].median():>18.0f}"
              f"{s['bracket'].quantile(.95):>9.0f}{s['mismatch'].median():>15.0f}")
rho = float(D.loc[D["anni"] > LUNGO, "bracket"].corr(
            D.loc[D["anni"] > LUNGO, MISURA]))
print(f"\n    corr(bracket, base) oltre i {LUNGO:.0f} anni = {rho:+.3f}")
print("    Se le negative avessero bracket molto piu' larghi delle sane, l'interpolazione")
print("    fra i due gemelli starebbe lavorando su una distanza dove la curva non e'")
print("    lineare. Se i bracket sono uguali, l'ipotesi cade e resta la A.")

# --- 4. ipotesi C: il prezzo era fermo? --------------------------------------------
print(f"\n--- 4. ipotesi C: il prezzo del titolo era stantio? ---")
if capro in px.columns:
    s = px[capro].dropna()
    s = s[(s.index.year >= 2008) & (s.index.year <= 2012)]
    d = s.diff()
    print(f"    {'anno':<7}{'date':>7}{'giorni a prezzo INVARIATO':>28}{'escursione':>13}")
    for y, g in s.groupby(s.index.year):
        dd = g.diff().dropna()
        fermi = int((dd == 0).sum())
        print(f"    {y:<7}{len(g):>7}{f'{fermi} ({fermi/max(1,len(dd)):.0%})':>28}"
              f"{g.max()-g.min():>13.2f}")
    print("\n    Un prezzo fermo per giorni produce SALTI quando si muove, non un livello")
    print("    negativo costante. Se i giorni fermi sono pochi, l'ipotesi C e' esclusa e")
    print("    la spiegazione sta in A o B.")
else:
    print(f"    {capro} non e' fra le colonne di px_mid_{MERCATO}.")

# --- 5. la serie mensile del sospetto ----------------------------------------------
print(f"\n--- 5. {capro}: la base mese per mese, 2009-2012 ---")
T = S[(S["anno"] >= 2009) & (S["anno"] <= 2012)]
if len(T):
    mm = T.groupby(T["date"].dt.to_period("M")).agg(
        n=(MISURA, "size"), base=(MISURA, "median"),
        zl=("z_lnk", "median"), zn=("z_nom", "median"))
    print(f"    {'mese':<10}{'n':>5}{'z(lnk)':>10}{'z(nom)':>10}{'base':>9}")
    for k, r in mm.iterrows():
        print(f"    {str(k):<10}{int(r['n']):>5}{r['zl']:>10.1f}{r['zn']:>10.1f}{r['base']:>9.1f}")
    print("\n    Un REGIME (mesi consecutivi allo stesso livello negativo) punta su A o B:")
    print("    un difetto di costruzione non va e viene. Dei PICCHI isolati punterebbero")
    print("    su prezzi sporchi, e allora si tratterebbero come stampe, non come periodo.")
else:
    print("    nessuna osservazione in finestra.")

print("\n--- come si chiude ---")
print("  A regge  -> l'ILS non copre la scadenza in quegli anni: e' un limite del dato.")
print("     Si esclude il titolo DOVE la proiezione e' estrapolata, con una regola scritta")
print("     sulla copertura della curva -- esogena -- e non sul valore della base.")
print("  B regge  -> si stringe MAX_BRACKET e si rifa il 24: costa un run e basta.")
print("  nessuna  -> non e' ne' la proiezione ne' il matching: allora il prezzo di quel")
print("     titolo in quel periodo va guardato a terminale, ed e' il caso in cui serve")
print("     davvero lo schermo.")
