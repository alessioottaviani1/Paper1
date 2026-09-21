"""16 - I DUE CAMPI Z-SPREAD MISURANO LA STESSA COSA? Rumore di snapshot o convenzioni diverse.

Offline: legge i due pannelli gia' su disco, NON chiama Bloomberg, non scrive nulla.

LA DOMANDA. Su uno stesso nominale, nello stesso istante, bdp restituisce lo stesso numero
per SP037 e BX326 (68.9 vs 68.9). Sullo storico pero' la mediana dello scarto e' 0.8 bp ma
il p99 e' 39 bp, con code fino a centinaia di punti. Delle due l'una:

  RUMORE  - i due campi sono la stessa misura presa in momenti o con arrotondamenti un po'
            diversi. Lo scarto non ha struttura, si concentra dove QUALSIASI Z-spread
            esplode (fine vita: lo spread e' ~ prezzo/duration e la duration va a zero), e
            sparisce appena si toglie quella zona.
  DIVERSE - i due campi scontano su curve diverse. Allora lo scarto e' SISTEMATICO: uno
            scalino per titolo o per periodo che sopravvive a qualsiasi filtro. In quel
            caso la base BX326 resta internamente coerente, ma non si puo' dire di averla
            validata contro SP037, e soprattutto non si possono mescolare i due campi.

COME SI DISTINGUONO, in ordine di forza della prova:

 1. AGGREGATO. Media, sd e percentili dello scarto, in tutto e per tipo di titolo. Serve
    solo a inquadrare: una mediana sotto il bp con code enormi non dice ancora nulla.
 2. VITA RESIDUA. Se lo scarto e' amplificazione, |diff| cresce come 1/ttm: in log-log la
    pendenza di |diff| su ttm deve stare attorno a -1. Una pendenza piatta dice che lo
    scarto non c'entra con la duration, e l'ipotesi rumore cade.
 3. SOGLIE. Quanto vale il p95 filtrando a vita residua crescente. Se crolla sotto il bp
    gia' al filtro che la pipeline applica comunque, il problema non tocca il campione.
 4. PERIODO. Uno scarto per anno che compare e sparisce e' un cambio di metodologia
    Bloomberg, non rumore: sarebbe la cosa peggiore, perche' entra nella serie storica.
 5. TITOLO. Per ogni titolo si testa se la media dello scarto e' distinguibile da zero
    (t = media / errore standard). Tanti titoli con |t| grande = scalini sistematici.
    Attenzione: con migliaia di osservazioni per titolo anche uno scarto economicamente
    nullo diventa "significativo", quindi accanto al t si guarda SEMPRE la media in bp.

Il verdetto combina i tre segnali che contano: il p95 dopo il filtro, la pendenza log-log,
e quanti titoli hanno uno scalino grande in valore assoluto, non solo significativo.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO = "IT"
CAMPO_A = "INDEX_Z_SPREAD_BP"     # quello nuovo, usato per la base
CAMPO_B = "Z_SPRD_MID"            # quello vecchio, solo nominali
SOGLIE  = [0, 30, 90, 180, 365, 730]   # filtri di vita residua da provare (giorni)
VITA_PIPE = 365                   # MIN_VITA_GG applicato in 04_basis_markets
SCALINO_BP = 1.0                  # media per titolo oltre cui lo scalino e' economicamente rilevante

# ----------------------------------------------------------------- dati
pa = CACHE / f"zsprd_{MERCATO}_{CAMPO_A}.parquet"
pb = CACHE / f"zsprd_{MERCATO}_{CAMPO_B}.parquet"
for p in (pa, pb):
    if not p.exists():
        raise SystemExit(f"manca {p.name}: lancia prima 15 con quel campo.")

A = pd.read_parquet(pa); A.index = pd.to_datetime(A.index)
B = pd.read_parquet(pb); B.index = pd.to_datetime(B.index)
com = [c for c in A.columns if c in B.columns]
if not com:
    raise SystemExit("nessuna colonna in comune fra i due pannelli.")
idx = A.index.union(B.index)

L = pd.concat([A[com].reindex(idx).stack().rename("za"),
               B[com].reindex(idx).stack().rename("zb")], axis=1).dropna()
L.index.names = ["date", "isin"]
L = L.reset_index()
L["diff"] = L["za"] - L["zb"]

ref_l = bbg.load("ref_linker"); ref_n = bbg.load("ref_nominal")
kind = pd.concat([pd.Series("linker", index=ref_l.index),
                  pd.Series("nominal", index=ref_n.index)])
kind = kind[~kind.index.duplicated()]
mat = pd.concat([pd.to_datetime(ref_l["maturity"], errors="coerce"),
                 pd.to_datetime(ref_n["maturity"], errors="coerce")])
mat = mat[~mat.index.duplicated()]
L["kind"] = L["isin"].map(kind)
L["ttm"] = (L["isin"].map(mat) - L["date"]).dt.days
L = L[L["ttm"].notna() & (L["ttm"] > 0)]
L["ad"] = L["diff"].abs()

print(f"=== {CAMPO_A} vs {CAMPO_B} ({MERCATO}) ===")
print(f"    {len(L)} celle grezze, {L['isin'].nunique()} titoli, "
      f"{L['date'].min():%Y-%m} -> {L['date'].max():%Y-%m}")
print(f"    per tipo: " + ", ".join(f"{k} {int(v)}" for k, v in L["kind"].value_counts().items()))

# I LINKER NON ENTRANO NEL CONFRONTO. Il campo B (SP037) sui linker non esiste: quel che
# ne esce sono poche celle sporadiche, e valgono mediane di centinaia di bp perche' e' uno
# Z-spread calcolato su flussi NON indicizzati. Non sono evidenza sul campo A: sono la
# prova, semmai, che B sul linker e' inservibile -- cioe' esattamente il motivo per cui si
# e' cambiato campo. Tenerle dentro sposterebbe ogni statistica e falserebbe il verdetto.
n_lk = int((L["kind"] == "linker").sum())
if n_lk:
    q_lk = np.percentile(L.loc[L["kind"] == "linker", "ad"].values, 50)
    print(f"    ESCLUSI {n_lk} celle linker (|mediana| {q_lk:.1f} bp): {CAMPO_B} sui linker")
    print(f"    non e' definito, quelle celle sono uno Z-spread su flussi non indicizzati.")
    print(f"    Il confronto e' quindi un test sui SOLI NOMINALI, per costruzione.")
L = L[L["kind"] == "nominal"].copy()
if not len(L):
    raise SystemExit("nessun nominale in comune: niente da confrontare.")


def _st(g: pd.DataFrame) -> str:
    if not len(g):
        return f"{'-':>8}" * 5
    q = np.percentile(g["ad"].values, [50, 95, 99])
    return (f"{g['diff'].mean():>+9.3f}{g['diff'].std():>9.2f}"
            f"{q[0]:>10.3f}{q[1]:>9.2f}{q[2]:>9.2f}")


HDR = f"{'n':>9}{'media':>9}{'sd':>9}{'|med|':>10}{'|p95|':>9}{'|p99|':>9}"

# --- 1. tutto il campione ----------------------------------------------------------
print(f"\n--- 1. campione intero ---")
print(f"    {'':<14}{HDR}")
print(f"    {'tutto':<14}{len(L):>9}{_st(L)}")
for k, g in L.groupby("kind"):
    print(f"    {k:<14}{len(g):>9}{_st(g)}")

# --- 2. vita residua ---------------------------------------------------------------
print(f"\n--- 2. per vita residua: lo scarto e' amplificazione di fine vita? ---")
tagli = [0, 90, 365, 1095, 2555, 5475, 1e9]
et = ["< 3 mesi", "3m - 1 anno", "1 - 3 anni", "3 - 7 anni", "7 - 15 anni", "> 15 anni"]
L["b"] = pd.cut(L["ttm"], tagli, labels=et, right=False)
print(f"    {'':<14}{HDR}")
for b, g in L.groupby("b", observed=True):
    print(f"    {str(b):<14}{len(g):>9}{_st(g)}")

pos = L[(L["ad"] > 1e-9) & (L["ttm"] > 0)]
slope = np.nan
if len(pos) > 100:
    x = np.log(pos["ttm"].values / 365.25)
    y = np.log(pos["ad"].values)
    slope, inter = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    print(f"\n    log|diff| ~ log(vita residua in anni):  pendenza {slope:+.2f}  (corr {r:+.2f})")
    print(f"    -1 = |diff| inversamente proporzionale alla vita residua, cioe' la firma")
    print(f"    dell'amplificazione: lo spread e' ~ scarto di prezzo / duration, e la")
    print(f"    duration va a zero. Vicino a 0 = lo scarto non c'entra con la duration.")

# --- 3. soglie ---------------------------------------------------------------------
print(f"\n--- 3. filtrando a vita residua crescente ---")
print(f"    {'filtro':<14}{HDR}")
for s in SOGLIE:
    g = L[L["ttm"] > s]
    tag = f"> {s}gg" + ("  <-- pipeline" if s == VITA_PIPE else "")
    print(f"    {tag:<14}{len(g):>9}{_st(g)}")

# --- 4. periodo --------------------------------------------------------------------
Lf = L[L["ttm"] > VITA_PIPE]
print(f"\n--- 4. per anno, dopo il filtro > {VITA_PIPE}gg ---")
print(f"    {'':<14}{HDR}")
ann = {}
for y, g in Lf.groupby(Lf["date"].dt.year):
    ann[y] = g["diff"].mean()
    print(f"    {y:<14}{len(g):>9}{_st(g)}")

# Un salto fra due anni contigui si cerca a macchina, non a occhio su venti righe: e' la
# firma di un cambio di metodologia Bloomberg, che sarebbe il caso peggiore perche' entra
# nella serie storica invece di restare nelle code.
ann = pd.Series(ann).sort_index()
salto, anno_salto = 0.0, None
if len(ann) > 1:
    dd = ann.diff().abs().dropna()
    if len(dd):
        salto, anno_salto = float(dd.max()), int(dd.idxmax())
        # La soglia e' la variabilita' delle medie annuali stesse, non un numero fisso: se
        # le medie ballano gia' di 2-3 bp per conto loro, un salto di 3 bp e' dentro il
        # rumore e segnalarlo come cambio di metodologia sarebbe un falso allarme.
        sd_ann = float(ann.std())
        if salto > max(SCALINO_BP, 2.5 * sd_ann):
            print(f"\n    SALTO fra {anno_salto-1} e {anno_salto}: la media passa da "
                  f"{ann[anno_salto-1]:+.2f} a {ann[anno_salto]:+.2f} bp ({salto:.2f} bp,")
            print(f"    contro una sd delle medie annuali di {sd_ann:.2f}). Uno scalino cosi'")
            print(f"    non e' rumore: e' un cambio di metodologia dentro la serie storica.")
        else:
            print(f"\n    salto massimo fra anni contigui: {salto:.2f} bp ({anno_salto}), "
                  f"contro una sd\n    delle medie annuali di {sd_ann:.2f} bp -> dentro il "
                  f"rumore, nessun cambio di metodologia.")
            salto = 0.0

# --- 5. struttura per titolo -------------------------------------------------------
print(f"\n--- 5. scalini per titolo, dopo il filtro > {VITA_PIPE}gg ---")
gb = Lf.groupby("isin")["diff"]
st = pd.DataFrame({"n": gb.size(), "media": gb.mean(), "sd": gb.std()})
st = st[st["n"] >= 30]
st["t"] = st["media"] / (st["sd"] / np.sqrt(st["n"]))
st["kind"] = st.index.map(kind)
n_sig = int((st["t"].abs() > 3).sum())
n_big = int((st["media"].abs() > SCALINO_BP).sum())
print(f"    {len(st)} titoli con almeno 30 osservazioni")
print(f"    {n_sig} con |t| > 3 (media distinguibile da zero)")
print(f"    {n_big} con |media| > {SCALINO_BP} bp (scalino economicamente rilevante)")
print(f"    NB: con migliaia di osservazioni per titolo il t e' significativo anche per")
print(f"    scarti nulli in bp. Conta la seconda riga, non la prima.")
if n_big:
    print(f"\n    i piu' grandi:")
    for i, r in st.reindex(st["media"].abs().sort_values(ascending=False).index).head(8).iterrows():
        print(f"      {i}  {str(r['kind']):8s} media {r['media']:+8.2f} bp  "
              f"sd {r['sd']:7.2f}  n {int(r['n']):5d}  t {r['t']:+8.1f}")

# ----------------------------------------------------------------- verdetto
# Tre domande separate, perche' hanno conseguenze diverse.
#   LIVELLO     - c'e' uno scarto sistematico? Se si', i due campi scontano su curve
#                 diverse, e nessun filtro lo toglie: la validazione salta.
#   DISPERSIONE - quanto e' rumorosa la differenza dove vive il campione.
#   STRUTTURA   - il rumore e' EPISODICO (stress di mercato, fine vita, titoli stantii)
#                 oppure e' sempre li'? Due curve diverse danno uno scalino stabile nel
#                 tempo; l'asincronia di snapshot da' code che si accendono quando il
#                 mercato si muove e si spengono quando sta fermo.
N_RECENTI = 3
p95f = np.percentile(Lf["ad"].values, 95) if len(Lf) else np.nan
p950 = np.percentile(L["ad"].values, 95) if len(L) else np.nan

anni_ord = sorted(ann.index)
rec = anni_ord[-N_RECENTI:]
Lr = Lf[Lf["date"].dt.year.isin(rec)]
Lv = Lf[~Lf["date"].dt.year.isin(rec)]
p95_rec = np.percentile(Lr["ad"].values, 95) if len(Lr) else np.nan
p95_vec = np.percentile(Lv["ad"].values, 95) if len(Lv) else np.nan
trend = np.polyfit(range(len(ann)), ann.values, 1)[0] if len(ann) > 2 else 0.0

print(f"\n--- verdetto ---")
print(f"  LIVELLO      media delle medie annuali {ann.mean():+.2f} bp "
      f"(sd fra anni {ann.std():.2f}, deriva {trend:+.3f} bp/anno)")
print(f"               |mediana| dello scarto dopo il filtro: "
      f"{np.median(Lf['ad'].values):.2f} bp")
print(f"  DISPERSIONE  p95 |diff|: {p950:.2f} bp senza filtro -> {p95f:.2f} dopo > {VITA_PIPE}gg")
print(f"               ultimi {N_RECENTI} anni {p95_rec:.2f} bp   contro {p95_vec:.2f} prima")
print(f"  STRUTTURA    pendenza log-log {slope:+.2f}; {n_big}/{len(st)} titoli con scalino "
      f"> {SCALINO_BP} bp")

liv_ok    = abs(ann.mean()) < SCALINO_BP and abs(trend) * len(ann) < 2 * SCALINO_BP and not salto
episodica = np.isfinite(p95_rec) and np.isfinite(p95_vec) and p95_rec < 0.5 * p95_vec

if not liv_ok:
    print("\n  SCARTO DI LIVELLO. Le medie annuali non stanno attorno a zero, o c'e' una")
    print("  deriva, o un salto netto fra due anni. Questa e' la firma di due curve di")
    print("  sconto diverse, e non la toglie nessun filtro di vita residua. Il campo nuovo")
    print("  resta usabile DA SOLO -- e' coerente con se stesso su entrambe le gambe -- ma")
    print("  non si puo' dire di averlo validato contro il vecchio, e i due non vanno mai")
    print("  mescolati nella stessa misura.")
elif episodica:
    print("\n  STESSO LIVELLO, RUMORE EPISODICO. Le medie annuali stanno attorno a zero")
    print("  senza deriva: sul livello i due campi concordano, quindi scontano sulla stessa")
    print(f"  curva. Le code pero' si accendono negli anni di stress e si spengono dopo")
    print(f"  ({p95_vec:.1f} -> {p95_rec:.1f} bp di p95): e' asincronia di snapshot o pricing")
    print("  stantio, non una convenzione diversa -- due curve diverse darebbero uno")
    print("  scalino stabile, non code che vanno e vengono col mercato.")
    print("\n  ATTENZIONE pero' al metro: una dispersione di qualche bp e' dello stesso")
    print("  ordine della base che vogliamo misurare. La domanda vera non e' quanto rumore")
    print("  ha il singolo Z-spread, ma quanto ne resta nella DIFFERENZA fra le due gambe:")
    print("  se l'asincronia e' comune (stesso istante per linker e nominale) si cancella,")
    print("  se e' idiosincratica no. Si misura sulla base, non qui.")
else:
    print("\n  STESSO LIVELLO, RUMORE PERSISTENTE. Sul livello concordano, ma la")
    print("  dispersione non si spegne col tempo ne' col filtro. Non e' una curva diversa;")
    print("  e' rumore di misura che resta. Va quantificato sulla base prima di usarla:")
    print("  se la differenza fra le gambe eredita questo rumore, il segnale ci affoga.")

if n_big:
    print(f"\n  I {n_big} titoli con scalino vanno guardati per eta' e liquidita': se sono")
    print("  i soliti vecchi, corti e con poche osservazioni, e' pricing stantio e si")
    print("  escludono documentando il criterio. Se sono sparsi e liquidi, no.")
