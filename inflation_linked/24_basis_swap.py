"""24 - LA BASE CALCOLATA DA NOI, CONTRO LA CURVA SWAP. Offline: nessuna chiamata Bloomberg.

COSA FA. Per ogni data e ogni linker:
    z(linker)   = shift parallelo della curva swap zero che riprezza i flussi nominali
                  SINTETICI del linker (cedole proiettate con gli ILS, index ratio e
                  floor gia' gestiti da basis.LinkerBond.cashflows) al suo dirty osservato;
    z(nominale) = lo stesso, sui flussi VERI del nominale, al suo dirty osservato;
    base        = z(linker) - z(nominale), bond per bond (PAIR) e contro l'interpolazione
                  dei due che lo bracciano (INTERP).

PERCHE' NON BASTAVA LA CURVA SOVRANA FITTATA. La usiamo gia' per basis_zspread, e funziona,
ma e' costruita DAI NOMINALI: lo Z-spread del nominale contro di essa e' vicino a zero per
costruzione e assorbe l'errore di fit, che la differenza fra le gambe poi eredita. La curva
swap e' esogena a entrambe: i due Z-spread diventano livelli con un significato proprio.

E LA CURVA SI CANCELLA NELLA DIFFERENZA. Sembra un dettaglio ed e' la cosa che rende tutto
questo robusto: un errore di bootstrap sposta z(linker) e z(nominale) nella stessa
direzione e sparisce nella base. Non serve la curva "giusta" -- serve LA STESSA curva
applicata in modo identico alle due gambe. E' per questo che la scelta di EUSA invece
dell'OIS si dichiara e non si difende.

IL BOOTSTRAP E' QUELLO DELLA I45, NON UNO QUALSIASI. La curva EUR vs 6M Euribor e'
MONO-INDICE: un solo deposito (il 6M) fissa DF(0.5), la striscia di FRA seriali porta i
forward fino a 18 mesi, gli swap par annuali prendono da li' in poi. Il passaggio che
conta e' che il FRA 6x12 consegna DF(1) ESATTO: lo swap a 2 anni ne ha bisogno, e senza
di esso l'unico modo di avere DF(1) e' inventarsi un par swap a 1 anno interpolando --
che e' precisamente cio' che facevo prima, ed e' sbagliato nel tratto che sconta i titoli
corti. Gli altri sei FRA partono da date che nessuno strumento fissa: danno FORMA alla
curva fra i nodi e si impongono per punto fisso, non esattamente.

I tenor swap mancanti si interpolano sui PAR prima di bootstrappare, mai sugli zero dopo:
interpolare dopo introduce forward a zigzag fra i pilastri.

I CONTROLLI CHE NON SI SALTANO, E SONO DUE. Il primo e' il giro par -> zero -> par sugli
swap. Ma quel giro NON TOCCA il tratto sotto i due anni, che e' esattamente quello appena
rifatto: sarebbe l'unico pezzo non verificato. Il secondo riprezza dalla curva finale il
deposito e ogni FRA. Questi tornano a zero PER COSTRUZIONE -- ogni strumento definisce il
proprio nodo -- e va detto chiaro, perche' un controllo che non puo' fallire non e' un
controllo: quel che il secondo vede davvero e' se la ricorsione swap abbia sovrascritto
DF(1) buttando via la striscia FRA, e se il punto fisso converga. Il tratto sotto i sei
mesi non lo vede nessuno dei due: li' la I45 non quota nulla e la curva e' pura
interpolazione. Uguale per le due gambe, quindi si cancella nella base.
"""
import time
import numpy as np
import pandas as pd
import bbg
import holidays as _hol
from basis import (settlement, LinkerBond, zspread, match_nominals,
                   MAX_MISMATCH_DAYS)
from inflation import InflationEngine
from config import CACHE, MARKETS, CAL_ANNI

# ----------------------------------------------------------------- impostazioni
MERCATO   = "DE"
PASSO     = 1          # 1 = ogni giorno, 5 = settimanale (prima passata veloce)
MAX_BRACKET = 1095     # giorni: oltre, l'interpolazione e' un'estrapolazione mascherata
TOL_PAR   = 1.0        # bp: tolleranza del giro par -> zero -> par
SOLO_I45  = True       # scarta le date dove la costruzione I45 non e' completa
SALVA     = True
GRID      = np.round(np.arange(0.25, 50.26, 0.25), 2)

# ----------------------------------------------------------------- bootstrap I45
TAU360 = 365.0 / 360.0      # da anni di calendario a frazione ACT/360 (depositi e FRA)
try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
except ImportError:
    _PCHIP = None


def _df_a(nodi: dict, t: float) -> float:
    """DF a t per interpolazione LOG-LINEARE fra i nodi noti, cioe' forward PIATTI fra i
    pilastri. E' la regola giusta per un tratto breve costruito da FRA: qualunque altra
    inventa forward che nessuno strumento ha quotato. DF(0) = 1 e' sempre un nodo."""
    if t <= 0:
        return 1.0
    ts = np.array(sorted(nodi), dtype=float)
    zt = -np.log(np.array([nodi[x] for x in ts], dtype=float))     # = z*t
    return float(np.exp(-np.interp(t, ts, zt)))


def _breve(r: pd.Series, meta: pd.DataFrame, passate: int = 8) -> tuple:
    """Deposito 6M + striscia FRA -> nodi DF fino a 18 mesi. Ritorna (nodi, n_passate).

    I FRA DI CATENA (6x12 da 0.5, 12x18 da 1.0) atterrano su nodi gia' noti e portano la
    curva avanti in modo esatto. Gli altri sei partono da date che nessuno strumento
    fissa, quindi il loro DF(t0) si interpola sulla curva costruita finora: per questo si
    itera, perche' alla passata dopo l'interpolazione vede anche i nodi appena creati. Il
    sistema e' quasi triangolare e converge in due o tre giri."""
    nodi = {0.0: 1.0}
    for tk, mr in meta[meta["tipo"] == "depo"].iterrows():
        v = r.get(tk)
        if pd.notna(v):
            nodi[float(mr["t1"])] = 1.0 / (1.0 + float(v) / 100.0 * float(mr["t1"]) * TAU360)
    if len(nodi) < 2:
        return {0.0: 1.0}, 0
    fr = meta[meta["tipo"] == "fra"].sort_values("t1")
    fr = fr[[pd.notna(r.get(t)) for t in fr.index]]
    n_p = 0
    for giro in range(passate):
        prima, n_p = dict(nodi), giro + 1
        for tk, mr in fr.iterrows():
            a, b = float(mr["t0"]), float(mr["t1"])
            nodi[b] = _df_a(nodi, a) / (1.0 + float(r[tk]) / 100.0 * (b - a) * TAU360)
        com = [k for k in nodi if k in prima]
        if giro >= 1 and max(abs(nodi[k] - prima[k]) for k in com) < 1e-12:
            break
    return nodi, n_p


def bootstrap(r: pd.Series, meta: pd.DataFrame, modo: str = "i45") -> tuple:
    """Riga di tassi (%) -> (zero continui sulla GRID, nodi DF, regime). modo='solo_swap'
    ignora deposito e FRA e ricostruisce DF(1) interpolando il par a 1 anno: e' la
    costruzione POVERA, serve per le date in cui la striscia FRA non esiste e per
    misurare quanto le due differiscono.

    Il REGIME va deciso QUI e restituito. Guardarlo dopo, chiedendo se 1.0 sta fra i
    nodi, da' sempre 'I45': la ricorsione swap scrive DF(1) per conto suo quando la
    catena non gliel'ha dato. Sarebbe il mescolamento silenzioso che questo campo esiste
    proprio per impedire."""
    nodi, _ = _breve(r, meta) if modo == "i45" else ({0.0: 1.0}, 0)
    regime = "I45" if 1.0 in nodi else "swap"    # DF(1) viene dalla catena FRA?
    p = pd.Series({float(mr["t1"]): float(r[tk])
                   for tk, mr in meta[meta["tipo"] == "swap"].iterrows()
                   if pd.notna(r.get(tk))}).sort_index()
    if len(p) < 3:
        return pd.Series(dtype=float), nodi, regime

    n_max = int(np.floor(p.index.max()))
    if 1.0 in nodi:
        # la striscia FRA ha gia' consegnato DF(1): la ricorsione parte dal 2 anni, che
        # e' il primo swap davvero quotato. Nessun par a 1 anno inventato.
        anni, acc = np.arange(2, n_max + 1, dtype=float), nodi[1.0]
    else:
        anni, acc = np.arange(1, n_max + 1, dtype=float), 0.0
    if not len(anni):
        return pd.Series(dtype=float), nodi, regime
    # PCHIP e non lineare: gli swap sono quotati a 12-15-20-25-30, ma la ricorsione ha
    # bisogno di OGNI anno. Interpolare il par in modo lineare fra quei pilastri sbaglia
    # DF(13) e DF(14), e l'errore si propaga a tutti i DF successivi. Su una curva ripida
    # costa 2.2 bp sullo zero; con PCHIP 0.27. PCHIP e non spline cubica perche' preserva
    # la monotonia e non puo' oscillare fra i pilastri radi del lungo.
    # Fuori dai pilastri quotati si CLAMPA, non si estrapola: senza il taglio, PCHIP che
    # prolunga il par sotto il primo pilastro (2 anni) si inventa fino a 40 bp su una
    # curva ripida, e quel numero entrerebbe in DF(1) e da li' in tutti i DF successivi.
    # Succede solo nel regime 'swap', dove sotto i 2 anni non c'e' nessuno strumento:
    # tenere il par piatto e' un'assunzione, estrapolare e' un'invenzione.
    a_ = np.clip(anni, p.index.values.min(), p.index.values.max())
    s = (_PCHIP(p.index.values, p.values)(a_) if _PCHIP is not None and len(p) >= 3
         else np.interp(a_, p.index.values, p.values)) / 100.0
    s = np.asarray(s, dtype=float)
    for k, n in enumerate(anni):                 # s_n*sum DF(i) + DF(n) = 1
        nodi[float(n)] = (1.0 - s[k] * acc) / (1.0 + s[k])
        acc += nodi[float(n)]

    t = np.array(sorted(k for k in nodi if k > 0), dtype=float)
    d = np.array([nodi[x] for x in t], dtype=float)
    # Il controllo e' sulla POSITIVITA', non sulla monotonia: con tassi negativi i
    # fattori di sconto sono > 1 e crescenti al breve (DF(1a) = 1.005 con il -0.5% del
    # 2020). E' corretto, e imporre DF decrescenti romperebbe meta' campione.
    if np.any(d <= 0):
        return pd.Series(dtype=float), nodi, regime
    z = -np.log(d) / t * 100.0
    return pd.Series(np.interp(GRID, t, z), index=GRID), nodi, regime


def riprezza_breve(r: pd.Series, meta: pd.DataFrame, nodi: dict) -> list:
    """Dalla curva FINALE (dopo la ricorsione swap) ricava il tasso implicito di deposito
    e FRA e lo confronta con quello quotato. [(ticker, tipo, quotato, implicito, bp)].

    COSA CONTROLLA DAVVERO, perche' vale la pena essere precisi. Ogni FRA definisce il
    proprio nodo, quindi dopo la convergenza si riprezza esattamente PER COSTRUZIONE: il
    numero non misura la qualita' dell'interpolazione. Misura due cose reali:
      - che la ricorsione swap non abbia SOVRASCRITTO un nodo del tratto breve. Se
        partisse dal primo anno invece che dal secondo, DF(1) verrebbe dagli swap e la
        striscia FRA sarebbe buttata via in silenzio: qui il 6x12 smetterebbe di tornare.
      - che il punto fisso sia arrivato a convergenza.
    Quel che NESSUN controllo qui vede e' il tratto sotto i 6 mesi: li' non c'e' nessuno
    strumento quotato nella I45, e la curva e' quel che dice l'interpolazione a forward
    piatti dal deposito. I sei FRA che partono li' dentro ereditano quella scelta. Su una
    curva ripida al brevissimo vale qualche bp sullo zero intorno ai 7-9 mesi -- uguale
    per le due gambe, quindi si cancella nella base, ma va detto e non nascosto."""
    out = []
    for tk, mr in meta.iterrows():
        if mr["tipo"] == "swap" or pd.isna(r.get(tk)):
            continue
        a, b = float(mr["t0"]), float(mr["t1"])
        da, db = _df_a(nodi, a), _df_a(nodi, b)
        imp = (da / db - 1.0) / ((b - a) * TAU360) * 100.0
        out.append((tk, mr["tipo"], float(r[tk]), imp, (imp - float(r[tk])) * 100.0))
    return out


def catena(meta: pd.DataFrame) -> list:
    """I FRA che agganciano un nodo gia' esistente, in ordine: sono quelli che portano la
    curva avanti ESATTAMENTE. Si ricavano dalla mappa, non si scrivono a mano."""
    dep = [float(v) for v in meta.loc[meta["tipo"] == "depo", "t1"]]
    if not dep:
        return []
    raggiunti, cat = {round(max(dep), 6)}, []
    fr = meta[meta["tipo"] == "fra"].sort_values("t0")
    for _ in range(len(fr)):
        for tk, mr in fr.iterrows():
            if tk in cat:
                continue
            if round(float(mr["t0"]), 6) in raggiunti:
                cat.append(tk); raggiunti.add(round(float(mr["t1"]), 6))
    return cat


def par_da_zero(z: pd.Series, tenor: np.ndarray) -> np.ndarray:
    """Giro inverso: dagli zero continui si ricalcola il par annuale. Serve al controllo."""
    out = []
    for T in tenor:
        n = int(round(T))
        if n < 1:
            out.append(np.nan); continue
        anni = np.arange(1, n + 1, dtype=float)
        zz = np.interp(anni, z.index.values.astype(float), z.values) / 100.0
        dfs = np.exp(-zz * anni)
        out.append(100.0 * (1.0 - dfs[-1]) / dfs.sum())
    return np.array(out)


# ----------------------------------------------------------------- dati
p_sw, p_meta = CACHE / "swap_EUR.parquet", CACHE / "swap_EUR_meta.parquet"
if not p_meta.exists():
    raise SystemExit(f"manca {p_meta.name}: rilancia il 23. Il pannello vecchio aveva per\n"
                     f"colonne dei TENOR e nessun FRA: con quello il tratto 0-2 anni e'\n"
                     f"costruito con gli strumenti sbagliati, e non c'e' modo di accorgersene\n"
                     f"guardando i risultati.")
sw = pd.read_parquet(p_sw); sw.index = pd.to_datetime(sw.index); sw = sw.sort_index()
meta = pd.read_parquet(p_meta)
meta = meta.loc[[t for t in meta.index if t in sw.columns]]
# ORDINATE PER TENOR. Il pannello arriva con le colonne in ordine alfabetico di ticker
# (EUSA10, EUSA11, EUSA12, EUSA15, EUSA2, ...), che come sequenza di tenor e' 10, 11, 12,
# 15, 2, ... Il bootstrap non se ne accorge perche' fa sort_index(), ma qualunque np.interp
# che prenda quell'ordine come ascisse restituisce numeri senza senso, in silenzio.
col_sw = list(meta.loc[meta["tipo"] == "swap", "t1"].astype(float).sort_values().index)
t2tk = {float(meta.at[tk, "t1"]): tk for tk in col_sw}
CAT = catena(meta)

m = MARKETS[MERCATO]
nom_mkt = bbg.NOMINAL_POOL_ALIAS.get(MERCATO, MERCATO)
ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]
if "base_cpi_final" in ref_l.columns:
    bad = ref_l["base_cpi_final"].isna() | (ref_l["base_cpi_final"] <= 0)
    if bad.any():
        print(f"  esclusi {int(bad.sum())} linker senza BASE_CPI valida")
        ref_l = ref_l[~bad]
ref_n = bbg.load("ref_nominal"); ref_n = ref_n[ref_n["mkt"] == nom_mkt]
px    = bbg.load(f"px_mid_{MERCATO}");  px.index = pd.to_datetime(px.index)
pxn   = bbg.load(f"px_nom_{nom_mkt}");  pxn.index = pd.to_datetime(pxn.index)
ytm   = bbg.load(f"ytm_{nom_mkt}");     ytm.index = pd.to_datetime(ytm.index)
ils   = bbg.load(f"ils_{m.ils}")
cpi   = bbg.load(f"cpi_{m.cpi}").iloc[:, 0]

# --- COSA C'E' DAVVERO IN CACHE, prima di un'ora di calcolo --------------------------
# Ogni ingresso puo' avere una finestra diversa, e il campione finale e' l'intersezione.
# Scoprire dopo che la gamba nominale si buca nel 2007 perche' px_nom parte nel 2009
# costerebbe il run intero: qui costa dieci secondi e si vede quale ingresso MORDE.
print(f"=== {MERCATO}: base contro curva swap ===")
# Un run da quattordici minuti non deve poter essere quello sbagliato senza accorgersene:
# questa riga dice subito quali file uscirano. Se manca, la copia e' vecchia.
print(f"    versione: bootstrap I45 + salva swapz_nom_{MERCATO}.parquet (per il 33)")
print("\n--- ingressi in cache ---")
ils_ok = ils.notna().sum(axis=1) >= 4
ils_d = ils.index[ils_ok]
ing = {
    f"px_mid_{MERCATO} (prezzi linker)":  px.index[px.notna().any(axis=1)],
    f"px_nom_{nom_mkt} (prezzi nominali)": pxn.index[pxn.notna().any(axis=1)],
    f"ils_{m.ils} (>=4 tenor)":            ils_d,
    "swap_EUR (>=4 swap)":                 sw.index[sw[col_sw].notna().sum(axis=1) >= 4],
}
print(f"    {'ingresso':<34}{'da':<12}{'a':<12}{'date':>8}{'colonne':>9}")
for nome, idx in ing.items():
    nc = {f"px_mid_{MERCATO} (prezzi linker)": px.shape[1],
          f"px_nom_{nom_mkt} (prezzi nominali)": pxn.shape[1],
          f"ils_{m.ils} (>=4 tenor)": ils.shape[1],
          "swap_EUR (>=4 swap)": len(col_sw)}[nome]
    if not len(idx):
        print(f"    {nome:<34}{'VUOTO':<24}{0:>8}{nc:>9}"); continue
    print(f"    {nome:<34}{idx.min():%Y-%m-%d}  {idx.max():%Y-%m-%d}{len(idx):>8}{nc:>9}")
print(f"    {f'ytm_{nom_mkt} (rif.)':<34}{ytm.index.min():%Y-%m-%d}  "
      f"{ytm.index.max():%Y-%m-%d}{len(ytm):>8}{ytm.shape[1]:>9}")

date = px.index.intersection(pxn.index).intersection(sw.index)
if len(ils_d):
    date = date[date >= ils_d.min()]
if not len(date):
    raise SystemExit("\nIntersezione vuota: uno degli ingressi non copre gli altri.")
# chi morde all'inizio: l'ingresso la cui prima data coincide con l'inizio del campione
lim = {n: i.min() for n, i in ing.items() if len(i)}
chi = max(lim, key=lambda k: lim[k])
print(f"\n    campione = intersezione: {date.min():%Y-%m-%d} -> {date.max():%Y-%m-%d} "
      f"({len(date)} date)")
print(f"    a MORDERE all'inizio e' '{chi}' ({lim[chi]:%Y-%m-%d}).")
if lim[chi] > pd.Timestamp("2004-06-30"):
    print(f"    -> il campione NON parte dal 2004: se serve, e' quell'ingresso da")
    print(f"       allungare, non gli altri.")

# copertura dei nominali: la gamba nominale esiste solo dove c'e' il PREZZO, non lo ytm
com = pxn.columns.intersection(ytm.columns)
solo_ytm = [c for c in ytm.columns if c not in pxn.columns]
print(f"\n    nominali con prezzo: {pxn.shape[1]}   con ytm: {ytm.shape[1]}   "
      f"in comune: {len(com)}")
if solo_ytm:
    print(f"    {len(solo_ytm)} nominali hanno ytm ma NON il prezzo: non possono fare da")
    print(f"    gamba nominale qui. Se sono quelli che il matching sceglie, il campione")
    print(f"    si riduce e va scaricato px_nom per loro.")
n_g = pxn.loc[date].notna().sum(axis=1)
print(f"    nominali quotati per data: mediana {int(n_g.median())}, "
      f"minimo {int(n_g.min())}, date con meno di 2: {int((n_g < 2).sum())}")

date = date[::PASSO]
print(f"\n    {len(ref_l)} linker, {len(ref_n)} nominali, {len(date)} date "
      f"dopo passo {PASSO}")

# --- quale COSTRUZIONE e' disponibile, data per data ------------------------------
# Una curva che cambia costruzione a meta' campione mette nella base uno scalino che non
# e' economia. Se succede si deve sapere QUANDO e QUANTO vale, non scoprirlo dal grafico.
print("\n--- costruzione disponibile per data ---")
c_dep = list(meta.index[meta["tipo"] == "depo"])
c_fra = list(meta.index[meta["tipo"] == "fra"])
ha_dep = sw[c_dep].notna().any(axis=1) if c_dep else pd.Series(False, index=sw.index)
ha_cat = sw[CAT].notna().all(axis=1) if CAT else pd.Series(False, index=sw.index)
n_fra = sw[c_fra].notna().sum(axis=1) if c_fra else pd.Series(0, index=sw.index)
piena = ha_dep & ha_cat
print(f"    catena (i FRA che portano avanti la curva in modo esatto): "
      f"{', '.join(CAT) if CAT else 'NESSUNA'}")
print(f"\n    {'anno':<8}{'date':>7}{'con depo':>10}{'catena ok':>11}{'FRA medi':>10}"
      f"{'costruzione':>16}")
for y, gg in sw.groupby(sw.index.year):
    q = float(piena.loc[gg.index].mean())
    print(f"    {y:<8}{len(gg):>7}{int(ha_dep.loc[gg.index].sum()):>10}"
          f"{int(ha_cat.loc[gg.index].sum()):>11}{n_fra.loc[gg.index].median():>10.0f}"
          f"{('I45' if q > 0.9 else ('mista' if q > 0.05 else 'solo swap')):>16}")
if piena.any():
    print(f"\n    costruzione I45 possibile dal {sw.index[piena][0]:%Y-%m-%d} "
          f"({piena.sum()}/{len(sw)} date, {piena.mean():.0%})")
else:
    print("\n    !! la costruzione I45 non e' MAI possibile: manca il deposito o la catena.")
d_i45 = sw.index[piena][0] if piena.any() else None

# Le date senza I45 sono poche e SPARSE: il deposito 6M ha qualche stampa mancante qua e
# la', non un buco di periodo. Su quelle la curva verrebbe costruita in un altro modo, e
# mescolare due costruzioni su date sparpagliate e' il peggiore dei mondi -- lo scalino
# non si vede nelle medie annuali e sporca le singole osservazioni. Si scartano: costano
# il 2% del campione e in cambio la costruzione resta una sola su tutto.
if SOLO_I45 and piena.any():
    tenute = piena.reindex(date).fillna(False).values
    if tenute.sum() < len(date):
        print(f"\n    SOLO_I45: scartate {len(date) - int(tenute.sum())} date su {len(date)} "
              f"senza costruzione I45 piena")
        date = date[tenute]
    if not len(date):
        raise SystemExit("SOLO_I45 non lascia nessuna data: metti SOLO_I45 = False.")

# --- controllo 1: il giro par -> zero -> par (riguarda SOLO gli swap) --------------
print("\n--- controllo 1: par -> zero -> par (swap) ---")
prova = date[:: max(1, len(date) // 40)]
err, err_br, n_i45 = [], {}, 0
# Si controlla SOLO su tenor davvero quotati, e si legge il par dalla sua colonna: cosi'
# nel confronto non entra nessuna interpolazione, e il controllo non puo' fallire per un
# difetto proprio invece che del bootstrap. E' quello che e' appena successo.
ten_chk = np.array([t for t in (2.0, 5.0, 10.0, 30.0) if t in t2tk])
if not len(ten_chk):
    raise SystemExit("nessuno dei tenor di controllo (2/5/10/30a) e' quotato.")
for d in prova:
    z, nodi, reg = bootstrap(sw.loc[d], meta)
    if not len(z):
        continue
    n_i45 += int(reg == "I45")
    ric = par_da_zero(z, ten_chk)
    vera = np.array([float(sw.at[d, t2tk[t]]) for t in ten_chk])
    if not np.all(np.isfinite(vera)):
        continue
    err.append(np.abs(ric - vera) * 100.0)      # bp
    for tk, tipo, quo, imp, sc in riprezza_breve(sw.loc[d], meta, nodi):
        err_br.setdefault(tk, []).append(abs(sc))
if not err:
    raise SystemExit("bootstrap impossibile su tutte le date di prova.")
E = np.array(err)
print(f"    {len(E)} date ({n_i45} in costruzione I45 piena), "
      f"errore |par ricalcolato - par vero| in bp:")
for i, t in enumerate(ten_chk):
    print(f"      {t:>5.0f}a   mediana {np.nanmedian(E[:, i]):6.3f}   max {np.nanmax(E[:, i]):6.3f}")
if np.nanmax(E) > TOL_PAR:
    raise SystemExit(f"\nIl giro non torna: errore massimo {np.nanmax(E):.2f} bp > {TOL_PAR}. "
                     f"Il bootstrap e' rotto, mi fermo prima di produrre basi.")
print(f"    entro {TOL_PAR} bp: la parte swap regge.")

# --- controllo 2: il TRATTO BREVE, che il controllo 1 non guarda -------------------
print("\n--- controllo 2: riprezzo deposito e FRA dalla curva finale ---")
if not err_br:
    print("    nessuno strumento breve nel campione di prova: il tratto 0-2 anni viene")
    print("    dagli swap interpolati, ed e' il caso che volevamo evitare.")
else:
    print(f"    {'strumento':<20}{'ruolo':<12}{'|scarto| mediano':>18}{'max':>10}")
    rotto = []
    for tk in [t for t in meta.index if t in err_br]:
        v = np.array(err_br[tk])
        ruolo = ("deposito" if meta.at[tk, "tipo"] == "depo"
                 else ("catena" if tk in CAT else "forma"))
        print(f"    {tk:<20}{ruolo:<12}{np.median(v):>15.3f} bp{np.max(v):>8.3f}")
        if np.max(v) > TOL_PAR:
            rotto.append(f"{tk} ({ruolo})")
    if rotto:
        raise SystemExit(f"\n{len(rotto)} strumenti brevi non si riprezzano dalla curva che "
                         f"hanno costruito:\n    {', '.join(rotto)}\n"
                         f"Questi tornano a zero per costruzione: se non tornano, o la\n"
                         f"ricorsione swap ha sovrascritto un nodo del tratto breve, o il\n"
                         f"punto fisso non converge. In entrambi i casi e' un bug: mi fermo.")
    print("\n    Zeri attesi: ogni strumento definisce il proprio nodo, quindi il numero NON")
    print("    misura la bonta' dell'interpolazione. Misura che la ricorsione swap non abbia")
    print("    sovrascritto DF(1) buttando via la striscia FRA, e che il punto fisso converga.")
    print("    Resta fuori il tratto SOTTO i 6 mesi: li' la I45 non quota nulla e la curva e'")
    print("    l'interpolazione a forward piatti dal deposito. Uguale per le due gambe.")

# --- quanto pesa la costruzione: I45 contro solo-swap -----------------------------
# La curva si cancella al primo ordine nella DIFFERENZA fra le due gambe, quindi questo
# numero e' una robustezza, non una correzione. Ma va misurato, non asserito.
print("\n--- I45 contro solo-swap: quanto cambia la curva ---")
ten_cmp = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
dd = [d for d in prova if piena.get(d, False)]
if dd:
    D2 = []
    for d in dd:
        za, _, _ = bootstrap(sw.loc[d], meta, "i45")
        zb, _, _ = bootstrap(sw.loc[d], meta, "solo_swap")
        if len(za) and len(zb):
            D2.append((np.interp(ten_cmp, GRID, za.values)
                       - np.interp(ten_cmp, GRID, zb.values)) * 100.0)
    if D2:
        D2 = np.array(D2)
        print(f"    differenza sullo zero, {len(D2)} date (bp):")
        for i, t in enumerate(ten_cmp):
            print(f"      {t:>5.0f}a   mediana {np.median(D2[:, i]):>8.2f}   "
                  f"|max| {np.max(np.abs(D2[:, i])):>8.2f}")
        print("    Se la differenza e' grande solo sotto i 2 anni, e' proprio il tratto")
        print("    che avevo sbagliato, e tocca i titoli corti. Sulla BASE si cancella in")
        print("    gran parte, perche' le due gambe la scontano allo stesso modo.")
else:
    print("    nessuna data del campione di prova ha la costruzione I45 piena.")
print()

# ----------------------------------------------------------------- calcolo
eng = InflationEngine(m.cpi, cpi, ils)
bonds = {i: LinkerBond.from_ref(i, r) for i, r in ref_l.iterrows()}
settle_cal = m.holidays
# Calendario delle CEDOLE (diverso dal calendario di regolamento, che viene da config).
# La Spagna mancava: con MERCATO = "ES" questa riga moriva di KeyError dopo il preambolo,
# cioe' dopo i controlli e prima del calcolo -- il momento peggiore per accorgersene.
# E un KeyError nudo non dice che cosa aggiungere ne' dove.
_CAL = {"IT": _hol.Italy, "ES": _hol.Spain, "FR": _hol.France, "FR_CPI": _hol.France,
        "DE": _hol.Germany, "UK": _hol.UK, "US": _hol.US}
if MERCATO not in _CAL:
    raise SystemExit(f"nessun calendario cedolare per '{MERCATO}': aggiungilo a _CAL qui "
                     f"sopra.\nDisponibili: {', '.join(sorted(_CAL))}")
# years= e' obbligatorio, non cosmetico: senza, dict(holidays.Italy()) restituisce ZERO
# voci -- l'oggetto si popola solo quando lo si interroga -- e next_business_day sposta le
# cedole per i soli fine settimana. Sull'Italia non e' mai costato nulla perche' nessuna
# cedola BTPei cade su un festivo italiano (sono tutte al 15 di marzo e settembre), ma un
# DBRi al 15 aprile prende il venerdi santo, ad esempio nel 2022.
cpn_cal = dict(_CAL[MERCATO](years=CAL_ANNI))
print(f"    calendario cedolare {MERCATO}: {len(cpn_cal)} festivita'")
n_set = getattr(m, "settle_days", 2)
mats_n = pd.to_datetime(ref_n["MATURITY"], errors="coerce")
cpns_n = pd.to_numeric(ref_n["CPN"], errors="coerce")
freq_n = pd.to_numeric(ref_n.get("CPN_FREQ"), errors="coerce").fillna(2)
fcd_n = ref_n["FIRST_CPN_DT"] if "FIRST_CPN_DT" in ref_n.columns else None


def cf_nominale(isin, p_clean, settle):
    """Flussi di un nominale nel formato che zspread si aspetta: prima riga -(dirty).
    Cedole generate all'indietro dalla scadenza, rateo ACT/ACT ICMA -- le stesse
    convenzioni del fit GSW in pipeline._euro_nss_params, per non avere due verita'."""
    mat = mats_n.get(isin)
    if pd.isna(mat) or pd.isna(cpns_n.get(isin)):
        return None
    if fcd_n is not None and isin in fcd_n.index:
        f = fcd_n.get(isin)
        if pd.notna(f) and settle < pd.Timestamp(f):
            return None                       # primo periodo irregolare: escluso
    freq = int(freq_n.get(isin, 2)) or 2
    step = max(1, round(12 / freq))
    dts = [mat]
    while dts[-1] > settle:
        dts.append(dts[-1] - pd.DateOffset(months=step))
    nxt = sorted(d for d in dts if d > settle)
    if not nxt:
        return None
    last = nxt[0] - pd.DateOffset(months=step)
    ced = float(cpns_n[isin]) / freq
    den = (nxt[0] - last).days
    acc = ced * (settle - last).days / den if den > 0 else 0.0
    amts = [ced] * len(nxt)
    amts[-1] += 100.0
    # L'indice e' di datetime.date, NON di Timestamp. basis.zspread fa cf.index[0] - asof
    # con asof di tipo date, e Timestamp - date non e' definito: con un DatetimeIndex la
    # gamba nominale esplode alla prima riga. LinkerBond.cashflows usa gia' date, quindi
    # questa e' l'unica convenzione che tiene le due gambe sulla stessa base.
    idx = [settle.date()] + [pd.Timestamp(d).date() for d in nxt]
    return pd.DataFrame({"Cashflows": [-(float(p_clean) + acc)] + amts}, index=idx)


# Gli z dei NOMINALI si salvano. Erano gia' calcolati a ogni data e venivano buttati
# dopo aver prodotto la gamba interpolata: senza di essi non si puo' chiedere se lo z
# interpolato alla scadenza di un linker stia sulla curva dei nominali vicini o sopra --
# che e' l'unica domanda capace di distinguere un difetto di interpolazione da un
# difetto del linker. Costano una colonna per titolo e zero calcolo in piu'.
righe, znom_righe, t0, n_err = [], [], time.time(), 0
for k, dt in enumerate(date):
    z, nodi, reg = bootstrap(sw.loc[dt], meta)
    if not len(z):
        continue
    # Il regime viaggia CON la riga. Se in qualche anno la striscia FRA non c'e', la
    # curva cambia costruzione e la base puo' avere uno scalino: con la colonna qui
    # dentro si puo' filtrare o testare a valle, senza doverci tornare a indovinare.
    asof = dt.date()
    settle = pd.Timestamp(settlement(asof, n_set, settle_cal))

    zn = {}
    for isin, p_cl in pxn.loc[dt].dropna().items():
        cf = cf_nominale(isin, p_cl, settle)
        if cf is None:
            continue
        v = zspread(cf, asof, z)
        if np.isfinite(v):
            zn[isin] = v * 100.0                       # bp
    if len(zn) < 2:
        continue
    znom_righe.append(pd.Series(zn, name=dt))
    zs = pd.Series(zn)
    mn = mats_n.reindex(zs.index)
    ordine = mn.sort_values().index
    zs, mn = zs[ordine], mn[ordine]
    mnum = mn.values.astype("datetime64[D]").astype(int)

    for isin, p_cl in px.loc[dt].dropna().items():
        b = bonds.get(isin)
        if b is None:
            continue
        try:
            cf = b.cashflows(asof, eng, float(p_cl), settle_cal, cpn_cal, True, True)
            zl = zspread(cf, asof, z)
        except Exception:
            n_err += 1
            continue
        if not np.isfinite(zl):
            continue
        zl *= 100.0
        ml = np.datetime64(b.maturity, "D").astype(int)

        pos = int(np.searchsorted(mnum, ml))
        a, bb = pos - 1, pos
        pair = interp = np.nan
        mism = brk = w = np.nan
        cand = [(abs(mnum[i] - ml), i) for i in (a, bb)
                if (i == a and a >= 0) or (i == bb and bb < len(mnum))]
        if cand:
            d0 = min(c[0] for c in cand)
            i_g = min((c for c in cand if c[0] == d0), key=lambda c: zs.index[c[1]])[1]
            mism = int(abs(mnum[i_g] - ml))
            if mism <= MAX_MISMATCH_DAYS:
                pair = zl - zs.iloc[i_g]
        if a >= 0 and bb < len(mnum):
            brk = int(mnum[bb] - mnum[a])
            if brk == 0:
                interp, w = zl - 0.5 * (zs.iloc[a] + zs.iloc[bb]), 0.5
            elif brk <= MAX_BRACKET:
                w = (ml - mnum[a]) / brk
                interp = zl - ((1 - w) * zs.iloc[a] + w * zs.iloc[bb])
        if np.isfinite(pair) or np.isfinite(interp):
            righe.append((dt, isin, zl, pair, interp, mism, brk, w,
                          int(ml - np.datetime64(asof, "D").astype(int)), reg))
    if (k + 1) % 250 == 0:
        print(f"  {k+1}/{len(date)} date, {len(righe)} osservazioni "
              f"({time.time()-t0:.0f}s)")

D = pd.DataFrame(righe, columns=["date", "isin", "z_lnk", "pair", "interp",
                                 "mismatch", "bracket", "w", "ttm", "regime"])
if not len(D):
    raise SystemExit("nessuna osservazione prodotta.")
print(f"\nfatto: {len(D)} osservazioni, {D['isin'].nunique()} linker, "
      f"{n_err} errori di flusso, {time.time()-t0:.0f}s")
qr = D["regime"].value_counts()
print(f"    costruzione della curva: " + ", ".join(f"{k} {v} ({v/len(D):.0%})"
                                                   for k, v in qr.items()))
if len(qr) > 1:
    cambio = D.groupby("regime")["date"].agg(["min", "max"])
    print("    !! il campione MESCOLA due costruzioni. Finestre:")
    for r_, rr in cambio.iterrows():
        print(f"       {r_:<6} {rr['min']:%Y-%m-%d} -> {rr['max']:%Y-%m-%d}")
    print("       Confronta le medie annuali della base sui due regimi prima di leggere")
    print("       come economia qualunque salto che cada sul confine.")


def _st(x):
    x = x.dropna()
    if len(x) < 2:
        return f"{len(x):>8}"
    q = np.percentile(x, [5, 50, 95])
    return (f"{len(x):>8}{x.mean():>9.2f}{x.std():>8.2f}{q[0]:>9.2f}"
            f"{q[1]:>8.2f}{q[2]:>9.2f}{x.skew():>8.2f}")


H = f"{'n':>8}{'media':>9}{'sd':>8}{'p5':>9}{'mediana':>8}{'p95':>9}{'skew':>8}"
print(f"\n--- 1. le due misure (bp) ---")
# Si stampano GREZZE e FILTRATE, perche' le grezze non vanno mai citate. Sull'ultimo anno
# di vita di un titolo la duration crolla e un errore di prezzo di un centesimo diventa
# decine di bp: sulla Francia lo skew grezzo di interp e' -66 e il 2026 ha sd 136, mentre
# con vita residua oltre l'anno la stessa misura ha sd 9.6. Due righe che convivono nella
# stessa tabella tolgono la tentazione di leggere quella sbagliata.
V1 = D[D["ttm"] > 365]
print(f"    {'':<12}{H}")
print(f"    {'pair':<12}{_st(D['pair'])}")
print(f"    {'interp':<12}{_st(D['interp'])}")
print(f"    {'':<12}{'--- con vita residua oltre 1 anno (le misure che si usano) ---'}")
print(f"    {'pair >1a':<12}{_st(V1['pair'])}")
print(f"    {'interp >1a':<12}{_st(V1['interp'])}")

print(f"\n--- 2. per anno (interp, vita residua oltre 1 anno) ---")
print(f"    {'':<12}{H}")
for y, g in V1.groupby(V1["date"].dt.year):
    print(f"    {y:<12}{_st(g['interp'])}")

# --- 3. contro Bloomberg, dove Bloomberg e' sano ----------------------------------
pb = CACHE / f"bbgbasis_interp_{MERCATO}.parquet"
print(f"\n--- 3. contro la base Bloomberg ---")
if not pb.exists():
    print(f"    {pb.name} non c'e': lancia 17 per il confronto.")
else:
    B = pd.read_parquet(pb); B.index = pd.to_datetime(B.index)
    bl = B.stack().dropna().rename("bbg"); bl.index.names = ["date", "isin"]
    M = D.set_index(["date", "isin"])[["interp", "ttm"]].join(bl, how="inner")
    print(f"    {len(M)} celle in comune")
    for eti, sub in (("tutto", M), ("dal 2012", M[M.index.get_level_values(0) >= "2012-01-01"])):
        x = sub[["interp", "bbg"]].dropna()
        if len(x) < 50:
            continue
        d = x["interp"] - x["bbg"]
        print(f"    {eti:<10} n {len(x):>6}   corr {x['interp'].corr(x['bbg']):+.3f}   "
              f"nostra {x['interp'].mean():+.2f} vs bbg {x['bbg'].mean():+.2f} bp   "
              f"scarto {d.mean():+.2f} (sd {d.std():.2f})")
    print("    Il 2006-2011 di Bloomberg lo sappiamo rotto: e' la riga 'dal 2012' che conta.")
    print("    Correlazione alta li' = due strade indipendenti verso lo stesso prezzo")
    print("    relativo, ed e' la validazione che cercavamo. La nostra copre anche prima.")

if SALVA:
    D.to_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
    for nome in ("pair", "interp"):
        (D.pivot_table(index="date", columns="isin", values=nome)
           .to_parquet(CACHE / f"swapbasis_{nome}_{MERCATO}.parquet"))
    print(f"\nsalvati: swapbasis_{MERCATO}.parquet (lungo, con mismatch/bracket/w/ttm)")
    print(f"         swapbasis_pair_{MERCATO}.parquet, swapbasis_interp_{MERCATO}.parquet")
    if znom_righe:
        ZN = pd.DataFrame(znom_righe).sort_index()
        ZN.to_parquet(CACHE / f"swapz_nom_{MERCATO}.parquet")
        print(f"         swapz_nom_{MERCATO}.parquet  ({ZN.shape[0]} date x "
              f"{ZN.shape[1]} nominali, z in bp)   <- serve al 33")
    else:
        print("    !! nessuno z nominale raccolto: il 33 non potra' girare.")
