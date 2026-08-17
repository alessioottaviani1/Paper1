"""
05 - Serie dei rendimenti d'asta BOT 6 mesi: il parametro di indicizzazione dei CCT.

PERCHE' SI PUO' DERIVARE INVECE CHE SCARICARE. Il BOT e' uno zero coupon che rimborsa
alla pari e il cui rendimento e' definito dalla scheda MEF come semplice annuo ACT/360:
      y = (100 - P) / P * 360 / n
Quindi basta il PREZZO del BOT semestrale alla sua data di emissione per ricostruire il
rendimento d'asta, senza una fonte dati separata. La serie ufficiale (dt.tesoro.it) resta
piu' precisa -- il prezzo secondario del giorno di emissione non coincide esattamente con
il prezzo medio ponderato d'asta -- e se disponibile viene usata al posto della derivata.

COPERTURA. Nell'anagrafica ci sono 321 BOT semestrali (durata 160-200 giorni) emessi fra
il 1995 e il 2026, senza anni mancanti, con mediana di 11 aste l'anno e 30 giorni fra
aste consecutive. La regola CCT chiede "l'ultima asta che precede il godimento della
cedola": con una cadenza mensile il ritardo massimo e' contenuto, ma il 2010 ha una sola
asta semestrale nell'anagrafica e li' il parametro va verificato a mano.

Output: PROC/bot_auction_6m.csv + results/03_bot_auction.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

print("== 05 bot auction ==")
L=[]; P=L.append
P("=== 03 RENDIMENTI D'ASTA BOT 6 MESI ===")

O = pd.read_csv(PROC/"static_bot.csv", parse_dates=["maturity","issue"])
O["tenor_d"] = (O.maturity - O.issue).dt.days
sem = O[(O.tenor_d >= BOT_6M_MIN_D) & (O.tenor_d <= BOT_6M_MAX_D)].copy()
P(f"BOT semestrali identificati ({BOT_6M_MIN_D}-{BOT_6M_MAX_D} gg): {len(sem)}")

off = PROC/"bot_auction_6m_official.csv"
if off.exists():
    s = pd.read_csv(off, index_col=0, parse_dates=True).iloc[:,0].dropna()
    P(f"[fonte UFFICIALE] {off.name}: {len(s)} aste, {s.index.min().date()} -> {s.index.max().date()}")
else:
    px = PROC/"px_bot.csv"
    if not px.exists():
        P(f"[STOP] manca {px.name}: lanciare 02_pull_prices.py")
        P("       in alternativa caricare la serie ufficiale in bot_auction_6m_official.csv")
        save_txt("03_bot_auction.txt", L); print("\n".join(L)); raise SystemExit
    PX = pd.read_csv(px, index_col=0, parse_dates=True)
    rows = {}
    for _, b in sem.iterrows():
        if b["isin"] not in PX.columns: continue
        s = PX[b["isin"]].dropna()
        s = s[s.index >= b.issue]
        if s.empty: continue
        p0, d0 = float(s.iloc[0]), s.index[0]
        n = (b.maturity - d0).days
        if not (0 < p0 < 120) or n <= 0: continue
        rows[d0] = (100.0 - p0) / p0 * 360.0 / n * 100.0     # in punti percentuali
    s = pd.Series(rows).sort_index()
    P(f"[fonte DERIVATA dai prezzi] {len(s)} aste ricostruite su {len(sem)} BOT semestrali")
    P("  [caveat] il prezzo del giorno di emissione approssima il prezzo medio ponderato")
    P("           d'asta: scarto tipico pochi centesimi di rendimento. Per la versione")
    P("           finale usare la serie ufficiale dt.tesoro.it.")

if len(s):
    P(f"\ncopertura: {s.index.min().date()} -> {s.index.max().date()} | {len(s)} osservazioni")
    P(f"rendimento: min {s.min():.3f}%, mediana {s.median():.3f}%, max {s.max():.3f}%")
    neg = (s < 0).sum()
    P(f"aste a rendimento NEGATIVO: {neg} ({s[s<0].index.min().date() if neg else '-'} in poi)"
      f"  -> rilevanti per il floor cedolare")
    gap = s.index.to_series().diff().dt.days
    P(f"giorni fra aste: mediana {gap.median():.0f}, max {gap.max():.0f}")
    if gap.max() > 90:
        P(f"  [!] buco massimo {gap.max():.0f} gg attorno a {gap.idxmax().date()}: le cedole CCT")
        P( "      con godimento in quella finestra usano un'asta lontana. Verificare a mano.")
    s.to_frame("bot6m_yield").to_csv(PROC/"bot_auction_6m.csv")
    P(f"[saved] {PROC/'bot_auction_6m.csv'}")
save_txt("03_bot_auction.txt", L); print("\n".join(L))

# ============================== BOT ANNUALI (regola A) ==============================
# Stessa derivazione dei 6M, finestra 340-380 giorni. Serie richiesta dalla regola
# pre-1995 ("media dei rendimenti dei BOT ANNUALI del bimestre"): finche' assente,
# il 05 marca le righe A come param_series="6m_PROXY".
P("\n=== 03 RENDIMENTI D'ASTA BOT 12 MESI (regola A) ===")
ann = O[(O.tenor_d >= BOT_12M_MIN_D) & (O.tenor_d <= BOT_12M_MAX_D)].copy()
P(f"BOT annuali identificati ({BOT_12M_MIN_D}-{BOT_12M_MAX_D} gg): {len(ann)}")
off12 = PROC/"bot_auction_12m_official.csv"
if off12.exists():
    s12 = pd.read_csv(off12, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    P(f"[fonte UFFICIALE] {off12.name}: {len(s12)} aste")
else:
    px = PROC/"px_bot.csv"
    s12 = pd.Series(dtype=float)
    if px.exists() and len(ann):
        PX12 = pd.read_csv(px, index_col=0, parse_dates=True)
        rows12 = {}
        for _, b in ann.iterrows():
            if b["isin"] not in PX12.columns: continue
            s = PX12[b["isin"]].dropna(); s = s[s.index >= b.issue]
            if s.empty: continue
            p0, d0 = float(s.iloc[0]), s.index[0]
            n = (b.maturity - d0).days
            if not (0 < p0 < 120) or n <= 0: continue
            rows12[d0] = (100.0 - p0) / p0 * 360.0 / n * 100.0   # in punti percentuali
        s12 = pd.Series(rows12).sort_index()
        P(f"[fonte DERIVATA dai prezzi] {len(s12)} aste ricostruite su {len(ann)} BOT annuali")
if len(s12):
    P(f"copertura 12m: {s12.index.min().date()} -> {s12.index.max().date()} | {len(s12)} oss.")
    s12.to_frame("bot12m_yield").to_csv(PROC/"bot_auction_12m.csv")
    P(f"[saved] {PROC/'bot_auction_12m.csv'}")
else:
    P("[!] nessuna asta 12m ricostruibile: la regola A resta sul proxy 6M (flag nel 05)")