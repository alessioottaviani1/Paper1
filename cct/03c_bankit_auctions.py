"""
03c - Serie ufficiali dei rendimenti d'asta BOT (Banca d'Italia), fonte per l'indicizzazione CCT.

Legge le cartelle scaricate da bancaditalia.it in data/raw/Bankit/ e produce due serie:
  - bot_auction_12m_official.csv : BOT annuali (regola A + campione esteso), 1994 -> oggi
  - bot_auction_6m_official.csv  : BOT semestrali (regola B), 2002 -> oggi

Fonti:
  Serie_storica_BOT/*.pdf   : un PDF per anno 1983-2011, SOLO BOT 12M (tabella testuale pulita)
  storico_aste/*.xlsx       : tutte le aste 2002-2025, tutti i titoli (si filtra BOT + durata)

I PDF datano il BOT per DATA DI EMISSIONE (l'asta precede di pochi giorni); l'xlsx ha la
data d'asta vera. Scarto sub-bp sulle cedole semestrali, dichiarato in nota nel paper.
Rendimento = lordo semplice, la convenzione che la norma CCT richiede.
"""
import pandas as pd, re, subprocess, glob
from pathlib import Path
from config import RAW, PROC
from utils import save_txt

BANKIT = RAW / "Bankit"
L = []; P = L.append
P("=== 03c SERIE UFFICIALI RENDIMENTI D'ASTA BOT (Banca d'Italia) ===")

def parse_date(s):
    d, m, y = s.split("/"); d, m, y = int(d), int(m), int(y)
    if y < 100: y = 1900 + y if y >= 83 else 2000 + y   # 83-99 -> 1900, 00-25 -> 2000
    return pd.Timestamp(y, m, d)

def parse_pdf(path):
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                         capture_output=True, text=True).stdout
    rows = []
    for ln in txt.splitlines():
        m = re.match(r"\s*(\d{2}/\d{2}/\d{2,4})\s+(\d{2,3})\s+([\d.,]+)\s+([\d.,]+)", ln)
        if m:
            dt, dur, _prezzo, rend = m.groups()
            try: rows.append((parse_date(dt), int(dur), float(rend.replace(",", "."))))
            except Exception: pass
    return rows

# --- PDF 12M storici (1983-2011) ---
pdf_dir = BANKIT / "Serie_storica_BOT"
pr = []
for f in sorted(glob.glob(str(pdf_dir / "*.pdf"))): pr += parse_pdf(f)
pdf = pd.DataFrame(pr, columns=["date", "dur", "rend"])
pdf12 = pdf[(pdf.dur >= 330) & (pdf.dur <= 380)].copy()
P(f"PDF 12M: {len(pdf12)} aste da {len(glob.glob(str(pdf_dir/'*.pdf')))} file, "
  f"{pdf12.date.min().date()} -> {pdf12.date.max().date()}")

# --- xlsx moderno (2002-2025, tutti i titoli) ---
xlsx = next(iter(glob.glob(str(BANKIT / "storico_aste" / "*.xlsx"))), None)
x = pd.read_excel(xlsx, skiprows=3, header=None)
x.columns = ['data_asta','data_reg','isin','tranche','ord_supp','descr','data_scad','coeff',
             'tipo','off','min_off','max_off','rich','ass','prezzo','rend_bot','rend_altri','n_oper']
xb = x[x['tipo'] == 'BOT'].copy()
xb['date'] = pd.to_datetime(xb['data_asta'], errors='coerce')
xb['dur']  = (pd.to_datetime(xb['data_scad'], errors='coerce') - xb['date']).dt.days
xb['rend'] = pd.to_numeric(xb['rend_bot'], errors='coerce')
xb = xb.dropna(subset=['date', 'rend'])
x12 = xb[(xb.dur >= 330) & (xb.dur <= 380)][['date', 'rend']]
x6  = xb[(xb.dur >= 150) & (xb.dur <= 210)][['date', 'rend']]
P(f"xlsx 12M: {len(x12)} aste | xlsx 6M: {len(x6)} aste, {xb.date.min().date()} -> {xb.date.max().date()}")

# --- unione: 12M = PDF (fino 2011) + xlsx (dal 2012); 6M = solo xlsx ---
s12 = (pd.concat([pdf12[['date','rend']], x12[x12.date >= '2012-01-01']])
         .drop_duplicates('date').sort_values('date').set_index('date')['rend'])
s6  = x6.drop_duplicates('date').sort_values('date').set_index('date')['rend']
s12 = s12[s12.index >= '1994-01-01']    # prima non servono: niente prezzi CCT
s6  = s6[s6.index >= '1994-01-01']

s12.to_frame('bot12m_yield').to_csv(PROC / "bot_auction_12m_official.csv")
s6.to_frame('bot6m_yield').to_csv(PROC / "bot_auction_6m_official.csv")
P(f"[saved] bot_auction_12m_official.csv: {len(s12)} aste, {s12.index.min().date()} -> {s12.index.max().date()}")
P(f"[saved] bot_auction_6m_official.csv : {len(s6)} aste, {s6.index.min().date()} -> {s6.index.max().date()}")

# --- diagnostica ---
P(f"\n6M pre-2002 (regola B 1995-2001): {int((s6.index < '2002-01-01').sum())} aste "
  f"-> il tratto 1995-2001 della regola B resta sul proxy 6M da prezzi (03)")
P(f"12M nel campione esteso 1995-98: {int(((s12.index>='1995-01-01')&(s12.index<='1998-12-31')).sum())} aste")
P("\ncontrollo di sanita' 12M (media annua, deve scendere 9% -> ~0% -> risalire):")
for yr in [1994, 1998, 2002, 2008, 2012, 2021, 2024]:
    v = s12[s12.index.year == yr]
    if len(v): P(f"  {yr}: {v.mean():5.2f}%  (n={len(v)})")

save_txt("03c_bankit.txt", L); print("\n".join(L))
