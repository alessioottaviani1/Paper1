# The CCT–BTP Basis: Is There a Premium for Mark-to-Market Stability in Italy?
## Master Document — Theory, Data, Methodology, and Complete Results

**Status:** pipeline complete on the full 1995–2026 sample; all indexation engines validated against Bloomberg coupons and official auction data; five candidate mechanisms tested under a uniform identification criterion — two confirmed, two rejected, one weakly supported.
**Last updated:** 17 August 2026.
**Reference paper:** Fleckenstein & Longstaff, *The US Treasury floating rate note puzzle: Is there a premium for mark-to-market stability?* (JFE 2020) — henceforth FL.

---

## 1. Research question and headline result

FL document that US Treasury FRNs trade **rich** relative to their fixed-rate replication and attribute the premium to mark-to-market stability: the floater's price barely moves, and investors pay for that. This project runs the same experiment on the Italian market — CCTs (Treasury floaters) swapped to fixed versus nominal BTPs — over three decades and two indexation regimes.

**The headline result is a sign reversal.** The premise of FL replicates perfectly: the CCT is 1.7–4.6× more price-stable than its matched BTP, with the ratio rising in maturity, exactly as FL find for FRNs. But the FL conclusion does not: where the CCT is relatively more stable, it trades **cheaper**, not richer. The stability-premium regression (FL Table 10) yields coefficients of the **opposite sign** — pooled −1.99 (t −4.8) against FL's +5.32 (t 4.5) — and the result survives within maturity buckets and outside crisis years. Italy does not pay for mark-to-market stability. What drives the CCT–BTP basis instead is a two-regime story: a **richness premium in normal times** (consistent with collateral/near-money value) and a **deep illiquidity discount in sovereign stress**, peaking at +169 bp in 2012, transmitted by the sovereign CDS and amplified by a bank-clientele channel.

---

## 2. Institutional setting: one instrument, two indexation regimes

The design exploits a natural experiment that the US market does not offer: the CCT changed its indexation mechanism in 2010, splitting the sample into two regimes that are never pooled.

**CCT-BOT (issues before 2010).** Coupon indexed to the *sovereign's own short-term debt*: the semi-annual coupon equals half the simple gross yield of the reference BOT auction plus a spread, rounded to the nearest 5 cents (MEF rule). Three sub-rules by issue date: (A) pre-1995 issues — arithmetic mean of the **12-month** BOT auctions in the two-month window ending one month before the accrual start; (B) 1995–2009 issues — the last **6-month** BOT auction preceding the accrual start; spread epochs 0.50/0.30/0.15. Price convention ACT/ACT.

**CCTeu (issues from 2010).** Coupon = 6-month Euribor (fixing two TARGET business days before the accrual start) + issue spread, ACT/360, with a **zero floor on the total coupon** (MEF circular, March 2016). Spreads 0.55–1.85%.

The regime split matters for identification: converting the CCT-BOT to fixed uses the **sovereign par yield from our own fitted curve** (model-based, declared), while the CCTeu uses the **market IRS on Euribor 6M** (EUSA grid) — so any effect that operates through the swap curve can be present only in the CCTeu leg. Section 8 uses exactly this asymmetry as the decisive test of whether sovereign risk moves the basis economically or mechanically.

**The zero floor is a genuine construction difference from FL.** In FL's data the floor is worth identically zero (T-bill auction yields cannot be negative). In ours, Euribor 6M was negative for seven years: the floor binds in 1.0% of CCTeu observations and the coupon sits within 25 bp of it in 8.6%. Version 1 does not price the floorlet (declared); the first-period floorlet enters correctly through the known current coupon. This is presented as a contribution of the Italian laboratory, not a caveat.

---

## 3. Three basis measures

**(1) ISIN vs ISIN, yields only.** Synthetic fixed yield of the swapped CCT minus the yield of the matched BTP (pairing: nearest maturity within 6 months, coverage constraint over the CCT's life, on-the-run tie-break). Model-free but demanding; guarded by an admissibility filter (τ ≥ 1y, |mismatch| relative to τ).

**(2) Curve vs curve.** A Nelson–Siegel fit of the swapped-CCT cross-section against the nominal curve — the term structure of the basis, which FL cannot compute (their FRNs are all two-year). Thin cross-section (~10 CCTs/day, RMSE 11.6 bp median): read as qualitative confirmation of the non-parametric bucket profile, never as point estimates.

**(3) ISIN vs curve — the primary measure.** The synthetic's cash flows (spread + fixed-for-floating conversion, first coupon at its *known* fixed value) priced on the sovereign zero curve versus the CCT's dirty market price. Identical cash flows on both sides — FL's gold standard — with no maturity twin required.

Two design choices deserve emphasis. First, **using the par swap rate K sidesteps the OIS-versus-Euribor discounting problem entirely**: a par swap has zero value under any discounting, so the synthetic requires no stance on the discount curve. Second, **no STRIPS enter the construction** (residual flows are discounted on the fitted BTP+BOT curve, the most liquid instruments in the market), so the Hartley–Jermann (JFE 2024) critique of FL — that their replication relies on extremely illiquid strips — does not apply here by construction.

**Internal consistency (identity check).** basis(1) = basis(3) + coupon wedge + (matched BTP's deviation from the curve). Realised: CCT-BOT −19.0 vs −14.9 bp with the −4.2 bp BTP-off-curve term accounting for the gap; CCTeu 9.7 vs 8.8 bp (difference 1.0 bp). The measures agree; the wedge is reported, never subtracted.

---

## 4. Data

**Prices (Bloomberg).** Daily PX_MID for 113 CCTs and 80 matched BTPs, 1995-01-03 → 2026-08-12. The 1995 start is structural, not chosen: it coincides with the inception of electronic secondary-market pricing on the MOT; earlier CCT prices exist in no machine-readable official source. CCT-BOT coverage by issue cohort: 14/21 (1985–89, priced only near their mid-1990s maturities), 31/31 (1990–94), 25/27 (1995–99), all thereafter.

**Auction yields (Banca d'Italia, official).** The indexation parameter is the auction's weighted-average simple gross yield — precisely what the MEF rule prescribes. Sources: per-year PDFs of 12-month BOT auctions 1983–2011 and the all-instruments auction archive 2002–2025, parsed by a dedicated pipeline script into two series: **12M official, 596 auctions, 1994 → 2025** (rule A and the extended sample; 93 auctions inside 1995–98) and **6M official, 559 auctions, 2002 → 2025** (rule B). For 1996–2001 the 6M series is completed by a Bloomberg-derived proxy (yield at the first post-issue price, 60 auctions); the combined 6M series holds 731 auctions. Sanity: the 12M series descends 9.46% (1994) → 4.48% (1998) → −0.35% (2021) → 2.54% (2024).

**Swap and CDS data (Bloomberg).** EUSA grid (annual 30/360 fixed vs Euribor 6M, converted to its semi-annual equivalent in the synthetic), Euribor 6M fixings, sovereign and bank CDS (UniCredit, Intesa, BPM — constant-composition index), ECB bank-holdings aggregates.

**Documented data boundaries.** (i) 31 rule-B coupon determinations with fixings before June 1996 lack any BOT price source (electronic BOT pricing begins March 1996) and are excluded — 2% of CCT-BOT coupons, concentrated where few CCTs are priced anyway. (ii) A 153-day auction gap around September 1998 in the 6M proxy: coupons fixing in that window use a distant auction (flagged). (iii) Bloomberg has no BOT or CCT prices before 1995–96 for the same MOT reason; the pre-1995 coupons of rule-A bonds are outside the sample by construction and marked `no_data`.

---

## 5. Validation of the indexation engines

The coupon engine is validated **against realised Bloomberg coupons**, the project's permanent regression test:

- **CCT-BOT formula:** the MEF-literal rule (semi-annual coupon = 0.5·y + s, i.e. annual y + 2s) beats the alternative 0.5·(y+s) in **45/45** verifiable bonds, with a median absolute error of **0.02 percentage points** on official auction data (residual exactly 0.000 wherever the coupon period matches the schedule's last row; matured bonds carry a declared period-mismatch caveat).
- **CCTeu engine:** on the nine live CCTeus, round(Euribor,3)+spread reproduces the Bloomberg coupon exactly in **8/9** (median error 0.0000). The ninth (+0.025) is a newly issued bond whose first accrual runs from the settlement date rather than the issue date (noted in code; sub-basis-point effect on one coupon).
- The TARGET-calendar fixing correction was validated live: a −2-business-day fixing straddling the 1 May TARGET holiday moved the read Euribor from 2.524 to 2.462 against Bloomberg's 2.437, closing two-thirds of the only residual.

Three matured bonds display a suspicious exact-zero parameter in the verification table (IT0001415402, IT0003097109, IT0003222087) — a residual diagnostic item on the combined 6M series, flagged for a targeted check; it does not touch live pricing.

---

## 6. The basis: levels, history, and term structure

**Full-sample medians of the primary measure** (positive bp = CCT yields more = cheap):

| regime | obs. | window | price (per 100) | yield (bp) |
|---|---|---|---|---|
| CCT-BOT | 89,672 | 1995-01 → 2015-12 | +0.442 | **−14.9** |
| CCTeu | 29,241 | 2010-03 → 2026-08 | −0.232 | **+8.8** |

**The two-regime history** (annual medians, measure 3): the CCT-BOT is *rich* through the pre-crisis era — around −10/−28 bp from 1996 to 2006, peaking at **−56.6 bp in 2007** — then flips violently cheap in the sovereign crisis: **+40.7 (2010), +97.3 (2011), +169.2 bp (2012)**, normalising after "whatever it takes" (+14.3 in 2014, +5.0 in 2015). The CCTeu shows the same crisis signature (+27.9/+32.7 bp in 2011–12) and a milder cheapness thereafter (+16.5 in 2022, +24.7 in 2023, +13.6 in 2025). Sub-period medians (measure 3, price): 1999–2004 +0.59 (rich), 2005–08 +0.81 (rich), 2009–12 −1.47 (cheap), 2013–16 −0.08, 2017–21 −0.10, 2022–26 −0.58.

**Term structure (non-parametric buckets, measure 3, bp):**

| period | 0–1.5y | 1.5–3y | 3–4.5y | 4.5–8y |
|---|---|---|---|---|
| 1999–07 | **−75.4** | −39.3 | −19.1 | −9.0 |
| 2008–10 | −12.9 | +9.8 | +30.8 | +39.3 |
| 2011–12 | +40.0 | **+149.7** | +112.4 | +43.8 |
| 2013–16 | +5.1 | +6.4 | +5.7 | −7.3 |
| 2017–19 | +4.7 | +1.2 | +3.4 | +8.1 |
| 2020–21 | +3.7 | +2.7 | +7.3 | +8.1 |
| 2022–26 | +9.6 | +20.1 | +13.9 | +16.9 |

The pre-crisis richness is **front-loaded** (−75 bp at the short end, fading to −9 by 4.5–8y) — the fingerprint of a money-like convenience premium on short floaters. This bucket is clean by construction: pre-2010 observations are CCT-BOT, whose conversion uses the fitted sovereign par yield (no swap extrapolation is involved); the sub-6-month noise band is filtered (τ ≥ 0.5). The crisis cheapness peaks in the 1.5–4.5y belly, where the bank-held stock concentrates. The parametric curve-vs-curve fit confirms the profile qualitatively (declared as such: ~10 bonds, 4 parameters).

**Magnitudes versus FL.** FL call 5.97/9.73 bp "economically large". Our normal-regime medians are of the same order (|basis| median 11.5 bp CCTeu, 26.8 bp CCT-BOT); the crisis is an order of magnitude above (median |basis| 3.36 price points in 2011–12).

---

## 7. The FL experiment: stability replicates, the premium does not

**Table 6 analogue — the premise holds.** Standard deviation of daily price changes, monthly panel (5,601 CCT-months):

| residual life | sd CCT | sd BTP | ratio |
|---|---|---|---|
| 0.5–1y | 0.016 | 0.027 | 1.71 |
| 1–1.5y | 0.019 | 0.046 | 2.43 |
| 2–3y | 0.031 | 0.102 | 3.24 |
| 4–5y | 0.046 | 0.180 | 3.94 |
| 5–8y | 0.054 | 0.248 | **4.63** |

The CCT is dramatically more mark-to-market stable than its matched BTP, more so at longer maturities — FL find 2–3× for FRNs. The *premise* of the stability-premium mechanism is stronger in Italy than in the US.

**Table 10 analogue — the conclusion reverses.** Monthly price basis regressed on (sd BTP − sd CCT), month and year effects, errors clustered by CCT. A positive coefficient would mean the premium pays for stability (FL: +5.32, t 4.47 vs bills; +1.87, t 4.52 vs notes).

| specification | all | CCT-BOT | CCTeu |
|---|---|---|---|
| (B1) raw, as in FL | **−1.99 (−4.8)** | −2.15 (−4.6) | −0.69 (−0.9) |
| (B2) + τ, τ² | −0.59 (−2.2) | −0.08 (−0.5) | +0.41 (+0.5) |
| (B3) + CCT fixed effects | −0.25 (−1.1) | −0.17 (−0.9) | −0.16 (−0.2) |

Within-maturity-bucket estimates (B4), where τ barely varies and no functional form is imposed: −0.33 (0.5–1.5y), −0.19 (1.5–3y), **−2.85 (t −2.1, 3–5y)**, **−0.99 (t −2.1, 5–8y)**. Excluding crisis years: **−1.62 (t −4.9)**. The confounding between sd_diff and maturity, flagged ex ante, is handled explicitly (corr = 0.46 in this panel; the raw FL specification without maturity controls would simply return the term structure of the basis).

**Reading, stated plainly:** relative stability does not command a premium in Italy — if anything, the more stable instrument trades cheaper, significantly so within maturity buckets and outside crises; under full fixed effects the relation dies rather than turning positive. The FL mechanism is a US phenomenon (plausibly tied to the 2014 MMF reform clientele), not a general property of floaters.

---

## 8. What does drive the basis: five mechanisms, one criterion

Every candidate faces the same bar: **survival under within-bond identification** (fixed effects). A coefficient that lives only across bonds or across periods is composition, not a mechanism.

| mechanism | pooled | with fixed effects | verdict |
|---|---|---|---|
| Mark-to-market stability (FL) | −1.99 (−4.8) | −0.25 (−1.1) | **rejected** |
| Near-money / Nagel opportunity cost | +0.06 (+2.4) | −0.20 (−3.2)* | **not supported** |
| Supply scarcity | +0.02 (+2.5) | −0.014 (−2.8) | weak support |
| Sovereign stress (CDS) | −0.98 (−4.2) | **−0.77 (−3.3)** | **survives** |
| Bank clientele × size | −1.43 (−2.4) | **−1.79 (−2.4)** | **survives** |

*The Nagel test carries a declared identification warning: within-year variance of the short rate is 4.5% of the total, so the year-FE specification is nearly uninformative — a different statement from "the mechanism does not exist". Two refinements matter for the write-up. First, the Nagel coefficient is not merely insignificant: under fixed effects it is **significantly negative** (−0.20, t −3.2 with bond and year effects; −0.29, t −2.4 on the CCTeu alone) — within the limited within-year variation, higher short rates coincide with a *cheaper* CCT, the exact opposite of a money-like premium. With the stated identification caveat, we read this as evidence against the near-money channel rather than as a structural elasticity, but the sign reversal is worth a line: it is more informative than a plain rejection. Second, supply scarcity is more than a footnote: under fixed effects the coefficient takes the predicted negative sign and is significant (−0.014, t −2.8), with similarly limited within-year identification (9.4%). The honest tally is therefore **two confirmed, two rejected, one weakly supported** — not "three rejected".

**The decisive test for the sovereign channel** exploits the two-regime construction: sovereign CDS cannot move the CCT-BOT basis mechanically (a single curve prices both legs), yet its coefficient there is **−1.30 (t −4.0)** with fixed effects — *stronger* than in the CCTeu (−0.52, t −2.7), where a mechanical two-curve effect is possible. The effect is economic: sovereign stress makes the floater cheap within the sovereign's own curve — a flight-to-liquidity toward benchmark BTPs, the mirror image of FL's flight-to-stability. The **bank-clientele interaction** (CDS × log amount held) survives at −1.79 (t −2.35) under a constant-composition three-name CDS index, robust to the four-name variant. One honest anomaly: idiosyncratic bank stress orthogonal to the sovereign enters with a **positive** sign (+0.64, t 3.3; +0.31, t 1.9 with FE), against the "banks dump CCTs" prior — reported as such.

---

## 9. The trading strategy (Paper 1 methodology, eq. 4–5)

Entry |basis| ≥ 10 bp, exit at zero, minimum six months to maturity; SW weights fixed at entry (|basis − expanding cross-sectional mean|), EW equal-weighted; DV01-normalised and literal (noDV01) variants.

| universe | trades | Sharpe SW | Sharpe EW | mean SW (t-NW) | exits at convergence |
|---|---|---|---|---|---|
| All | 1,316 | 1.19 | 1.09 | 2.01 (4.7) | 1,221 — **100% profitable** |
| CCT-BOT | 1,133 | 1.48 | 1.52 | 2.75 (4.7) | 1,054 — 100% |
| CCTeu | 183 | 0.89 | 0.58 | 1.38 (3.9) | 167 — 100% |

Threshold robustness is monotone and **increasing** (Sharpe SW 1.12 → 1.46, EW 1.23 → 1.57 across 5–30 bp entries): the 10 bp gate is not merely "not chosen ex post" — it is *conservative*, since stricter entries do better; the baseline understates the strategy rather than optimising it, an argument stronger than mere stability. On the CCTeu subsample, DV01-normalisation inverts the SW/EW ranking (SW 0.89 vs EW 0.58, while the literal EW-noDV01 stands at 0.94): with a median of only 5–6 concurrent trades, per-trade risk normalisation concentrates exposure in short-duration dislocations; no single scheme is elected as baseline on this subsample and all four variants are reported. Volatility-managed versions **worsen** sharply (SW 1.19 → 0.11): the strategy earns precisely when volatility is high — the signature of an arbitrage that pays for balance-sheet risk in crises, consistent with the slow-moving-capital reading and with Paper 1's CDS-bond and BTP-Italia strategies (EW Sharpes 0.79–1.46). Excess kurtosis is high (16.5 all; 26.0 CCTeu): winsorised versions and the SW-cap95 variant (1.17) are reported alongside. MPPM positive at all aversion levels. Convergence trades close in profit in 100% of cases by construction — the sanity identity holds; maturity exits (95 trades) lose a median 0.51 points, as expected for unconverged bases.

---

## 10. Robustness checklist (FL Section 11) — closed and open items

**Closed.** (11.2) No illiquid closure instruments: residuals discount on the fitted BTP+BOT curve; the Hartley–Jermann critique does not apply. (11.5) Tax: CCT and BTP share the identical 12.5% substitute-tax regime — no OID-style asymmetry to correct, unlike FL's US case (to be stated with the normative reference). (11.6) Floor: second-order in frequency (1.0% active) but a positive-value option FL could not have; presented as a contribution. (11.7) Supply: CCT share of the stock reconstructed from AMT_ISSUED of live bonds (not AMT_OUTSTANDING, which is survivorship-biased on matured bonds): 12.9% (1999–2004) → 17.4% (2005–10) → 7.8% (2017–26).

**Open, in referee order.** (1) **ECB haircut schedules** — the most important: floating-rate notes take the 0–1y haircut bucket *regardless of maturity*, a real collateral advantage for the CCT that could explain the normal-regime richness the clientele channel does not; to be documented from Eurosystem haircut tables. (2) **Transaction costs** — one explicit paragraph: benchmark Italian governments trade on tight bid-asks and the crisis basis (3.4 price points median in 2011–12) dwarfs any plausible cost layer; Bloomberg bid-ask history on this market is unreliable and adding it would fake precision. (3) **Security-level holdings** — only via ECB SHS or Banca d'Italia RDC on request; the sectoral aggregate is automated (02c).

---

## Figures — to be produced

Three figures accompany the tables (scripts pending): **(F1)** the annual basis series by regime, 1995–2026, with the two-regime shading (normal-time richness, crisis cheapness) — the paper's signature image; **(F2)** the non-parametric term structure by period (the −75 bp front-loaded richness of 1999–2007 against the 2011–12 belly peak); **(F3)** the cumulative strategy P&L, SW and EW, with the vol-managed variant shown to *underperform* — the visual proof that the strategy earns in crises.

---

## 11. Honesty box — what we do not claim

- **1995 is flagged pending a short-end diagnostic**: BOT prices begin in March 1996, so the 1995 sovereign curve is fitted on BTPs alone with no short-end anchor; the year is anomalous in the series (−1.48 price / +39.5 bp on 1,142 obs., against +0.3 / −10 bp in 1996–98) and coincides with the sample start. Headline results should be re-checked excluding it before the year is retained.
- The floorlet is not priced in v1: near-floor CCTeu observations (8.6% within 25 bp) carry a downward-biased basis by the floorlet value; the first-period floorlet is captured through the known coupon.
- 31 rule-B coupons (fixings before June 1996) and a 153-day 1998 auction window are excluded/flagged; the 6M series is official from 2002 and price-derived 1996–2001.
- Valuation is at trade date; the T+2 settlement convention affects both legs identically and cancels in the basis.
- The curve-vs-curve measure (2) is qualitative by construction (~10 bonds, NS-4).
- The DE-style caveats of Paper 1 apply to inference on the CCTeu subsample (24 bonds, 2010–26, one full crisis).
- Three matured CCT-BOTs show an exact-zero verification parameter — a pending targeted diagnostic on the combined 6M series (does not affect live bonds).
- The first coupon of newly issued CCTeus accrues from settlement, not issue (noted; one coupon, few bp).

---

## 12. Reproducibility

**Pipeline** (`src/cct/`): `00` Bloomberg diagnostics → `01` static universe (129 CCTs, spread epochs, regime split) → `02/02b/02c` prices, extra series, ECB holdings → `03c` **official auction parser** (Banca d'Italia PDFs 1983–2011 + xlsx 2002–2025 → two official series) → `03` auction series assembly (official primary, price-derived proxy fill 1996–2001) → `03b` data QC → `04` pairing (nearest maturity, coverage constraint, on-the-run tie-break) → `05` coupon schedule (rules A/B/C, TARGET fixings, floor, `param_series` provenance flag) → `05b` **permanent regression test** (coupons vs Bloomberg: 45/45 and 8/9) → `06` sovereign NSS on dirty prices (GSW objective; the 1/(D·P) weight is exactly the yield error in continuous compounding, so RMSE is genuinely in bp; median 5.9 bp over 8,231 days 1995–2026) → `06b` curve diagnostics → `07` the three basis measures with the coherence identity → `07b/07c` robustness diagnostics → `08` term structure → `09` FL Tables 6/10 → `10/10c` mechanism (sovereign CDS, constant-composition bank index, clientele) → `11/12` FL robustness checklist and remaining tests (Nagel, supply) → `13` strategy. Modules: `config` (all conventions and dates), `utils`, `qlutils` (TARGET calendar, ISMA day counts), `bbg`.

**Fix log (all permanent, empirically validated where possible).** E1 first coupon already fixed at the previous reset — the synthetic's first flow is now known-coupon + (K − E_today)·stub; the error removed equals ½·ΔEuribor since fixing (~1.4 price points in 2022) and was mechanically correlated with the rate-change regressors of the mechanism tests. E2 EUSA annual 30/360 converted to its semi-annual equivalent, spread accrued ACT/360 (2–6 bp, rate-level-dependent — sat on the Nagel regressor). E3 market-data intersection restricted to CCTeu — the fix that brought the 1995–98 sample to life. E5 Euribor fixing on the TARGET calendar (validated live: 8.7 bp on a May-Day fixing). E7 ultra-long schedule guard. E10 sub-6-month filter. E13 no spot masquerading as future fixings. E8 investigated and **closed as a false alarm**: the flagged "compensated humps" days have the best fit quality (RMSE 5.1 vs 5.9 bp overall) and sane 1–7y zeros — NSS legitimately representing double curvature, no constraint added. E11 counted: **zero** pre-circular floor determinations — a one-line note. E4 rule-A now on official 12M auctions (the 6M-proxy flag retired to `12m`/`no_data` provenance).

**Data provenance summary.** Prices: Bloomberg (MOT-era, 1995→). Indexation: Banca d'Italia official auctions (1994→), Bloomberg-derived proxy 1996–2001 (6M only), Euribor/EUSA: Bloomberg. The extended sample begins in 1995 — the inception of electronic secondary-market pricing for Italian government bonds — and this boundary is structural, not elective.
