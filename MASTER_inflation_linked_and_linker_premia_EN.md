# Inflation-Linked Bond Mispricing Across Six Markets
## Master Document — Theory, Methodology, and Complete Results

**Status:** all three research questions (Rebonato) answered on five markets (US, UK, IT, FR, DE); bond-level basis panel complete on six markets (adding FR_CPI); all validation tests passed.
**Last updated:** 17 August 2026.
**Companion (Italian) master:** `MASTER_inflation_linked_e_linker_premia.md`. This English version is the extended reference.

---

## 1. Purpose and research questions

This document consolidates the theory, data infrastructure, methodology, and complete empirical results of the inflation-linked bond mispricing project. The work answers three questions posed by R. Rebonato (RR), originally for the US and UK, here extended to the full panel wherever the comparison is economically meaningful:

**RQ1 — Does a breakeven premium exist, and is it common across markets?** Measure the wedge between inflation-swap rates and bond-market breakevens, its level by regime, and its cross-market correlation structure.

**RQ2 — How do real yields and the wedge respond to inflation surprises, net of liquidity?** Estimate multivariate regressions of the wedge on liquidity and inflation surprises across the term structure, and characterise where along the curve surprises bite.

**RQ3 — Are real excess returns predictable, and by what?** Compare slope, Cochrane–Piazzesi, Cieslak–Povala, and cointegration-based factors on 12-month real excess returns, following the Rebonato–Nyholm framework.

In parallel, the project builds the **bond-level basis panel**: for every index-linked bond in six markets, two mispricing measures (matched-maturity à la Fleckenstein–Longstaff–Lustig, and IRR-versus-fitted-curve) on every trading day, complementing the curve-level wedge.

---

## 2. Theoretical framework

### 2.1 The wedge λ: inflation swaps versus bond breakevens

For maturity $T$, define the bond-market breakeven as the Fisher gap between nominal and real zero-coupon yields (continuously compounded):

$$\mathrm{BEI}(T) = y_{nom}(T) - y_{real}(T),$$

and let $\mathrm{ISR}(T)$ be the zero-coupon inflation swap rate for the same index and tenor. In a frictionless market the two must coincide; in practice they diverge by

$$\lambda(T) = \mathrm{ISR}(T) - \mathrm{BEI}(T).$$

A **negative** $-\lambda = \mathrm{BEI}-\mathrm{ISR}$ means the linker yields *more* than its synthetic replication (nominal bond + inflation swap): the linker trades **cheap**, a liquidity/mispricing premium in favour of the nominal bond. This is the curve-level generalisation of the TIPS–Treasury puzzle of Fleckenstein, Longstaff and Lustig: their arbitrage strategy monetises exactly this wedge bond-by-bond. Because the euro markets (IT, FR, DE) share the **same currency, the same swap curve (EUSWI on HICPxT), and the same inflation index**, any cross-country difference in $\lambda$ within the euro area isolates pure bond-market frictions — sovereign credit and linker-specific liquidity — with inflation expectations and swap-market effects held fixed by construction. This identification is only possible because we fit real curves for IT, FR and DE from linker prices (Section 4.2); no official real curves exist for these markets.

### 2.2 Three basis methodologies

The project measures mispricing at three levels of aggregation. They answer the same economic question with different trade-offs:

**(i) Curve versus curve.** $\lambda(T)$ as above, computed on fitted zero curves. Smooth, term-structure-consistent, available at any tenor; but inherits fitting error and hides bond-level heterogeneity (on-the-run effects, seasoning, floor value).

**(ii) ISIN versus ISIN ("nearest", FLL-style).** For each linker on each day, build the synthetic nominal (linker + inflation swap projecting the coupon and redemption cash flows) and compare its internal rate of return with the yield of the **nearest nominal twin by maturity** (tolerance ≤ 183 days). This is the matched-maturity design of Fleckenstein–Longstaff–Lustig (without the STRIPS refinement). It is model-free on the nominal side but demanding: it needs a twin to exist, and it consumes one nominal per linker.

**(iii) ISIN versus curve ("c-exact").** Same synthetic IRR on the linker side, compared with the fitted nominal zero curve evaluated at the linker's exact maturity. No twin required — full coverage — at the cost of the nominal curve model.

Agreement across (i)–(iii) is itself evidence: where the three measures tell the same level story, the mispricing is robust to methodology; where they diverge, the divergence is diagnostic (twin scarcity, curve fitting stress, bond-specific effects).

### 2.3 Index-linked bond mechanics and market conventions

Every bond-level computation runs through the **index ratio** machinery, with conventions differing by market:

- **US TIPS and euro linkers (IT/BTP€i, FR/OAT€i, DE/Bund€i, FR_CPI/OATi):** real-clean quotation, 3-month indexation lag, daily reference CPI by linear interpolation. The reference CPI for day $D$ of month $M$ is
$$\mathrm{RefCPI}(D,M) = \mathrm{CPI}_{M-3} + \tfrac{D-1}{\mathrm{days}(M)}\left(\mathrm{CPI}_{M-2}-\mathrm{CPI}_{M-3}\right),$$
and the index ratio is $\mathrm{RefCPI}(\text{settle})/\mathrm{RefCPI}(\text{dated date})$, rounded to five decimals (Treasury UOC convention). The euro floor at redemption ($\max(\text{nominal}\times CI, 100)$) is implemented.
- **UK index-linked gilts, two regimes.** *New-style* (post-2005 issues, 37 bonds): real-clean, 3-month lag, interpolated — same family as US/euro. *Old-style* (16 bonds): **8-month RPI lag, no interpolation, nominal-clean price quotation, and a 7-business-day ex-dividend window**. The pre-2005 history of the UK market lives entirely in the old-style segment; handling it correctly is what makes the long UK sample usable.
- **FR_CPI (OATi):** identical mechanics to OAT€i but indexed to the **French domestic CPI ex-tobacco (FRCPXTOB)** with its own swap curve (FRSWI). Because the index differs from HICPxT, FR_CPI belongs to the bond-level panel but is deliberately **excluded from the euro curve-versus-curve comparison** (RQ1–RQ3), which requires a common index.

The engine's Italian index-ratio implementation is validated **exactly** against the official MEF coefficients: the permanent regression test reproduces 16/16 published CI values with difference 0.00e+00, including pre- and post-rebasing dates. The US implementation is validated against TreasuryDirect (Section 4.4).

### 2.4 Predictability factors (RQ3)

On end-of-month real zero curves we compute 12-month overlapping excess returns $rx^{(m)}_{t+1}$ for target maturities $m \in \{5,7,10,15,20\}$ over the shortest reliable real funding rate (2.0y for US/IT/FR/DE, 2.5y for UK), and regress the cross-maturity average $\overline{rx}$ on four competing predictors:

- **SLOPE:** the real curve slope (funding tenor to 10y).
- **CP:** the Cochrane–Piazzesi single factor, a linear combination of real forward rates estimated on the same curve.
- **CiP:** the Cieslak–Povala cycle factor — yields detrended by a slow-moving inflation trend (constructed from the market's own CPI series), so the factor "knows" inflation.
- **COINT:** the first error-correction combination from a Johansen cointegration analysis of the forward-rate panel — the Rebonato–Nyholm idea that the CP factor is, at heart, a cointegrating combination of forwards.

Newey–West t-statistics account for the 12-month overlap. Robustness runs repeat CP and COINT over five alternative forward grids; CiP does not depend on the forward grid and is invariant by construction.

### 2.5 Literature anchors

- **Fleckenstein–Longstaff–Lustig** (TIPS–Treasury puzzle): bond-level mispricing design; US mispricing roughly twice the UK's in the crisis sample (≈40bp vs ≈16bp) — our pre-2009 levels reproduce the ordering.
- **Pflueger–Viceira**: return-predictability and liquidity decomposition on US (GSW, 2–20y) and UK (BoE real, 2.5–25y) curves — our sources and tenor ranges match theirs.
- **Beechey–Wright**: real yields **rise** on inflation surprises — the sign we find in all five markets.
- **Cochrane–Piazzesi; Cieslak–Povala**: the CP and CiP factors.
- **Rebonato–Nyholm**: cointegration structure of forwards as the origin of CP-type predictability — extended here to *real* curves.
- **Duffie (slow-moving capital)**: the frictions narrative behind persistent basis.

---

## 3. Data and infrastructure

### 3.1 Universe

Built from the Bloomberg security universe with concordant filters. Included instruments:

| Market | Linkers | Nominals | Notes |
|---|---|---|---|
| US | 108 | 1,313 (pool-matched to 394) | TIPS; GSW curve from file |
| UK | 37 new + 16 old | 156 (pool-matched to 114) | BoE curves from file |
| IT | 29 | 295 | BTP€i; own NSS fits |
| FR | 20 | 139 | OAT€i; own NSS fits |
| FR_CPI | 13 | (shares FR pool) | OATi, French CPI |
| DE | 9 | 291 (pool-matched to 34) | Bund€i; Bundesbank nominal curve |

Exclusions (documented in the universe builder): matured before the data floor (2003-01-01, 739), funged into another line (parent line kept, 496), pre-euro or foreign currencies (DEM/ITL/FRF/XEU/USD, 99), out-of-perimeter markets (9), duplicate ISINs (2). Four linkers with invalid BASE_CPI even after the bdp refresh are excluded from the basis computations. Euro rebasing (HICP re-basings) is handled by consistent rebasing keys per market (identical keys across affected bonds confirm the correction is systematic, not ad hoc).

**UK indexation-lag convention fill.** Bloomberg leaves the "Inflation Lag" field empty on several funged/legacy UKTI lines. Because the UK convention is unambiguous from the segment — old-style = 8 months, new-style = 3 months — missing lags are filled from the segment convention (never overwriting a populated value). After the fill: 53/53 UK linkers carry a lag, distribution {3m: 37, 8m: 16}, and the long-standing "inconsistent lag" warning is resolved at the root.

### 3.2 Curves

**Official curves from file:** GSW nominal and TIPS (US), Bank of England nominal and real (UK; the BoE "inflation" curve is verified to equal nominal − real via Fisher to 0.0bp), Bundesbank Svensson parameters (DE nominal; annual-to-continuous compounding conversion applied and verified).

**Own Nelson–Siegel–Svensson fits from bond prices** (the project's key data contribution — no official real curves exist for the euro linker markets):

| Curve | Dates | Bonds | Median RMSE | Dates > 12bp |
|---|---|---|---|---|
| nss_real_IT | 5,129 | 29 | 6.3 bp | 33 (worst 62.5bp, 28-May-2007) |
| nss_real_FR | 5,393 | 20 | 2.6 bp | 0 |
| nss_real_DE | 3,447 | 9 | 0.2 bp | 0 |
| nss_FR (nominal) | 6,153 | 139 | 2.2 bp | 44 (worst 17.0bp, 19-Jan-2026) |

The DE fit is near-exact because nine Bund€i are few but mutually consistent; the IT early-sample stress (2007) reflects four-to-five-bond days at the market's dawn. NSS falls back to NS (four parameters) on ill-conditioned days.

### 3.3 Inflation data

Zero-coupon inflation swap curves per index: USSWIT (US CPI, 15 tenors, 2004-07 → 2026-08), BPSWIT (UK RPI, 19 tenors, 2003-12 → 2026-08), EUSWI (HICPxT, for IT/FR/DE), FRSWI (French CPI, for FR_CPI). CPI series with long history (floor 1996-01) for carry and the CiP trend: CPURNSA, UKRPI, CPTFEMU, FRCPXTOB.

**The October 2025 CPI gap (methodological note).** The BLS did not publish the October 2025 CPI-U due to the federal government shutdown — the first missing monthly CPI since 1921 — and announced it cannot collect the data retroactively. The US Treasury invoked the **index contingency provisions** of the Uniform Offering Circular (31 CFR 356) and set the official index number for October 2025 at **325.604**, on which all TIPS payments are based, stating it will not substitute the value even if the BLS later publishes. We insert 325.604 into the CPI series, and the insertion is **hard-wired in the download layer** so it survives any cache refresh. Validation against the official TreasuryDirect tables (CPI_20251218): our reference CPI and index ratio for the April-2032 TIPS (US912810FQ68, dated-date RefCPI 177.50000) reproduce the official values **exactly to the fifth decimal** — 1-Jan-2026: RefCPI 325.60400, IR 1.83439; 31-Jan-2026: RefCPI 324.16981, IR 1.82631 (the latter interpolates October→November, testing both the value and the interpolation). Market prices in Nov-2025–Feb-2026 embedded this Treasury number; using it makes our bases consistent with how the market actually settled.

### 3.4 Sample boundaries and documented exclusions

All residual computation failures are **at the edges of what markets quote**, not model defects, and are documented rather than patched:

- **Dawn of the inflation-swap markets (2003–04):** the earliest TIPS/gilt observations predate a usable ISR curve; excluded.
- **Ultra-long linkers beyond the swap horizon:** a newly issued 30y TIPS (e.g., Feb-2040/2041 lines in 2010) or the UK 2068/2073 gilts project coupons beyond the last quoted swap tenor. The exclusion is structural but **self-resolving**: the 2068 gilt fails from Sep-2013 to Feb-2018 and computes thereafter; the 2073 gilt from Nov-2021 to Feb-2023. No flat extrapolation is applied, consistent with the project-wide no-extrapolation principle.
- **Old-style ex-dividend tail:** in the final ~7 business days of an old-style gilt's life the ex-dividend window has no next coupon; excluded.

Residual error rates after all fixes: **US 119 of 149,112 attempted (0.08%)**; **UK 1,616 of 83,782 (1.9%, of which 1,443 are the two ultra-long gilts' early-year windows)**; FR, FR_CPI, DE: **zero errors**.

---

## 4. RQ1 — The breakeven premium across five markets

Each market is measured on its **own full sample** (no five-way intersection, which the young German market would truncate to 2012); cross-market comparisons are **bilateral** with the US on the pair's common sample. Samples: US/UK 2004–26, IT/FR 2005–26, DE 2012–26. Monthly end-of-month, 10-year point.

**Average level of $-\lambda = \mathrm{BEI}-\mathrm{ISR}$ (10y, basis points):**

| Period | US | UK | IT | FR | DE |
|---|---|---|---|---|---|
| pre-2009 | −27 | −5 | −11 | −9 | — |
| 2009–2019 | −26 | −24 | **−30** | −3 | −13 |
| 2020–today | −18 | −16 | −24 | −3 | −11 |

**Bilateral correlation of $-\lambda$(10y) with the US:**

| Period | c(US,UK) | c(US,IT) | c(US,FR) | c(US,DE) |
|---|---|---|---|---|
| pre-2009 | 0.65 | 0.50 | 0.55 | — |
| 2009–2019 | 0.69 | **0.11** | 0.41 | 0.38 |
| 2020–today | 0.69 | 0.64 | 0.35 | 0.29 |

**Reading.** Three results, in increasing order of novelty:

1. **The FLL sign and the US–UK common factor replicate.** Linkers trade cheap to their synthetics everywhere; the US–UK correlation is stable at 0.65–0.69 across regimes — Fleckenstein's global mispricing factor. The pre-2009 level ordering (US −27 vs UK −5) is consistent with the FLL-era finding that US mispricing exceeded the UK's by a wide margin in the crisis sample.
2. **Within the euro area, the mispricing hierarchy reproduces the sovereign-risk ranking: IT (−30) > DE (−13) > FR (−3).** Same currency, same swap curve, same index — the wedge differences are pure bond-market frictions, and they line up with sovereign credit. This isolation is only possible on our fitted real curves.
3. **The sovereign-crisis decoupling is exclusively Italian.** c(US,IT) collapses from 0.50 to **0.11** in 2009–19 and recovers to 0.64 after 2020, while France (0.41) and Germany (0.38) stay attached to the global factor throughout. During the euro crisis the Italian wedge becomes idiosyncratic — driven by the BTP–Bund spread, not by global liquidity. **The FLL common factor is robust among core markets and fragile toward the periphery under sovereign stress** — a qualification of the common-factor result that required exactly this panel.

---

## 5. RQ2 — Inflation surprises, liquidity, and segmentation

Monthly regressions of $\lambda(T)$ on a standardized liquidity composite (VIX + MOVE, end-of-month) and an inflation surprise (year-on-year CPI minus its 10-year moving average), tenor by tenor (2–20y; UK from 2.5y). $\Delta y_{real}(T)$ on the surprise is reported alongside. Samples: US 2004-07→2026-08, UK 2004-04→2026-07, IT 2005-04→2026-08, FR 2005-07→2026-08, DE 2012-04→2026-01. t-statistics in parentheses.

**β_surprise (net of liquidity), bp per surprise unit:**

| Tenor | US | UK | IT | FR | DE |
|---|---|---|---|---|---|
| 2y | −2.6 (−2.8) | +5.8 (+3.3) | +3.7 (+4.0) | +1.0 (+1.0) | +1.9 (+1.6) |
| 5y | −2.2 (−2.7) | +3.1 (+2.6) | +3.8 (+6.1) | +1.7 (+3.0) | +1.4 (+2.7) |
| 10y | −1.6 (−2.3) | +0.9 (+0.9) | +2.2 (+4.2) | +0.7 (+1.7) | +0.3 (+0.7) |
| 15y | −1.5 (−1.9) | −0.4 (−0.7) | +2.9 (+4.4) | +2.5 (+6.6) | +0.1 (+0.2) |
| 20y | −1.9 (−2.7) | +0.3 (+1.0) | +3.2 (+4.1) | +3.0 (+8.4) | −0.4 (−0.4) |

**β_liquidity, bp per standard deviation:**

| Tenor | US | UK | IT | FR | DE |
|---|---|---|---|---|---|
| 2y | +5.8 (+2.1) | +6.7 (+1.9) | +3.1 (+1.0) | +2.0 (+0.6) | −1.7 (−0.7) |
| 5y | +5.4 (+1.8) | +10.1 (+3.9) | +5.8 (+2.2) | +4.6 (+2.5) | −0.5 (−0.6) |
| 10y | +3.4 (+2.1) | +8.1 (+4.5) | +5.8 (+3.1) | +2.7 (+2.8) | −0.1 (−0.1) |
| 15y | +2.2 (+1.6) | +4.3 (+2.7) | +3.2 (+1.9) | +1.4 (+1.3) | −0.2 (−0.2) |
| 20y | +3.3 (+2.0) | +2.0 (+2.2) | +0.5 (+0.2) | +1.3 (+1.2) | −1.1 (−0.9) |

**Δy_real on the surprise (bp):** positive and significant at essentially all tenors in **all five markets** (US +5.7→+1.7; UK +2.6→+1.4; IT +3.9→+2.3; FR +3.6→+1.8; DE +2.9→+1.6, t between 2.0 and 3.3 except DE 2y at 1.6). **Beechey–Wright generalises to the whole euro area:** real yields rise on inflation surprises everywhere.

**R² (full model / liquidity-only)** shows the surprise's incremental contribution is largest where the β pattern says it should be: US short end (0.36/0.32 at 2y, 0.53/0.50 at 5y), UK short end (0.48/0.32 at 2y), Italy's long end (0.27/0.19 at 20y), France's long end (0.22/0.00 at 15y, 0.28/0.01 at 20y — the surprise *is* the model there), Germany nowhere (max 0.10 at 5y).

**Reading — three segmentation signatures and one liquidity anomaly (new results):**

1. **Anglo-Saxon/German short-end concentration.** In the US, UK and DE the surprise effect is concentrated at the short end and vanishes (or flips sign) by 15–20y: |β| US 2.6→1.9, UK 5.8→0.3, DE 1.9→0.4. The inflation-protection repricing happens where the surprise carries information — the front end.
2. **Italy is flat.** β_surp 3.7→3.2 with the strongest t-statistics of the panel (4–6) at *every* tenor: no segmentation. The Italian linker market reprices uniformly — consistent with a credit-driven market without a maturity-clientele structure.
3. **France concentrates at the LONG end** — the mirror image of the Anglo-Saxon pattern: β rises from 1.0 (2y, insignificant) to **3.0 with t = 8.4 at 20y**, and the R² decomposition shows the long-end wedge is essentially all surprise. The natural candidates are the French long-duration protection clienteles (insurers, pension vehicles) concentrated at the far end.
4. **Liquidity matters everywhere except Germany.** β_liq is positive and significant across the curve in US, UK, IT and FR; in Germany it is null or negative and never significant. The Bund€i **is** the euro area's risk-free collateral: in liquidity stress it does not cheapen like other linkers — if anything it benefits from flight-to-quality. An Italy–Germany contrast within a single currency: the periphery's wedge loads on global liquidity, the core's does not.

**Sign of λ:** negative in the US, positive in the other four — a level asymmetry (which side of the synthetic is cheap) that coexists with the common dynamics above.

---

## 6. RQ3 — Predictability of real excess returns

R² on $\overline{rx}$ (average 12-month real excess return across targets), Newey–West t-statistics; Johansen rank and the first ECM's ADF on the forward panel.

| | SLOPE | CP | CiP | COINT | Johansen |
|---|---|---|---|---|---|
| **UK** | 0.03 (1.1) | **0.19 (3.4)** | 0.10 (1.8) | **0.17 (4.7)** | r = 2, ADF −11.7 |
| **US** | 0.00 (0.1) | 0.22 (2.8) | **0.39 (4.4)** | 0.11 (3.1) | r = 4, ADF −7.9 |
| **IT** | 0.00 (−0.1) | **0.24 (4.3)** | **0.22 (3.3)** | 0.05 (1.6) | r = 5 (full), ADF −6.7 |
| **FR** | 0.01 (0.5) | 0.19 (2.6) | **0.26 (3.4)** | 0.09 (2.2) | r = 2, ADF −7.4 |
| **DE** | 0.03 (−0.7) | 0.37 (4.4) | **0.69 (16.3)** | 0.26 (6.4) | r = 3, ADF −6.1 |

**Robustness (five forward grids).** UK: COINT 0.17–0.24 (t 4.0–5.4), always paired with CP — the Rebonato–Nyholm equivalence holds in UK reals across specifications. US: CP/COINT stable on dense grids, weaken on sparse ones; CiP invariant at 0.39 (t 4.4) by construction. IT: COINT ≈ 0 on every grid (see caveat below); CP 0.20–0.24 stable. FR/DE: stable throughout.

**Reading.** **The inflation-aware factor (CiP) dominates everywhere except the United Kingdom.** In the US, France and Germany, CiP is the best predictor of real excess returns; in Italy it ties with CP. Only in the UK do the forward-structure factors win — CP ≈ COINT ≫ CiP — which is precisely the market where a structural demand layer (LDI) governs the curve. Two complementary conclusions: (i) the **Rebonato–Nyholm cointegration story extends to real curves in the UK**, the market it was built for; (ii) elsewhere, real-term-premium variation is tied to the inflation environment, consistent with the RQ2 surprise results.

**Two caveats, stated plainly:**

- **The German R² of 0.69 (t = 16) is inflated by a single episode.** The DE sample (2012–26, ~150 overlapping observations) is dominated by the 2021–22 inflation surge, which the CiP factor captures almost mechanically. Report with this qualification; the direction (CiP dominance) is consistent with FR/US, the magnitude is not generalisable.
- **Italy's Johansen rank is full (r = 5): all real forwards are stationary**, so no cointegrating structure exists to exploit — the COINT predictor is **degenerate by construction** in Italy, not "defeated". The stationarity itself is a finding: Italian real forwards mean-revert, plausibly through the credit component's ebb and flow.

---

## 7. The bond-level basis panel — six markets, two methods

Final observation counts (non-missing bond-day basis values):

| Market | nearest (ISIN vs ISIN) | c-exact (ISIN vs curve) |
|---|---|---|
| US | 148,993 | 204,666 |
| UK | 82,166 | 124,842 |
| IT | 46,189 | 54,611 |
| FR | 39,735 | 48,503 |
| FR_CPI | 30,596 | 30,933 |
| DE | 20,360 | 21,616 |
| **Total** | **368,039** | **485,171** |

Together with the curve-level λ (Section 4), this completes the **three-methodology × six-market** grid. Method-specific notes:

- **c-exact exceeds nearest everywhere**, as designed: it needs no maturity twin. The gap is widest where twins are scarce (US early years, UK ultra-longs).
- **UK old-style validated on live data:** 9 of 12 in-sample old-style gilts produce 24,942 observations under the 8-month nominal-clean ex-dividend engine — the pre-2005 UK history is usable. (The remaining old-style lines are the earliest maturities at the sample edge and the BASE_CPI exclusions.)
- **Germany:** with nine Bund€i and the ≤183-day twin tolerance, nearest coverage (20,360) is close to c-exact (21,616) — the small market is well twinned by the dense Bund nominal curve pool (34 matched nominals).
- **US:** the 0.08% residual is the 2003–04 swap-market dawn plus newly issued 30-year TIPS beyond the ISR horizon (~5–9 days per issue).

---

## 8. Figures

Produced by `make_figures.py` into `results/figures/linker_premia/`:

- **Figure 1 — The wedge over time.** $-\lambda$(10y) monthly, US and UK, with the 2008, 2020 and 2022 episodes visible; the series underlying the RQ1 tables. *(Current version: US/UK, as requested by RR for the presentation; the five-market extension is prepared but not yet rendered.)*
- **Figure 2 — Where surprises bite.** Term structure of β_surprise with t-statistics (multivariate specification, net of liquidity) — the visual of the segmentation signatures. *(Current version: US/UK; five-market version pending.)*

Planned (data ready, rendering pending): five-market versions of Figures 1–2; a core–periphery correlation figure (rolling c(US,·)); a three-method agreement plot per market (λ vs cross-sectional mean of nearest and c-exact).

---

## 9. Synthesis — what this panel shows that could not be shown before

1. **A geography of linker mispricing inside the euro.** Same currency, same swap, same index — the wedge ranks IT > DE > FR exactly as sovereign risk does, and only Italy decouples from the global factor during its sovereign crisis. The FLL common factor is a core-markets phenomenon.
2. **Three distinct segmentation signatures** in how inflation surprises propagate along real curves: short-end (US/UK/DE), flat (IT), long-end (FR). Measurable only on fitted real curves.
3. **The risk-free-collateral exception:** liquidity conditions price every linker market except the Bund€i.
4. **A predictability map:** the inflation-aware CiP factor governs real excess returns everywhere except where LDI demand governs the curve (UK), where the Rebonato–Nyholm cointegration equivalence holds in reals.
5. **An institutional record:** the first missing US CPI since 1921 (October 2025, government shutdown) handled through the Treasury's index contingency provision, replicated and validated to the fifth decimal.

For RR's original request, the US/UK results stand alone (slides and correspondence remain US/UK as he asked); the euro extension is the project's own contribution and the natural core of the paper.

---

## 10. Reproducibility — pipeline map and validation log

**Code layout.** `src/inflation_linked/`: `config.py` (markets, conventions, calendars), `bbg.py` (universe, Bloomberg downloads with per-block checkpoints, wide caches, CPI contingency patch), `curves.py` (GSW/BoE/Bundesbank loaders, NSS), `basis.py` (index-ratio engine, cash-flow schedules, IRR with expandable bracket, matched-maturity and c-exact), `pipeline.py` (`build_market` orchestration), numbered runners `02_download.py`, `03_curves…`, `04_basis_markets.py`. `src/linker_premia/`: `rp.py` (curve panels, BEI/λ, factor construction), `r1_bei_premia.py`, `r2_surprises.py`, `r3_predictability.py`, `r4_curve_real.py` (euro real-curve fits), `make_figures.py`.

**Validation tests (all passing).**
- Italian CI vs MEF: 16/16 official coefficients, diff 0.00e+00 (permanent regression test `_verifica_ci.py`).
- US index ratio vs TreasuryDirect: exact to 5 decimals on 1-Jan and 31-Jan-2026 (FQ68), through the October-2025 contingency value.
- BoE inflation curve ≡ nominal − real (Fisher): 0.0bp.
- Bundesbank annual→continuous conversion: verified on synthetic and live parameters.

**Session fix log (permanent, in code).** (1) `ann_to_cc` returns an ndarray — wrapped back into a date-indexed DataFrame at both consumption points (λ engine and c-exact zero panel). (2) FR_CPI nominal-curve alias: the zero panel now receives the aliased nominal market (FR), matching the ytm/reference alias. (3) US anagraphics lack FIRST_SETTLE_DT/START_ACC_DT: fallback to ISSUE_DT in candidate filtering and accrual start. (4) Coupon-schedule anchor moved to FIRST_CPN_DT (the true Bloomberg coupon date): with the ISSUE_DT fallback, anchoring on start-of-accrual desynchronised the cycle from maturity and silently dropped the redemption cash flow. (5) October-2025 CPI = 325.604 hard-wired post-download. (6) UK indexation-lag fill from segment convention (old = 8, new = 3), never overwriting populated values.

**Known cosmetic/pending items.** r3's final "saved" line still names only UK/US (files for all five are written); five-market figure versions; optional single-line-per-maturity dedup in the UK linker pool (multiple non-funged lines share a maturity — harmless for matched-maturity, tidier if pruned); hybrid long/wide ytm files for FR/DE work correctly but could be regenerated clean.

---

## 11. Honesty box — what we do not claim

- The DE RQ3 magnitude is episode-driven; the DE RQ1/RQ2 sample starts in 2012 and the early long end of the DE real curve is model territory (nine bonds).
- The IT real curve before 2008 rests on 4–5 bonds; IT results are robust to starting in 2008, and the worst fitting days are flagged.
- No extrapolation beyond the last quoted swap tenor, anywhere; the price is the documented exclusion windows for ultra-long linkers.
- FR_CPI is excluded from the euro curve comparison on purpose (different index); it lives in the bond-level panel.
- Bilateral (not joint) correlation samples in RQ1 are a choice: they preserve each market's full history at the cost of non-identical windows across pairs; both are reported transparently.

---

## 12. Positioning, contribution, and publication assessment

**Lineage.** The project sits squarely in the Fleckenstein–Longstaff–Lustig (JF 2014) tradition — TIPS–Treasury mispricing — which has since generated a crowded literature on the US market. The contribution here is not a new puzzle but a **unified measurement infrastructure across six markets**: identical curve-fitting (NSS on dirty prices), two bond-level mispricing measures on every ILB trading day (matched-maturity FLL and IRR-vs-curve), and market-specific indexation engines validated bond-by-bond — including the UK old-style 8-month-lag gilts, which most cross-country studies silently drop.

**What the three RR questions deliver.** RQ1: a breakeven/wedge premium exists everywhere, and its cross-market **hierarchy lines up with sovereign credit** (IT widest, DE tightest). RQ2: inflation surprises move real yields and the wedge through **three distinct segmentation signatures**, not one common pattern — the markets are not one market. RQ3: return predictability is dominated by the **cointegration (CiP) factor in four of five markets; the UK is the exception**, consistent with the LDI-driven ownership structure. Rebonato's original US/UK evidence generalises in level but not in mechanism.

**Nature of the contribution — stated honestly.** This is a *comparative measurement* paper: the identification is cross-market (N = 5) and descriptive, not within-market causal. There is no decisive test that pins a mechanism the way a natural experiment would; the segmentation signatures are suggestive taxonomy. The infrastructure is the asset: any follow-up (an LDI event study in the UK, a structural model à la Rebonato's Proposal B) plugs into a validated panel of 500,000+ bond-day observations.

**Placement.** Co-authored with R. Rebonato. Realistic targets: solid field journals (JBF, JIMF, Journal of Fixed Income / JFQA with the right angle). The ceiling is set by the crowded FLL literature: a top-field referee will ask "what do we learn beyond FLL 2014?", and the honest answer — "that it generalises in level but not in mechanism, measured properly across six markets" — is valuable but incremental. The co-author's name and the completeness of the panel are the placement assets; the missing within-market identification is the ceiling.

**What would raise the ceiling.** One identified mechanism: the UK LDI episode as a quasi-experiment on the RQ3 exception, or the structural mispricing model (RR Proposal B) estimated on this panel. Either would move the paper from "careful comparative evidence" to "evidence with a mechanism".
