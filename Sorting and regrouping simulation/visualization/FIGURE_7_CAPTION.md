**Fig. 7 | Uncertainty-aware sorting and regrouping for second-life battery pack assembly.**

**a,** Simulation workflow comprising bootstrap sampling of held-out test-cell records from the LFP dataset, cell sorting and simulated regrouping into eight-cell series-connected packs. Five strategies are compared: capacity-estimate-only sorting (S1), conservative sorting (S2), lower-bound sorting (S3), uncertainty-guided sorting (S4) and uncertainty-guided sorting with targeted RPT (S5).

**b,** Estimated capacity versus uncertainty, quantified by capacity interval width, for a representative 300-record batch under S5. Panels **b–e** show this example at a pool usage ratio of 42.86% (1,000 training cells) and a tail segment of 2.5–2.75 V. The vertical and horizontal lines indicate the 14 Ah threshold and batch median uncertainty, respectively. Shading identifies the high-capacity/high-uncertainty region. Within this region, the 20% of records with the widest intervals are rejected by S4 or selected for targeted RPT by S5 (diamonds). Grey and teal circles denote capacity-based rejection and retention without targeted RPT, respectively.

**c,** Sorting and targeted RPT outcomes for the same batch.

**d,** Measured cell capacities within 11 assembled packs. Rows denote packs and columns denote cell positions; outlined entries identify records subjected to targeted RPT under S5.

**e,** Corresponding pack capacities, defined as the minimum measured capacity among the eight constituent cells. Pack numbers match the rows in **d**; the horizontal dashed line marks the 14 Ah qualification threshold.

**f,** Mean sorting quality and residual risk across 3,000 bootstrap batches per sorting strategy. Panels **f–i** aggregate three random trials, each comprising 1,000 bootstrap batches, at the same setting as **b–e**.

**g,** Pack pass rate versus effective yield, showing mean results across the three random trials.

**h,i,** Distributions of batch-wise mean pack capacity (**h**) and mean within-pack capacity standard deviation (SD; **i**). Shaded densities represent 3,000 batches per strategy; horizontal lines span the 2.5th–97.5th percentiles and open circles show overall means. Filled circles, squares and triangles show the respective means of the three random trials.

**j,k,** Mean paired differences (S5 − S4) in effective yield (**j**) and pack pass rate (**k**), expressed in percentage points (pp), across all pool usage ratios (14.29%, 42.86% and 71.43%) and tail segment settings.

**l,** Corresponding mean number of records selected for targeted RPT per 300-record batch.


RPT is simulated by revealing existing measured labels only for selected records; no new physical pack tests are represented. Central 95% batch ranges are not population mean confidence intervals.
