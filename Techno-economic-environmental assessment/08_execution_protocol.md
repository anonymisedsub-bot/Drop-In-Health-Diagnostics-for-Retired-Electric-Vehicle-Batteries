# Targeted RPT deployment protocol

All deployment cells, (1-p)N, first undergo common de-energisation from initial
SOC s to zero, and obtain predictions from that record. A fraction q of this
deployment pool is selected for additional capacity measurement according to
high uncertainty and high estimated capacity. The target count is q(1-p)N.

Each additional RPT starts at SOC zero: full CCCV charge 1.5 h, rest 10 min,
complete discharge r h. Stored charging energy is E_cell and grid charging
energy is E_cell/eta. End SOC is zero. The common initial discharge is
subtracted once from the protocol burden.

Sensitivity scans retention r, initial SOC s and training-label fraction p,
with q fixed at 0.04. Monte Carlo uses q~Tri(0.03,0.04,0.05), independently of
the 15 shared inputs. The shared inputs use seed 20260921; q uses NumPy
SeedSequence([20260921,20260926]). Uniform-distribution checks use identical
bounds. Each seed is summarised separately. Simulation summaries have no
convergence-based acceptance threshold.

Training-label proportion p and deployment retest fraction q have different
denominators. At p=0.15 and q=0.04, 3.4% of the whole batch receives extra
deployment RPT. Additional measurements receive no automatic training credit.

Charge efficiency affects grid energy, not time. The common battery model is
E_cell=Vnom*Qnom*r. Arrival charge time is (1-s)*1.5 h, baseline discharge is
s*r h, and complete discharge is r h. Pulse testing uses five 0.1 h
conditioning segments; EIS uses five 0.15 h segments; partial-charge testing
uses 15 min charge plus 15 min extra discharge. These are fixed-time
approximations. Rest occupies the tester.

Targeted cells use the same representative retention r as the batch; the model
does not include a subgroup-specific capacity/uncertainty joint distribution.
This is an economic protocol scenario, not execution of a sorting classifier
or proof that the selected fraction identifies all high-risk cells. No 14 Ah
threshold is imposed on an equivalent 2.6 Ah cell. Targeted selection accuracy,
extra handling, instrument auxiliary power, training/inference energy, EIS
capital premium and discharge-energy recovery are outside the core model.

The registered SOC range satisfies s<=0.6 and the partial-charge condition
s+0.25<=1. For a protocol with infeasible comparisons, retain draws and
unaffected comparators, mark the infeasible comparisons unavailable, and
report coverage.
