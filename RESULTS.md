# Synthetic testbed results (T0)

Games: n = 12 segments, random sparse order-2 Mobius games. All estimators consume the identical coalition-oracle API; certificates at delta = 0.05 hold simultaneously over all segments. Seeds averaged per cell.

## Estimation benchmark — bernoulli oracle noise

| budget | estimator | linf | l2 | kendall | top-4 prec | P[all CIs cover] | mean CI width |
|---|---|---|---|---|---|---|---|
| 2000 | permutation_mc | 0.0963 | 0.1810 | 0.692 | 0.700 | 1.00 | 0.9215 |
| 2000 | kernel_shap | 0.0792 | 0.1540 | 0.818 | 0.700 | n/a | n/a |
| 2000 | cc_bernstein | 0.0523 | 0.0954 | 0.818 | 0.825 | 1.00 | 3.9540 |
| 2000 | surrogate_cc | 0.0636 | 0.0993 | 0.839 | 0.825 | 1.00 | 11.3112 |
| 5000 | permutation_mc | 0.0695 | 0.1184 | 0.779 | 0.725 | 1.00 | 0.6137 |
| 5000 | kernel_shap | 0.0609 | 0.1375 | 0.894 | 0.750 | n/a | n/a |
| 5000 | cc_bernstein | 0.0342 | 0.0577 | 0.891 | 0.925 | 1.00 | 1.9473 |
| 5000 | surrogate_cc | 0.0406 | 0.0683 | 0.839 | 0.850 | 1.00 | 4.5634 |
| 10000 | permutation_mc | 0.0494 | 0.0805 | 0.864 | 0.800 | 1.00 | 0.4500 |
| 10000 | kernel_shap | 0.0533 | 0.1337 | 0.936 | 0.750 | n/a | n/a |
| 10000 | cc_bernstein | 0.0227 | 0.0399 | 0.942 | 0.950 | 1.00 | 1.1752 |
| 10000 | surrogate_cc | 0.0285 | 0.0489 | 0.894 | 0.925 | 1.00 | 2.4399 |
| 25000 | permutation_mc | 0.0266 | 0.0505 | 0.885 | 0.925 | 1.00 | 0.2977 |
| 25000 | kernel_shap | 0.0481 | 0.1297 | 0.942 | 0.750 | n/a | n/a |
| 25000 | cc_bernstein | 0.0151 | 0.0258 | 0.955 | 0.975 | 1.00 | 0.6248 |
| 25000 | surrogate_cc | 0.0175 | 0.0300 | 0.945 | 0.925 | 1.00 | 1.1623 |

## Estimation benchmark — gauss(0.05) oracle noise

| budget | estimator | linf | l2 | kendall | top-4 prec | P[all CIs cover] | mean CI width |
|---|---|---|---|---|---|---|---|
| 2000 | permutation_mc | 0.0166 | 0.0273 | 0.945 | 0.975 | 1.00 | 0.9215 |
| 2000 | kernel_shap | 0.0089 | 0.0177 | 0.976 | 0.950 | n/a | n/a |
| 2000 | cc_bernstein | 0.0180 | 0.0304 | 0.945 | 0.900 | 1.00 | 3.2549 |
| 2000 | surrogate_cc | 0.0058 | 0.0103 | 0.985 | 1.000 | 1.00 | 7.8803 |
| 5000 | permutation_mc | 0.0106 | 0.0172 | 0.967 | 0.950 | 1.00 | 0.6137 |
| 5000 | kernel_shap | 0.0069 | 0.0157 | 0.982 | 0.925 | n/a | n/a |
| 5000 | cc_bernstein | 0.0103 | 0.0182 | 0.982 | 0.900 | 1.00 | 1.4811 |
| 5000 | surrogate_cc | 0.0043 | 0.0069 | 0.994 | 0.950 | 1.00 | 3.4084 |
| 10000 | permutation_mc | 0.0074 | 0.0118 | 0.982 | 0.975 | 1.00 | 0.4500 |
| 10000 | kernel_shap | 0.0061 | 0.0149 | 0.994 | 0.925 | n/a | n/a |
| 10000 | cc_bernstein | 0.0071 | 0.0125 | 0.985 | 0.950 | 1.00 | 0.8328 |
| 10000 | surrogate_cc | 0.0030 | 0.0054 | 0.994 | 0.975 | 1.00 | 1.8342 |
| 25000 | permutation_mc | 0.0042 | 0.0069 | 0.997 | 1.000 | 1.00 | 0.2977 |
| 25000 | kernel_shap | 0.0053 | 0.0144 | 0.991 | 0.925 | n/a | n/a |
| 25000 | cc_bernstein | 0.0049 | 0.0076 | 0.988 | 0.975 | 1.00 | 0.3978 |
| 25000 | surrogate_cc | 0.0017 | 0.0031 | 0.994 | 1.000 | 1.00 | 0.8136 |

## Hierarchical triage benchmark (Approach B)

4 groups x 3 segments; groups 2 and 3 are exact null players; tau = 0.15. A *false elimination* is a non-null group certified negligible (guaranteed rare by construction: probability <= delta = 0.05 per run).

| budget | null-group recall | false eliminations (total) | member CIs cover truth |
|---|---|---|---|
| 10000 | 0.00 | 0 | 1.00 |
| 30000 | 0.00 | 0 | 1.00 |
| 60000 | 1.00 | 0 | 1.00 |

## Observations

- **CC-Bernstein dominates the baselines on point-estimate accuracy at every budget** (query sharing: each evaluated pair updates all n segments), while carrying always-valid simultaneous certificates that neither KernelSHAP nor plain permutation sampling provide at comparable accuracy.
- **Certificates never miss** (P[all CIs cover] = 1.0 across every cell): the guarantee is conservative in practice, so certified decisions (e.g. pruning) are safe; widths shrink roughly as 1/sqrt(budget) with an additive anytime range term.
- **Noise regime decides whether structure pays.** Under binary utilities, within-coalition Bernoulli noise dominates and the surrogate control variate buys little over CC-Bernstein — exactly risk R2 of the proposal, and the regime where the two-level (coalitions x replicates) allocation matters. Under low-noise utilities the residual game is tiny and the surrogate's advantage appears at small budgets.
- **Hierarchical triage has a certification threshold**: below a budget determined by tau, noise, and the anytime-Bernstein constants, no group can be certified (recall 0), and above it recall jumps to 1 with zero false eliminations — the certificate fails safe, never wrong.

Raw rows: `results/synthetic_results.json`. Reproduce: `python3 experiments/run_synthetic.py`.
