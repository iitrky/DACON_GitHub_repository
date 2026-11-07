# DACON — Dependence Aware Evidence Fusion for Auditing Noisy Labels

Complete implementation, raw results and figures for the paper

> G. Arul Dalton and P. Sankar. *Dependence Aware Evidence Fusion for Auditing Noisy Labels.* Submitted to *Machine Learning* (Springer).

DACON estimates, for every training example, the probability that its label is correct, and repairs suspect labels while the network trains. It combines two views of each example:

* **Temporal evidence** – the mean prediction margin of the example along the training trajectory.
* **Spatial evidence** – the agreement of its label with its *k* nearest neighbours in the learned representation, where each neighbour's vote is weighted by its own estimated reliability.

Both views are modelled per class by two-component Gaussian mixtures and fused in log-odds space. The new element is **dependence aware fusion**: a bivariate mixture estimates the within-component correlation *r* of the two views, and the spatial log-odds are shrunk by λ = 1 − *r* (floor 0.2). Under independence this reduces to a product of experts; under instance-dependent noise, where neighbours share the same wrong label, it stops the two views from double counting the same evidence. The fused posterior drives a soft relabelling of suspect examples.

Everything runs in NumPy and scikit-learn on a single CPU core. No GPU or deep learning framework is needed.

## Quick start (about two minutes)

```
pip install -r requirements.txt
python demo.py
```

`demo.py` uses UCI Digits (bundled with scikit-learn) with 40% pair-flip noise. Expected output: cross-entropy test accuracy ≈ 56%, DACON ≈ 90%, AUROC of the clean probability ≈ 0.91, and about 82% of the wrong labels repaired to the true class.

## Using DACON on your own data

```python
from dacon_api import DACON
model = DACON(hidden=(256, 128), epochs=20, warmup=5).fit(X_train, y_noisy)
model.clean_probability_   # P(label is correct) for every training example
model.repaired_labels_     # repaired training labels
model.audit_report(top=20) # most suspect examples with suggested labels
model.predict(X_test)      # predictions of the trained network
```

Set `dependence_aware=False` to obtain the independence-based variant (DyNeC).

## Reproducing the paper

```
sh download_data.sh        # MNIST, Fashion-MNIST, Electricity, Weather, Covertype -> ./data
python run_main.py         # 13 methods x 6 datasets x 4 noise settings x 3 seeds = 936 runs
python run_ablation.py     # 105 ablation runs
python analyze.py          # average ranks, Wilcoxon tests, audit metrics, Fig3
python make_block_diagrams.py   # Fig1 and Fig2
```

Both run scripts are resumable: they skip runs already stored in `results.jsonl` / `ablation.jsonl`. The released result files already contain every run reported in the paper, so `python analyze.py` alone reproduces all tables and figures.

## Files

| File | Content |
|---|---|
| `dacon_core.py` | NumPy MLP with Adam; CE, GCE, SCE, APL; small-loss GMM selection; DyNeC and DACON (`run(..., 'dynec', flags={'dep': True})`); data loaders and noise models (symmetric, pair flip, instance dependent) |
| `benchmarks.py` | Tabular loaders; Co-teaching, JoCoR, simplified DivideMix, SELC, confident learning, ENN; experiment configurations |
| `dacon_api.py` | Scikit-learn style `DACON` class |
| `demo.py` | Two-minute demonstration |
| `run_main.py`, `run_ablation.py` | Full experiment grid and ablation study |
| `analyze.py` | Ranks, significance tests, label-audit statistics, Fig3 |
| `make_block_diagrams.py` | Fig1 (system model) and Fig2 (DACON module) |
| `results.jsonl`, `ablation.jsonl`, `summary.json` | Raw results of all 1,041 runs and the aggregated summary |
| `Fig1`–`Fig3` (`.png`, `.eps`) | Figures of the paper |
| `download_data.sh` | Downloads the benchmark data |

## Main results (test accuracy, three seeds)

* Image benchmarks (UCI Digits, MNIST, Fashion-MNIST; 12 settings): DACON has the best average rank among 13 methods (2.08), ahead of DyNeC (2.46) and small-loss selection (4.71), and is significantly better than every compared method (Wilcoxon, p ≤ 0.041).
* Tabular benchmarks (Electricity, Weather, Covertype): DACON does not outperform simple robust losses (average rank 7.62 against 4.88 for SCE); see Section 6.4 of the paper.

## Notes on the baselines

All baselines are reimplemented from their papers in one code base with the same networks and schedule. Co-teaching and JoCoR receive the true noise rate as their forget rate. DivideMix is simplified (no data augmentation; soft cross-entropy for the unlabelled part).

## Licence

MIT. If you use this code, please cite the paper (see `CITATION.cff`).
