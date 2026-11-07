"""Two-minute demonstration on UCI Digits (bundled with scikit-learn, no download needed).

Injects 40% pair-flip label noise, trains plain cross entropy and DACON,
and reports test accuracy and how well DACON finds and repairs the wrong labels.
Run:  python demo.py
"""
import numpy as np
import dacon_core as core
from dacon_api import DACON
from sklearn.metrics import roc_auc_score

Xtr, ytr, Xte, yte = core.load('digits')
y_noisy = core.corrupt(ytr, 10, 'asym', 0.4, seed=0)
wrong = y_noisy != ytr
print(f'training examples: {len(ytr)}, corrupted labels: {wrong.mean():.1%}')

acc_ce, _ = core.run((Xtr, y_noisy, Xte, yte, ytr), 'ce', 60, 10, 64, [64, 128, 64, 10], 0, 10)
print(f'cross entropy test accuracy: {acc_ce:.1%}')

m = DACON(hidden=(128, 64), epochs=60, warmup=10, batch_size=64).fit(Xtr, y_noisy, Xte, yte, y_true=ytr)
print(f'DACON test accuracy:         {m.validation_accuracy_:.1%}')
print(f'AUROC of P(clean) for finding wrong labels: {roc_auc_score(~wrong, m.clean_probability_):.3f}')
print(f'wrong labels repaired to the true class:    {(m.repaired_labels_[wrong] == ytr[wrong]).mean():.1%}')
print('five most suspect training labels:')
for r in m.audit_report(5):
    print(f"  #{r['index']}: given {r['given']}, suggested {r['suggested']}, true {ytr[r['index']]}, P(clean)={r['p_clean']:.3f}")
