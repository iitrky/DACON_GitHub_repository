"""Simple, documented interface to DACON.

Example
-------
>>> from dacon_api import DACON
>>> model = DACON(hidden=(256, 128), epochs=20, warmup=5)
>>> model.fit(X_train, y_noisy)
>>> model.clean_probability_      # P(label is correct) for every training example
>>> model.repaired_labels_        # repaired training labels
>>> model.predict(X_test)         # class predictions of the trained network
>>> model.audit_report(top=20)    # examples most likely to be mislabelled
"""
import numpy as np
import dacon_core as core


class DACON:
    """Dependence aware evidence fusion for auditing noisy labels.

    Parameters
    ----------
    hidden : tuple of int
        Hidden layer sizes of the multilayer perceptron.
    epochs, warmup : int
        Total training epochs and cross entropy warm-up epochs.
    batch_size : int
    k : int
        Number of nearest neighbours for the spatial evidence.
    dependence_aware : bool
        True gives DACON; False gives the independence based variant (DyNeC).
    seed : int
    """

    def __init__(self, hidden=(256, 128), epochs=20, warmup=5, batch_size=128, k=10,
                 dependence_aware=True, seed=0):
        self.hidden, self.epochs, self.warmup = tuple(hidden), epochs, warmup
        self.batch_size, self.k, self.dep, self.seed = batch_size, k, dependence_aware, seed

    def fit(self, X, y_noisy, X_val=None, y_val=None, y_true=None):
        X = np.asarray(X, np.float32); y_noisy = np.asarray(y_noisy).astype(int)
        K = int(y_noisy.max() + 1); self.n_classes_ = K
        Xv = X if X_val is None else np.asarray(X_val, np.float32)
        yv = y_noisy if y_val is None else np.asarray(y_val).astype(int)
        yt = y_noisy if y_true is None else np.asarray(y_true).astype(int)
        sizes = [X.shape[1], *self.hidden, K]
        flags = {'dep': True} if self.dep else None
        acc, aux = core.run((X, y_noisy, Xv, yv, yt), 'dynec', self.epochs, self.warmup, self.batch_size,
                            sizes, self.seed, K, k=self.k, flags=flags, log_auc=y_true is not None, ret_labels=True)
        self.net_ = aux.pop('model')
        self.clean_probability_ = np.array(aux.pop('w'))
        self.repaired_labels_ = np.array(aux.pop('labels'))
        self.given_labels_ = y_noisy
        self.validation_accuracy_ = acc
        self.diagnostics_ = {k: v for k, v in aux.items() if k != 'sel'}
        return self

    def predict_proba(self, X):
        return self.net_.predict(np.asarray(X, np.float32))[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def audit_report(self, top=20):
        """Indices, given labels, suggested labels and clean probabilities of the most suspect examples."""
        order = np.argsort(self.clean_probability_)[:top]
        return [dict(index=int(i), given=int(self.given_labels_[i]), suggested=int(self.repaired_labels_[i]),
                     p_clean=float(self.clean_probability_[i])) for i in order]
