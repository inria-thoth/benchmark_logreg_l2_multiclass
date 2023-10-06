from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from lightning.classification import CDClassifier
    import numpy as np


class Solver(BaseSolver):
    name = 'Lightning'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/scikit-learn-contrib/lightning.git'
    ]

    def set_objective(self, X, y, lmbd, name):

        self.X, self.y, self.lmbd = X, y, lmbd

        self.clf = CDClassifier(
            loss='log', penalty='l2', C=1, alpha=self.lmbd,
            tol=0, permute=False, shrinking=False, warm_start=False)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def skip(self, X, y, lmbd, name):
        if name == "dino_places":
            return True, "Seg fault !"
        return False, None

    def get_result(self):
        return np.transpose(self.clf.coef_)