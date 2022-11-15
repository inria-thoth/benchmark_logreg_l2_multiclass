from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import scipy
    import numpy as np
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from cyanure import estimators


# 'solver': ['catalyst-miso', 'qning-miso', 'qning-ista',  'auto',  'acc-svrg']
class Solver(BaseSolver):
    name = 'cyanure'

    install_cmd = 'conda'
    requirements = ['cyanure']

    _solver_strategy = "debug"

    parameters = {
        'solver': ['catalyst-miso', 'qning-miso',
                   'qning-ista',  'auto',  'acc-svrg']
    }

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        if (scipy.sparse.issparse(self.X) and
                scipy.sparse.isspmatrix_csc(self.X)):
            self.X = scipy.sparse.csr_matrix(self.X)

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.solver_parameter = dict(lambda_1=self.lmbd / X.shape[0],
                                     duality_gap_interval=10000000,
                                     tol=1e-15, verbose=True,
                                     solver=self.solver, max_iter=1000)

        self.solver_instance = estimators.Classifier(loss='logistic',
                                                     penalty='l2',
                                                     fit_intercept=False,
                                                     **self.solver_parameter)

    def run(self, n_iter):
        self.solver_instance.fit(self.X.astype("float64"), self.y)

    def get_result(self):
        return np.squeeze(self.solver_instance.get_weights()).flatten()
