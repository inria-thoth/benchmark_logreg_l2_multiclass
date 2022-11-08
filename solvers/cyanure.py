import warnings
from sklearn.exceptions import ConvergenceWarning
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import sys
    import scipy
    import numpy as np
    from cyanure import estimators
    from cyanure.data_processing import preprocess

# 'solver': ['catalyst-miso', 'qning-miso', 'qning-ista',  'auto',  'acc-svrg']
class Solver(BaseSolver):
    name = 'cyanure_norm'

    install_cmd = 'conda'
    requirements = ['cyanure']

    parameters = {
        'solver': ['catalyst-miso', 'qning-miso', 'qning-ista',  'auto',  'acc-svrg']
    }
    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        if (scipy.sparse.issparse(self.X) and
                scipy.sparse.isspmatrix_csc(self.X)):
            self.X = scipy.sparse.csr_matrix(self.X)

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.solver_parameter = dict(
        lambda_1= self.lmbd / X.shape[0], duality_gap_interval=10000000,
        tol=1e-15, verbose=True, solver=self.solver, max_iter=1000
        )

        self.solver_instance = estimators.Classifier(loss='logistic', penalty='l2',
                                       fit_intercept=False,
                        **self.solver_parameter)

        self.dataset = "New dataset"
       
    def compute_relative_optimality_gap(self):
        min_eval=100
        max_dual=-100
        self.solver_instance.optimization_info_ = np.squeeze(self.solver_instance.optimization_info_)
        if len(self.solver_instance.optimization_info_.shape) > 1 :
            min_eval=min(min_eval,np.min(self.solver_instance.optimization_info_[1,]))
            max_dual=max(max_dual,np.max(self.solver_instance.optimization_info_[2,]))
            info = np.array(np.maximum((self.solver_instance.optimization_info_[1,]-max_dual)/min_eval,1e-9))
        
        else:
            min_eval=min(min_eval,np.min(self.solver_instance.optimization_info_[1]))
            max_dual=max(max_dual,np.max(self.solver_instance.optimization_info_[2]))
            info = np.array(np.maximum((self.solver_instance.optimization_info_[1]-max_dual)/min_eval,1e-9))

        return info

    def run(self, n_iter):
        self.solver_instance.max_iter = n_iter
        self.solver_instance.fit(self.X, self.y)

    def get_result(self):
        return np.squeeze(self.solver_instance.get_weights())