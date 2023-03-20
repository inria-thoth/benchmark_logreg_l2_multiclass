import warnings


from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    import xgboost as xgb


class Solver(BaseSolver):
    name = 'xgboost'

    install_cmd = 'conda'
    requirements = ['xgboost']

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        self.xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)

    def run(self, n_iter):
        self.xgb_model.max_iter = n_iter
        self.xgb_model.fit(self.X, self.y)

    def get_result(self):
        return np.transpose(self.clf.coef_)
