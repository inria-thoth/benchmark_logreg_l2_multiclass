from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import get_cuda_version

cuda_version = get_cuda_version()
if cuda_version is not None:
    cuda_version = cuda_version.split("cuda_", 1)[1][:4]

with safe_import_context() as import_ctx:
    if cuda_version is None:
        raise ImportError("cuml solver needs a nvidia GPU.")

    import cudf
    import numpy as np
    import pandas as pd
    from cuml.linear_model import LogisticRegression
    import dask

class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "rapidsai::rapids",
        f"nvidia::cudatoolkit={cuda_version}",
        "dask-sql",
    ] if cuda_version is not None else []

    parameters = {
        "solver": [
            "qn",
        ],
    }

    support_sparse = False
    parameter_template = "{solver}"

    def set_objective(self, X, y, lmbd, name):

        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        import dask_cudf

        self.X, self.y, self.lmbd = X, y, lmbd
        self.X = cudf.DataFrame.from_pandas(pd.DataFrame(self.X.astype(np.float64)))
        print(self.X.dtypes)
        self.y = cudf.Series(self.y.astype(np.float64))
        

        self.clf = LogisticRegression(
            fit_intercept=False,
            C=1 / self.lmbd,
            penalty="l2",
            tol=1e-15,
            solver=self.solver,
            verbose=3, max_iter=500000, multi_class='multinomial'
        )

    def run(self, n_iter):
        self.clf.solver_model.max_iter = n_iter
        self.clf.fit(self.X, self.y, convert_dtype=True)

    def get_result(self):
        print(np.transpose(self.clf.coef_.to_numpy()))
        return np.transpose(self.clf.coef_.to_numpy())
