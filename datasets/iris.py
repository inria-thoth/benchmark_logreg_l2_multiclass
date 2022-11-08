from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn import datasets


class Dataset(BaseDataset):
    name = "covtype_binary"

    install_cmd = 'conda'
    requirements = ['pip:scikit-learn']

    def get_data(self):

        iris = datasets.load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target
        return dict(X=X, y=y)