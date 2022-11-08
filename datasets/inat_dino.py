import benchopt
from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    import os
    import numpy as np
    from download import download


class Dataset(BaseDataset):
    name = "dino_inat"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            cachedir = os.path.dirname(benchopt.__file__) + os.path.sep + "cache"
            path_X = download("http://pascal.inrialpes.fr/data2/mairal/data/feat_INAT_train.npy", os.path.join(cachedir, "feat_INAT_train.npy"))
            path_y = download("http://pascal.inrialpes.fr/data2/mairal/data/lab_INAT_train.npy", os.path.join(cachedir, "lab_INAT_train.npy"))
            self.X = np.load(os.path.join(path_X), allow_pickle=True)
            self.y = np.load(os.path.join(path_y), allow_pickle=True)
            self.y=np.squeeze(self.y)

        data = dict(X=self.X, y=self.y)
        return data
