import benchopt
from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    import os
    import numpy as np
    from download import download


class Dataset(BaseDataset):
    name = "dino_imagenet"
    is_sparse = False

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    root_url = "http://pascal.inrialpes.fr/data2/cyanure/datasets"
    x_url = root_url + "/feat_IMNET_train.npy"
    y_url = root_url + "/lab_IMNET_train.npy"

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            root_path = os.path.dirname(benchopt.__file__)
            cachedir = root_path + os.path.sep + "cache"
            path_X = download(self.x_url,
                              os.path.join(cachedir, "feat_IMNET_train.npy"))
            path_y = download(self.y_url,
                              os.path.join(cachedir, "lab_IMNET_train.npy"))
            self.X = np.load(os.path.join(path_X), allow_pickle=True)
            self.y = np.load(os.path.join(path_y), allow_pickle=True)
            self.y = np.squeeze(self.y)

        data = dict(X=self.X, y=self.y)
        return data
