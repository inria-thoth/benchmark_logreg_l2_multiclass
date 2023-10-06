import benchopt
from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    import os
    import numpy as np
    from download import download


class Dataset(BaseDataset):
    name = "mnist"
    is_sparse = False

    install_cmd = 'conda'
    requirements = ['pip:download']

    root_url = "http://pascal.inrialpes.fr/data2/cyanure/datasets"
    x_url = root_url + "/ckn_mnist.npz"

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            root_path = os.path.dirname(benchopt.__file__)
            cachedir = root_path + os.path.sep + "cache"
            path = download(self.x_url,
                              os.path.join(cachedir, "ckn_mnist.npz"))
            data = np.load(os.path.join(path), allow_pickle=True)
            self.y=data['y']
            self.X=data['X'].astype('float64')
            self.y=np.squeeze(np.float64(self.y))
            
        data = dict(X=self.X, y=self.y, name=self.name)
        return data
