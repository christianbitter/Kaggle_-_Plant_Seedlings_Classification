import os
from .Params import Params


class Model(object):
    def __init__(self, params_fp, verbose=False):
        print("Model: {0}".format(verbose))
        self.params_fp = params_fp
        self.verbose = verbose
        self.model_name = "Generic-Model"
        self.model_type = "Generic-Model_Type"

        if self.params_fp and os.path.exists(self.params_fp):
            # now update where possible
            if self.verbose:
                print("_load_params - loading parameters from json")
            self.params = Params.create(json_file_path=self.params_fp)
            self.model_name = self._load_checked_param("model_name")
            self.model_type = self._load_checked_param("model_type")
        else:
            self.params = Params()

    def _load_checked_param(self, param_name: str):
        if param_name not in self.params:
            raise ValueError("_load_checked_param - {0} not present".format(param_name))

        return self.params[param_name]


    def train(self):
        raise ValueError("Model - train not implemented")

    def build(self):
        raise ValueError("build - not implemented")

    def plot(self, what: str=None):
        raise ValueError("Model.plot - not implemented")

    def __str__(self):
        return "{0}\{1}".format(self.model_type, self.model_name)

    #TODO: implement other methods