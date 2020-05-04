import os
import pickle


class ModelUtil:
    path_name = ''

    @classmethod
    def __init__(cls, dataset_name=None, test_ratio=0.3):
        if not cls.path_name:
            cls.path_name = "model/" + dataset_name + '-' + str(test_ratio) + 'ratio'

    def get_model_path(self, model_name):
        return self.path_name + "-%s" % model_name

    def save(self, model, model_name):
        if not os.path.exists('model'):
            os.mkdir('model')
        pickle.dump(model, open(self.get_model_path(model_name), "wb"))

    def load(self, model_name):
        if not os.path.exists(self.get_model_path(model_name)):
            raise ModelLoadException('%s not found in model dir' % model_name)
        return pickle.load(open(self.get_model_path(model_name), "rb"))


class ModelLoadException(Exception):
    pass
