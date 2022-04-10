import numpy as np
from cbdt.features.feature import Feature
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split


class FeatureBuilder(object):

    def __init__(self, *args):
        self.features = list(args)
        self.build_features = None
        self.build_features_split = None
        self.feature_names = None

    def add_feature(self, feature, data_field_name):
        self.features.append((feature, data_field_name))
        return self

    def build(self, data, save=False):
        _result = None
        self.feature_names = []

        def push(result, f):
            if result is None:
                result = f
            else:
                result = hstack((_result, f))
            return result

        for f in self.features:
            if isinstance(f[0], Feature):
                _result = push(_result, f[0].assparse(data.get_x(f[1])))
                self.feature_names += f[0].name
        try:
            self.build_features_split = train_test_split(_result, np.asarray(data.get_y()).T, random_state=42)
        except Exception:
            self.build_features_split = None
        print(_result)
        self.build_features = _result
