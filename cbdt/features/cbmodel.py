import joblib
import numpy as np
import sklearn.metrics as skm
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def normalized_mean_squared_error(truth, predictions):
    norm = skm.mean_squared_error(truth, np.full(len(truth), np.mean(truth)))
    return skm.mean_squared_error(truth, predictions) / norm


class ClickbaitModel(object):
    __regression_measures = {
        'Explained variance': skm.explained_variance_score,
        'Mean absolute error': skm.mean_absolute_error,
        'Mean squared error': skm.mean_squared_error,
        'Median absolute error': skm.median_absolute_error,
        'R2 score': skm.r2_score,
        'Normalized mean squared error': normalized_mean_squared_error,
    }

    __classification_measures = {'Accuracy': skm.accuracy_score,
                                 'Precision': skm.precision_score,
                                 'Recall': skm.recall_score,
                                 'F1 score': skm.f1_score
                                 }

    def __init__(self):
        self.models = {"LogisticRegression": LogisticRegression(),
                       "MultinomialNB": MultinomialNB(),
                       "RandomForestClassifier": RandomForestClassifier(),
                       "SVR_linear": svm.SVR(kernel='linear'),
                       "SVR": svm.SVR(),
                       "Ridge": Ridge(alpha=1.0, solver="auto"),
                       "Lasso": Lasso(),
                       "ElasticNet": ElasticNet(),
                       "SGDRegressor": SGDRegressor(),
                       "RandomForestRegressor": RandomForestRegressor(),
                       "RidgeClassifier": RidgeClassifier(alpha=1.0, solver="auto")
                       }
        self.model_trained = None

    def classify(self, x, y, model, evaluate=True):
        if isinstance(model, str):
            self.model_trained = self.models[model]
        else:
            self.model_trained = model
        if evaluate:
            x_train, x_test, y_train, y_test = train_test_split(x, y.T, random_state=42)
        else:
            x_train = x
            y_train = y

        self.model_trained.fit(x_train, y_train)
        if evaluate:
            pred = self.model_trained.predict(x_test)
            self.eval_classify(y_test, pred)

    def predict(self, x):
        return self.model_trained.predict(x)

    def eval_classify(self, y_test, y_predicted):
        for cm in self.__classification_measures:
            print("{} & {} \\\\".format(cm, round(self.__classification_measures[cm](y_test, y_predicted), 3)))

    def save(self, filename):
        joblib.dump(self.model_trained, filename)

    def load(self, filename):
        self.model_trained = joblib.load(filename)
