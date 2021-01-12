from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import random


class Model:
    def __init__(self, feature_matrix, labels, folds, cfg):

        self.X = feature_matrix
        self.encoder = LabelEncoder()
        self.y = self.encoder.fit_transform(labels)
        self.folds = folds
        self.cfg = cfg
        
        self.predicted_sounds = []
        self.actual_sounds = []
        self.val_fold_scores_ = []

    def train_kfold(self):

        logo = LeaveOneGroupOut()
        for train_index, test_index in logo.split(self.X, self.y, self.folds):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            ss = StandardScaler(copy=True)
            X_train = ss.fit_transform(X_train)
            X_test = ss.transform(X_test)

            clf = self.cfg["model"]
            clf = clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            fold_acc = accuracy_score(y_test, y_pred)
            self.actual_sounds.append(y_test)
            self.predicted_sounds.append(y_pred)
            self.val_fold_scores_.append(fold_acc)

        return clf, self.val_fold_scores_, self.predicted_sounds, self.actual_sounds 