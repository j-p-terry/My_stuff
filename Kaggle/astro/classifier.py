
"""Classifies objects"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

directory_path = "/Users/jasonterry/Documents/Scripts/Misc/My_stuff/" \
                 "Kaggle/astro"

metadata_columns = ["object_id", "ra", "decl", "gal_l", "gal_b"	"ddf",
                    "hostgal_specz", "hostgal_photoz", "hostgal_photoz_err",
                    "distmod", "mwebv", "target"]

data_columns = ["object_id", "mjd", "passband", "flux", "flux_err",
                "detected"]


train_data_path = directory_path + "/data/train_galaxy_merged.csv"
test_data_path = directory_path + "/data/test_galaxy_merged.csv"


def k_nearest_n(k=10):

    """Makes a k-nearest neighbor classifier"""

    classifier = KNeighborsClassifier(k)
    return classifier


def random_forest():

    """Makes a random forest classifier"""

    classifier = RandomForestClassifier()
    return classifier


def ada_boost():

    """Makes Ada Boost classifier"""

    classifier = AdaBoostClassifier()
    return classifier


def mlp():

    """Makes MLP classifier"""

    classifier = MLPClassifier()
    return classifier


def sgd_class():

    """Makes SGD classifier"""

    classifier = SGDClassifier()
    return classifier


def grad_boost():

    """Makes gradient boostin classifier"""

    classifier = GradientBoostingClassifier()
    return classifier


def d_tree(depth=10):

    """Makes decision tree classifier"""

    classifier = DecisionTreeClassifier(depth)
    return classifier


def classify(knn=True, rf=False, ada=False, mlp=False, sgd=False,
             grad_b=False, tree=False, k=10, depth=10):

    """Classifies objects with selected models"""

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    columns = list(train)

    X = pd.DataFrame()
    for column in columns:
        if column != "target":
            X[column] = train[column]

    y = train["target"]

    if knn:
        classifier = k_nearest_n(k)
        fit = classifier.fit(X, y)
        r2 = fit.score(X, y)
        r2 = r2.__round__(4)
        print("KNN, k = " + str(k) + 'R^2 = ' + str(r2))
        classifications = fit.predict(test)

        results = pd.DataFrame()
        results["object_id"] = test["object_id"]
        results["classes"] = classifications

        results.to_csv(directory_path + "/data/knn_class_k" + str(k)
                       + ".csv")
    if rf:
        classifier = random_forest()
        fit = classifier.fit(X, y)
        r2 = fit.score(X, y)
        r2 = r2.__round__(4)
        print("Random forst 'R^2 = " + str(r2))
        classifications=fit.predict(test)

        results=pd.DataFrame()
        results["object_id"]=test["object_id"]
        results["classes"]=classifications

        results.to_csv(directory_path + "/data/rand_forest_class.csv")

    if ada:
        classifier=ada_boost()
        fit=classifier.fit(X, y)
        r2=fit.score(X, y)
        r2=r2.__round__(4)
        print("Ada R^2 = " + str(r2))
        classifications=fit.predict(test)

        results=pd.DataFrame()
        results["object_id"]=test["object_id"]
        results["classes"]=classifications

        results.to_csv(directory_path + "/data/ada_class.csv")

    if mlp:
        classifier=mlp(k)
        fit=classifier.fit(X, y)
        r2=fit.score(X, y)
        r2=r2.__round__(4)
        print("MLP 'R^2 = " + str(r2))
        classifications=fit.predict(test)

        results=pd.DataFrame()
        results["object_id"]=test["object_id"]
        results["classes"]=classifications

        results.to_csv(directory_path + "/data/mlp_class.csv")

    if sgd:
        classifier=sgd_class(k)
        fit=classifier.fit(X, y)
        r2=fit.score(X, y)
        r2=r2.__round__(4)
        print("SGD 'R^2 = " + str(r2))
        classifications=fit.predict(test)

        results=pd.DataFrame()
        results["object_id"]=test["object_id"]
        results["classes"]=classifications

        results.to_csv(directory_path + "/data/sgd_class.csv")

    if grad_b:
        classifier=grad_boost(k)
        fit=classifier.fit(X, y)
        r2=fit.score(X, y)
        r2=r2.__round__(4)
        print("grad b'R^2 = " + str(r2))
        classifications=fit.predict(test)

        results=pd.DataFrame()
        results["object_id"]=test["object_id"]
        results["classes"]=classifications

        results.to_csv(directory_path + "/data/grad_b_class.csv")

    if tree:
        classifier=d_tree(depth)
        fit=classifier.fit(X, y)
        r2=fit.score(X, y)
        r2=r2.__round__(4)
        print("Decision tree, depth = " + str(depth) + 'R^2 = ' + str(r2))
        classifications=fit.predict(test)

        results=pd.DataFrame()
        results["object_id"]=test["object_id"]
        results["classes"]=classifications

        results.to_csv(directory_path + "/data/d_tree_class_d" + str(depth)
                       + ".csv")
