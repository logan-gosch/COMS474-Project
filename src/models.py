from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from config import RANDOM_STATE

def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }

def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "SVM (RBF)": SVC(kernel="rbf"),
    }