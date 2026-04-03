from data_loader import load_data, clean_data
from preprocessing import build_features, make_preprocessor
from models import get_regression_models, get_classification_models
from evaluate import run_regression, run_classification, save_results

def main():
    df = load_data()
    df = clean_data(df)

    print("Dataset shape after cleaning:", df.shape)

    X, y_reg, y_cls, feature_cols = build_features(df)
    preprocessor = make_preprocessor(feature_cols)

    reg_models = get_regression_models()
    cls_models = get_classification_models()

    print("\nRunning regression models...")
    reg_results = run_regression(X, y_reg, feature_cols, reg_models, preprocessor)
    print(reg_results)

    print("\nRunning classification models...")
    cls_results, cls_reports = run_classification(
        X, y_cls, feature_cols, cls_models, preprocessor
    )
    print(cls_results)

    save_results(reg_results, cls_results, cls_reports)

if __name__ == "__main__":
    main()