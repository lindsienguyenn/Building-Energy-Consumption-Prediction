"""
Building Energy Consumption Prediction
WiDS Datathon 2022 - Predicting site_eui (site energy use intensity)

This script compares multiple regression models and uses hyperparameter tuning
to find the best model for predicting building energy consumption.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import xgboost as xgb
import lightgbm as lgb

# Configuration

# Data paths - update these to match your local setup
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# Target column
TARGET_COL = "site_eui"

# Data Loading and Preprocessing Functions
def read_data(train_path: str, test_path: str):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def preprocess_data(train_df, test_df, target_col):
    # Drop ID columns
    test_ids = test_df['id']
    train_df = train_df.drop(columns=['id'])
    test_df = test_df.drop(columns=['id'])

    # Split features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    test_df[num_cols] = num_imputer.transform(test_df[num_cols])

    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

    # Encode categorical variables
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le

    # Scale numeric features
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    return X, y, test_df, test_ids, num_cols, cat_cols


# Model Training and Evaluation Functions
def evaluate_models(X_train, y_train, X_val, y_val):
    models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=204,
            learning_rate=0.15091256567397734,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=3
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel='rbf', C=4, gamma=0.5, epsilon=0.1)
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        results[name] = {"model": model, "rmse": rmse, "y_pred": y_pred}
        print(f"{name} Validation RMSE: {rmse:.4f}")

    # Select best model
    best_model_name = min(results, key=lambda x: results[x]["rmse"])
    best_model = results[best_model_name]["model"]
    print(f"\nBest model: {best_model_name} with RMSE = {results[best_model_name]['rmse']:.4f}")

    return best_model, best_model_name, results


def tune_gradient_boosting(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6, 8],
    }

    halving_search = HalvingGridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        factor=2
    )

    halving_search.fit(X_train, y_train)
    print(f"Best parameters: {halving_search.best_params_}")
    print(f"Best RMSE: {-halving_search.best_score_:.4f}")

    return halving_search.best_estimator_


def train_xgboost(X_train, y_train, X_val, y_val):
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    print(f"XGBoost Validation RMSE: {rmse:.4f}")

    return xgb_model, rmse


def train_lightgbm(X_train, y_train, X_val, y_val):
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    print(f"LightGBM Validation RMSE: {rmse:.4f}")

    return lgb_model, rmse


# Visualization Functions

def visualize_results(results, y_val, X_columns, best_model_name):
    # 1. RMSE comparison
    plt.figure(figsize=(8, 5))
    rmse_values = [results[name]['rmse'] for name in results]
    sns.barplot(x=list(results.keys()), y=rmse_values)
    plt.ylabel("Validation RMSE")
    plt.title("Model Comparison - RMSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison_rmse.png", dpi=150)
    plt.show()

    # 2. Predicted vs Actual for best model
    best_pred = results[best_model_name]['y_pred']
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_val, y=best_pred)
    plt.xlabel("Actual site_eui")
    plt.ylabel("Predicted site_eui")
    plt.title(f"Predicted vs Actual - {best_model_name}")
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=150)
    plt.show()

    # 3. Feature importance (top 10 only)
    model = results[best_model_name]['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx = importances.argsort()[-10:][::-1]
        top_features = [(X_columns[i], importances[i]) for i in top_idx]

        print("\nTop 10 Feature Importances (largest to smallest):")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
    else:
        print(f"{best_model_name} does not provide feature importances.")


def plot_feature_importance(best_model, X):
    """Plot top 10 feature importances."""
    importances = best_model.feature_importances_
    top_idx = importances.argsort()[-10:][::-1]
    top_features = [X.columns[i] for i in top_idx]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[top_idx], y=top_features, palette="viridis")
    plt.title("Top 10 Features Driving Energy Use")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()


def plot_feature_correlations(X, y, best_model):
    """Plot correlation of top features with target."""
    importances = best_model.feature_importances_
    top_idx = importances.argsort()[-10:][::-1]
    top_features = [X.columns[i] for i in top_idx]

    corr_with_target = X[top_features].corrwith(y)
    corr_with_target = corr_with_target.reindex(
        corr_with_target.abs().sort_values(ascending=False).index
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(x=corr_with_target.values, y=corr_with_target.index, palette="viridis")
    plt.xlabel("Correlation with Target (site_eui)")
    plt.title("Top 10 Features Correlation with Target")
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig("feature_correlations.png", dpi=150)
    plt.show()


def plot_top_predicted_buildings(test_ids, predictions, n=10):
    """Plot top N highest predicted energy-use buildings."""
    pred_df = pd.DataFrame({'id': test_ids, 'predicted_site_eui': predictions})
    top_buildings = pred_df.sort_values(by='predicted_site_eui', ascending=False).head(n)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='id',
        y='predicted_site_eui',
        data=top_buildings,
        palette="magma",
        order=top_buildings.sort_values(by='predicted_site_eui', ascending=False)['id']
    )
    plt.xlabel("Building ID")
    plt.ylabel("Predicted site_eui")
    plt.title(f"Top {n} Predicted Highest Energy-Use Buildings (High → Low)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("top_predicted_buildings.png", dpi=150)
    plt.show()


def plot_feature_vs_target(X, y, feature, plot_type='scatter'):
    """Plot a single feature against the target variable."""
    plt.figure(figsize=(8, 5))
    if plot_type == 'scatter':
        sns.scatterplot(x=X[feature], y=y)
    elif plot_type == 'box':
        sns.boxplot(x=X[feature], y=y)
        plt.xticks(rotation=45)
    plt.xlabel(feature)
    plt.ylabel("site_eui")
    plt.title(f"{feature} vs Energy Use")
    plt.tight_layout()
    plt.show()


# Main Function
def main():
    if not os.path.exists(TRAIN_PATH):
        print(f"\nError: Training data not found at '{TRAIN_PATH}'")
        print(f"Please place your data files in the '{DATA_DIR}/' directory.")
        print("Expected files: train.csv, test.csv")
        return

    if not os.path.exists(TEST_PATH):
        print(f"\nError: Test data not found at '{TEST_PATH}'")
        print(f"Please place your data files in the '{DATA_DIR}/' directory.")
        return

    # Read data
    train_data, test_data = read_data(TRAIN_PATH, TEST_PATH)
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Preprocess data
    X, y, test_df, test_ids, num_cols, cat_cols = preprocess_data(
        train_data, test_data, target_col=TARGET_COL
    )
    print(f"Features shape: {X.shape}")
    print(f"Numeric columns: {len(num_cols)}, Categorical columns: {len(cat_cols)}")

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    # Train and evaluate models
    best_model, best_model_name, all_results = evaluate_models(
        X_train, y_train, X_val, y_val
    )

    # Train additional models (XGBoost and LightGBM)
    xgb_model, xgb_rmse = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model, lgb_rmse = train_lightgbm(X_train, y_train, X_val, y_val)

    # Add to results for comparison
    all_results["XGBoost"] = {
        "model": xgb_model,
        "rmse": xgb_rmse,
        "y_pred": xgb_model.predict(X_val)
    }
    all_results["LightGBM"] = {
        "model": lgb_model,
        "rmse": lgb_rmse,
        "y_pred": lgb_model.predict(X_val)
    }

    # Update best model if XGBoost or LightGBM is better
    all_rmse = {name: results["rmse"] for name, results in all_results.items()}
    best_model_name = min(all_rmse, key=all_rmse.get)
    best_model = all_results[best_model_name]["model"]
    print(f"\nFinal best model: {best_model_name} with RMSE = {all_rmse[best_model_name]:.4f}")

    # Visualize results
    visualize_results(all_results, y_val, X.columns, best_model_name)

    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, X)
        plot_feature_correlations(X, y, best_model)

    # Predict on test set with best model
    print("\nGenerating predictions on test set...")
    test_predictions = best_model.predict(test_df)

    # Plot top predicted buildings
    plot_top_predicted_buildings(test_ids, test_predictions)

    # Save output
    output = pd.DataFrame({'id': test_ids, 'site_eui': test_predictions})
    output.to_csv('submission_best_model.csv', index=False)
    print("\nPredictions saved to 'submission_best_model.csv'")



if __name__ == "__main__":
    main()
