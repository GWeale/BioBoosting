import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from BioBoost import BioBoost, preprocess_data, feature_engineering, load_bioinformatics_data
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

def cross_validate(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = np.mean((y_val - y_pred) ** 2)
        mse_scores.append(mse)
    return np.mean(mse_scores), np.std(mse_scores)

def hyperparameter_tuning(X, y, param_grid, k=5):
    grid = ParameterGrid(param_grid)
    best_mse = float('inf')
    best_params = None
    results = []
    
    for params in grid:
        model = BioBoost(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3)
        )
        mean_mse, std_mse = cross_validate(model, X, y, k)
        results.append({'params': params, 'mean_mse': mean_mse, 'std_mse': std_mse})
        print(f"Params: {params}, Mean MSE: {mean_mse:.4f}, Std MSE: {std_mse:.4f}")
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_params = params
    
    return best_params, best_mse, results

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():

    X, y = load_bioinformatics_data()
    X = preprocess_data(X)
    X = feature_engineering(X)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    }
    
    best_params, best_mse, tuning_results = hyperparameter_tuning(X, y, param_grid, k=5)
    print(f"Best Params: {best_params}, Best Mean MSE: {best_mse:.4f}")
    
    final_model = BioBoost(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth']
    )
    final_model.fit(X, y)
    
    save_model(final_model, 'bioboost_final_model.pkl')
    print("Final model saved to 'bioboost_final_model.pkl'")
    
    final_model.plot_loss()
    plt.show()
    
    importances = final_model.feature_importances_
    indices = np.argsort(list(importances.values()))[::-1]
    sorted_importances = [importances[i] for i in indices]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
    plt.xticks(range(len(sorted_importances)), indices, rotation=90)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
