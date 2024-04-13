import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

def feature_selection_using_rfecv_on_f1score(Model, X_train_data, Y_train_data):
    steps = 1
    cv = StratifiedKFold(5)
    f1_scorer = make_scorer(f1_score)
    # Initialize RFECV with the model and desired parameters
    rfecv = RFECV(estimator=Model, step=steps, cv=cv, scoring="f1", min_features_to_select=1, n_jobs=-1)
    rfecv.fit(X_train_data, Y_train_data.values.ravel())
    
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    
    # Visualize the optimal number of features
    plt.figure(figsize=(10, 6))
    plt.title("Model Performance vs. Number of Features")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross-Validation Score")
    plt.plot(range(1, (n_scores * steps) + 1, steps), rfecv.cv_results_["mean_test_score"], marker='o')
    plt.xticks(np.arange(0, (n_scores * steps) + 1, 5))
    plt.grid(True)
    plt.show()

    # Gather selected features into a DataFrame
    df_features = pd.DataFrame({
        "attribute": X_train_data.columns,
        "selected": rfecv.support_,
        "ranking": rfecv.ranking_
    })

    print("Optimal number of features based on F1 Score: %d" % rfecv.n_features_)
    return df_features

def feature_selection_using_rfecv_on_rocaucscore(Model, X_train_data, Y_train_data):
    steps = 1
    cv = StratifiedKFold(5)
    # Initialize RFECV with the model and desired parameters
    rfecv = RFECV(estimator=Model, step=steps, cv=cv, scoring="roc_auc", min_features_to_select=1, n_jobs=-1)
    rfecv.fit(X_train_data, Y_train_data.values.ravel())
    
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    
    # Visualize the optimal number of features
    plt.figure(figsize=(10, 6))
    plt.title("Model Performance vs. Number of Features")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross-Validation Score")
    plt.plot(range(1, (n_scores * steps) + 1, steps), rfecv.cv_results_["mean_test_score"], marker='o')
    plt.xticks(np.arange(0, (n_scores * steps) + 1, 5))
    plt.grid(True)
    plt.show()

    # Gather selected features into a DataFrame
    df_features = pd.DataFrame({
        "attribute": X_train_data.columns,
        "selected": rfecv.support_,
        "ranking": rfecv.ranking_
    })

    print("Optimal number of features based on ROC_AUC Score: %d" % rfecv.n_features_)
    return df_features

# forward selection based on accuracy_score comparisons
def forward_selection_with_metrics(X_train, y_train, X_test, y_test, model, total_features=None):
    """
    Perform forward selection for feature selection with a given machine learning model.

    Parameters:
    - X_train (array-like): Input features for training.
    - y_train (array-like): Target variable for training.
    - X_test (array-like): Input features for testing.
    - y_test (array-like): Target variable for testing.
    - model: Machine learning model object with fit and predict methods.
    - total_features (int): Total number of features to be selected. If None, select until no further improvement.

    Returns:
    - selected_features (list): List of selected feature indices.
    """
    selected_features = []
    best_accuracy = 0
    total_features = len(X_train.columns)
    # Convert DataFrame to NumPy array
    y_train_array = y_train.values.ravel()

    while len(selected_features) < total_features:
        best_feature = None
        for feature in X_train.columns:
            if feature not in selected_features:
                trial_features = selected_features + [feature]
                # Train the ML model with trial_features
                model.fit(X_train[trial_features], y_train_array)
                # Evaluate model performance using accuracy_score
                y_pred = model.predict(X_test[trial_features])
                accuracy = accuracy_score(y_test, y_pred)
                # Check if accuracy improved
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature

        # If a new feature improves performance, add it to selected_features
        if best_feature is not None:
            selected_features.append(best_feature)
        else:
            # No further improvement can be made
            break

    print("Selected features:", selected_features)
    print("Best accuracy:", best_accuracy)

    return selected_features

#backward elimination based on accuracy_score comparison
def backward_elimination_with_metrics(X_train, y_train, X_test, y_test, model, threshold=0, verbose=False):
    """
    Perform backward elimination for feature selection with a given machine learning model.

    Parameters:
    - X_train (array-like): Input features for training.
    - y_train (array-like): Target variable for training.
    - X_test (array-like): Input features for testing.
    - y_test (array-like): Target variable for testing.
    - model: Machine learning model object with fit and predict methods.
    - threshold (float): Minimum improvement threshold to continue removing features.
    - verbose (bool): Whether to print progress information.

    Returns:
    - selected_features (list): List of selected feature indices.
    """
    y_train_array = y_train.values.ravel()
    selected_features = list(X_train.columns)
    best_accuracy = accuracy_score(y_test, model.fit(X_train, y_train_array).predict(X_test))
    improvement = True

    while improvement:
        improvement = False
        for feature in selected_features:
            trial_features = selected_features.copy()
            trial_features.remove(feature)
            model.fit(X_train[trial_features], y_train_array)
            y_pred = model.predict(X_test[trial_features])
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy + threshold:
                selected_features.remove(feature)
                best_accuracy = accuracy
                improvement = True
                if verbose:
                    print(f"Removed feature {feature}, Accuracy: {accuracy}")
    
    print("Selected features:", selected_features)
    print("Best accuracy:", best_accuracy)
   
    return selected_features

def compute_permutation_importance(model, X_train, Y_train):
    """
    Compute permutation importance for tree-based model.

    Parameters:
    - model: Machine learning model object with fit and predict methods.
    - X_train: Input features for training.
    - y_train: Target variable for training.

    Returns:
    - feature_importances (list): List of feature importance scores sorted in increasing order.
    - feature_names (list): List of feature names sorted based on feature importance scores.
    """
    pi_result = permutation_importance(model, X_train, Y_train.values.ravel(), random_state=47, n_repeats=10)
    # Get feature importance score
    feature_importances = pi_result['importances_mean']
    # Get feature names
    feature_names = X_train.columns
    # Get indices that would sort feature importances
    sorted_indices = np.argsort(feature_importances)
    return feature_importances[sorted_indices], feature_names[sorted_indices]

def plot_feature_importance(feature_importance, feature_names):
    """
    Plotting feature importance.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()
