import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold



def feature_selection_using_rfecv_on_f1score(Model, X_train_data, Y_train_data):
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to select features based on the F1 score.
    
    This function applies RFECV to a provided classifier model to determine the optimal number of features that
    maximizes the F1 score during cross-validation. It visualizes the model's performance as the number of features varies.

    Parameters:
    - Model : A scikit-learn-compatible classifier.
    - X_train_data : DataFrame, feature set for training.
    - Y_train_data : DataFrame/Series, target variable for training.
    
    Returns:
    - df_features : DataFrame containing the features, whether they are selected, and their ranking.
    """
    steps = 1
    cv = StratifiedKFold(5)
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
        "feature": X_train_data.columns,
        "selected": rfecv.support_,
        "ranking": rfecv.ranking_
    })

    print("Optimal number of features based on F1 Score: %d" % rfecv.n_features_)
    return df_features

def feature_selection_using_rfecv_on_accuracyscore(Model, X_train_data, Y_train_data):
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to select features based on the Accuracy score.
    
    This function applies RFECV to a provided classifier model to determine the optimal number of features that
    maximizes the Accuracy score during cross-validation. It visualizes the model's performance as the number of features varies.

    Parameters:
    - Model : A scikit-learn-compatible classifier.
    - X_train_data : DataFrame, feature set for training.
    - Y_train_data : DataFrame/Series, target variable for training.
    
    Returns:
    - df_features : DataFrame containing the features, whether they are selected, and their ranking.
    """
    steps = 1
    cv = StratifiedKFold(5)
    # Initialize RFECV with the model and desired parameters
    rfecv = RFECV(estimator=Model, step=steps, cv=cv, scoring="accuracy", min_features_to_select=1, n_jobs=-1)
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
        "feature": X_train_data.columns,
        "selected": rfecv.support_,
        "ranking": rfecv.ranking_
    })

    print("Optimal number of features based on Accuracy Score: %d" % rfecv.n_features_)
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
    - DataFrame with all features and their selected status.
    """
    selected_features = []
    best_accuracy = 0
    total_features = len(X_train.columns) if total_features is None else total_features
    y_train_array = y_train.values.ravel()

    while len(selected_features) < total_features:
        best_feature = None
        for feature in X_train.columns:
            if feature not in selected_features:
                trial_features = selected_features + [feature]
                model.fit(X_train[trial_features], y_train_array)
                y_pred = model.predict(X_test[trial_features])
                accuracy = accuracy_score(y_test, y_pred)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
        else:
            break

    # Creating a DataFrame with feature names and their selection status
    feature_data = pd.DataFrame({
        "feature": X_train.columns,
        "selected": [feature in selected_features for feature in X_train.columns]
    })
    print("Selected features:", selected_features)
    print("Best accuracy:", best_accuracy)

    return feature_data

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
    - DataFrame with all features and their selected status.
    """
    y_train_array = y_train.values.ravel()
    selected_features = list(X_train.columns)
    best_accuracy = accuracy_score(y_test, model.fit(X_train, y_train_array).predict(X_test))
    print("Initial score: ", best_accuracy)
    improvement = True

    while improvement:
        improvement = False
        for feature in selected_features.copy():
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

    # Creating a DataFrame with feature names and their selection status
    feature_data = pd.DataFrame({
        "feature": X_train.columns,
        "selected": [feature in selected_features for feature in X_train.columns]
    })
    print("Selected features:", selected_features)
    print("Best accuracy:", best_accuracy)

    return feature_data

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

def select_k_best_with_mutual_info(X_train, y_train, k):
    """
    Perform feature selection using the SelectKBest method with mutual_info_classif.

    Parameters:
    - X_train (DataFrame): Input features for training.
    - y_train (Series or array-like): Target variable for training.
    - k (int): Number of top features to select based on mutual information.

    Returns:
    - DataFrame with all features and their selected status.
    """
    # Initialize SelectKBest with mutual_info_classif and the number of features to select
    selector = SelectKBest(mutual_info_classif, k=k)

    # Fit SelectKBest to the training data
    selector.fit(X_train, y_train)

    # Get the mask of the selected features
    selected_features_mask = selector.get_support()

    # Create a DataFrame with feature names and their selection status
    feature_data = pd.DataFrame({
        "feature": X_train.columns,
        "selected": selected_features_mask
    })

    return feature_data

def save_feature_data_to_csv(models_feature_data, model_name):
    def transform_feature_data(selection_name, feature_data):
        transformed_df = feature_data.pivot(index=None, columns='feature', values='selected')
        compressed_row = transformed_df.max(axis=0, skipna=True)
        compressed_df = pd.DataFrame([compressed_row])
        compressed_df.insert(0, 'Selection Name', selection_name)
        return compressed_df

    transformed_dfs = [transform_feature_data(model_name, feature_data) for model_name, feature_data in models_feature_data]

    final_df = pd.concat(transformed_dfs, ignore_index=True)
    final_df.to_csv(f"Data/feature_selection/{model_name}_selected_features.csv")
    return final_df

def get_selected_features_as_list(df_features):
    return df_features[df_features['selected'] == True]['feature'].tolist()
