import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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

def export_final_selected_features_to_csv(features_list, modelName):
    """
    Exports the final selected list of features to a CSV file for a model.

    Parameters:
    features_list (list of str): The list of feature names to export.
    modelName (str): Name of the Model to be Saved
    """
    features_df = pd.DataFrame({'selected_features': features_list})
    features_df.to_csv(f"FinalModels/Feature_Selection/{modelName}_selected_features.csv", index=False)
    print(f"Features exported successfully to FinalModels/Feature_Selection/{modelName}")

def import_final_selected_features_from_csv(file_path):
    """
    Imports a list of features from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file to import the features from.

    Returns:
    list of str: selected features from model
    """
    features_df = pd.read_csv(file_path)
    # print(f"Imported features from {file_path} successfully")
    return features_df['selected_features'].tolist()

def get_selected_features_as_list(df_features):
    return df_features[df_features['selected'] == True]['feature'].tolist()
