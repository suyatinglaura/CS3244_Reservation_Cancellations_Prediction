from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def feature_selection_using_rfe_on_accuracy(Model, X_train_data, Y_train_data, X_test_data, Y_test_data):
    num_features = []
    accuracies = []

    bestAccuracy = 0
    currentBestRFE = None
    df_features = pd.DataFrame({
    'attribute': X_train_data.columns,
    'selected': None,
    'ranking': 0
    })

    # Iterate over a range of desired features
    for n in range(1, X_train_data.shape[1], 5):  # Adjust the step size and range as necessary
        # RFE with random Forest
        rfe = RFE(estimator=Model, n_features_to_select=n)
        rfe.fit(X_train_data, Y_train_data.values.ravel())

        # Predict on the test set
        y_pred = rfe.predict(X_test_data)

        # Calculate accuracy and store the result
        accuracy = accuracy_score(Y_test_data, y_pred)
        accuracies.append(accuracy)
        num_features.append(n)

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            currentBestRFE = rfe
            df_features = pd.DataFrame({
            'attribute': X_train_data.columns,
            'selected': rfe.support_,
            'ranking': rfe.ranking_
            })

    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Model Performance vs. Number of Features')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    print("Best Accuracy: ", bestAccuracy)
    print("Best Number Of Features: ", currentBestRFE.n_features_to_select)
    return df_features