import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score

# Plot comparision of metrics between models
def plotMetricsGraphComparison(metrics):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    colormap = plt.cm.get_cmap('tab10')

    for i, data in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score']):
        colors = colormap.colors[:len(metrics['Classification Model'])]
        for index, (model, value) in enumerate(zip(metrics['Classification Model'], metrics[data])):
            axs[i].bar(model, value, color=colors[index], label=model if i == 0 else "", edgecolor='k')
        axs[i].set_title(data)
        axs[i].set_ylabel('Score')
        axs[i].set_xlabel('Classification Model')
        axs[i].set_xticks(metrics['Classification Model'])
        axs[i].set_xticklabels(metrics['Classification Model'], rotation=45)
        for index, value in enumerate(metrics[data]):
            axs[i].text(index, value, str(round(value, 3)), ha='center', va='bottom')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.03), ncol=len(metrics['Classification Model']))

    plt.tight_layout()
    plt.show()

# Plot the correlation heatmap
def plot_corrlation_heatmap(data):
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Variables')
    plt.show()

# plot mutual information heatmap
def plot_mi_heatmap(data):
    # calculate mutual information
    def calculate_mutual_info(df, col1, col2):
        return mutual_info_score(df[col1], df[col2])

    # plot heatmap
    mi_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
    for i in data.columns:
        for j in data.columns:
            mi_matrix.loc[i, j] = calculate_mutual_info(data, i, j)

    plt.figure(figsize=(20, 16))  # Adjust the figure size if needed
    sns.heatmap(mi_matrix.astype(float), cmap='coolwarm', fmt='.2f')
    plt.title('Mutual Information Heatmap')
    plt.show()