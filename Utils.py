
import matplotlib.pyplot as plt

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
            axs[i].text(index, value, str(round(value, 2)), ha='center', va='bottom')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.03), ncol=len(metrics['Classification Model']))

    plt.tight_layout()
    plt.show()

