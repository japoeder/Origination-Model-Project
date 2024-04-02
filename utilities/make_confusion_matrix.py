import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def make_cm(
    inputs,
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    sum_stats=True,
    labels=False,
    figsize=None,
    cmap="Blues",
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    """

    group_names = ["True Positive", "False Negative", "False Positive", "True Negative"]
    categories = labels

    cf_pre = confusion_matrix(inputs[0], inputs[1], labels=[0, 1])
    cf = np.array([[cf_pre[0][0], cf_pre[1][0]], [cf_pre[0][1], cf_pre[1][1]]])

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[1, :])
            recall = cf[1, 1] / sum(cf[:, 1])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy = {:0.3f}\nPrecision = {:0.3f}\nRecall      = {:0.3f}\nF1 Score  = {:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    plt.ylabel("True label", fontsize=12)
    plt.title("Predicted label", fontsize=12)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position("top")
    ax.text(0.75, 2.4, stats_text)
    plt.show()