import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def prob_to_label(preds, threshold=0.5):
    for i in range(len(preds)):
        if preds[i] > threshold:
            preds[i] = 1
        else:
            preds[i] = 0

    return preds


##  Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(preds, sample_dict, print_flag=False, validation=False):
    """
    model : classifier to predict values of X

    """

    # defining an empty list to store train and test results
    score_list = []
    pred_train = preds["pred_train"]
    if validation is True:
        pred_test = preds["pred_val"]
        sample_test = sample_dict["y_val"]
    else:
        pred_test = preds["pred_test"]
        sample_test = sample_dict["y_test"]
    # round the pred_time to 4 decimal places
    est_time = round(preds["est_time"], 4)

    train_acc = accuracy_score(sample_dict["y_train"], pred_train)
    test_acc = accuracy_score(sample_test, pred_test)

    train_recall = recall_score(sample_dict["y_train"], pred_train)
    test_recall = recall_score(sample_test, pred_test)

    train_precision = precision_score(sample_dict["y_train"], pred_train)
    test_precision = precision_score(sample_test, pred_test)

    train_f1 = f1_score(sample_dict["y_train"], pred_train)
    test_f1 = f1_score(sample_test, pred_test)

    score_list.extend(
        (
            train_acc,
            test_acc,
            train_recall,
            test_recall,
            train_precision,
            test_precision,
            train_f1,
            test_f1,
            est_time,
        )
    )

    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if print_flag == True:
        print(
            "Accuracy on training set : ",
            accuracy_score(sample_dict["y_train"], pred_train),
        )
        print(
            "Accuracy on test set : ",
            accuracy_score(sample_dict["y_test"], pred_test),
        )
        print(
            "Recall on training set : ",
            recall_score(sample_dict["y_train"], sample_dict["pred_train"]),
        )
        print(
            "Recall on test set : ",
            recall_score(sample_dict["y_test"], sample_dict["pred_test"]),
        )
        print(
            "Precision on training set : ",
            precision_score(sample_dict["y_train"], sample_dict["pred_train"]),
        )
        print(
            "Precision on test set : ",
            precision_score(sample_dict["y_test"], sample_dict["pred_test"]),
        )
        print(
            "F1-Score on training set : ",
            f1_score(sample_dict["y_train"], sample_dict["pred_train"]),
        )
        print(
            "F1-Score on test set : ",
            f1_score(sample_dict["y_test"], sample_dict["pred_test"]),
        )

    return score_list  # returning the list with train and test scores


def model_comparisons(models_dict, sample_dict, validation=False):
    # defining list of models
    models = [v for v in models_dict.values()]
    labels = [k for k in models_dict.keys()]

    # defining empty lists to add train and test results
    conv_time = []
    acc_train = []
    acc_test = []
    recall_train = []
    recall_test = []
    precision_train = []
    precision_test = []
    f1_train = []
    f1_test = []

    # looping through all the models to get the metrics score - Accuracy, Recall and Precision
    for l in labels:
        if validation is True:
            j = get_metrics_score(models_dict[l], sample_dict, print_flag=False, validation=models_dict[l]["validation"])
        else:
            j = get_metrics_score(models_dict[l], sample_dict, print_flag=False, validation=models_dict[l]["validation"])
        acc_train.append(j[0])
        acc_test.append(j[1])
        recall_train.append(j[2])
        recall_test.append(j[3])
        precision_train.append(j[4])
        precision_test.append(j[5])
        f1_train.append(j[6])
        f1_test.append(j[7])
        conv_time.append(j[8])

    comparison_frame = pd.DataFrame(
        {
            "Model": labels,
            "Conv. Time (sec.)": conv_time,
            "Train_Accuracy": acc_train,
            "Test_Accuracy": acc_test,
            "Train_Recall": recall_train,
            "Test_Recall": recall_test,
            "Train_Precision": precision_train,
            "Test_Precision": precision_test,
            "Train_F1-Score": f1_train,
            "Test_F1-Score": f1_test,
        }
    )

    # Sorting models in decreasing order of test recall
    comparison_frame.sort_values(by="Test_F1-Score", ascending=False)

    return comparison_frame