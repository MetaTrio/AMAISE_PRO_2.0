# from helper import *
import pandas as pd
import logging
import click
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


@click.command()
@click.option(
    "--pred",
    "-p",
    help="path to predicted labels file",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--true",
    "-t",
    help="path to true labels file",
    type=click.Path(exists=True),
    required=True,
)
@click.help_option("--help", "-h", help="Show this message and exit")
#def main(predfile, truefile):
def main(pred, true):

    logger = logging.getLogger(f"amaisepro")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    pred_df = pd.read_csv(pred)
    # # Read the true labels file into a DataFrame, using the first two columns and naming them "id" and "label"
    # true_df = pd.read_csv(true, usecols=[0, 1], names=["id", "label"], header=None)
    
    # Read the true labels file into a DataFrame, selecting the 'id' and 'y_true' columns
    true_df = pd.read_csv(true, usecols=['id', 'y_true'])   

    pred_dict = {}
    # for row in pred_df:
    #     pred_dict[row["id"]] = row["pred_label"]

    for _, row in pred_df.iterrows():
        pred_dict[row["id"]] = row["pred_label"]

    true_list = [[], [], [], [], [], []]

    # for row in true_df:
    #     true_list[row["label"]].append(row["id"])
    for _, row in true_df.iterrows():
        # true_list[row["label"]].append(row["id"])
        true_list[row["y_true"]].append(row["id"])

    pred = []
    true = []

    for clz in range(6):
        for ele in true_list[clz]:
            pred.append(pred_dict[ele])
        true.extend(np.full(len(true_list[clz]), clz))

    logger.info(
        f'\n {classification_report(true,pred,target_names=["Host", "Bacteria", "Virus", "Fungi", "Archaea", "Protozoa"],)}'
    )

    #ROC
    # Step 3: Extract true labels and predicted probabilities
    true_labels = true_df['y_true'].values

    # Predicted probabilities for each class
    pred_probs = pred_df[['prob_host', 'prob_bacteria', 'prob_virus', 'prob_fungi', 'prob_archaea', 'prob_protozoa']].values

    # Step 4: Binarize the labels for multi-class ROC computation
    # Assuming you have 6 classes (0: host, 1: bacteria, 2: virus, 3: fungi, 4: archaea, 5: protozoa)
    true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3, 4, 5])

    # Step 5: Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = true_labels_bin.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Step 6: Plot ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class Classification')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    # Step 7: Print AUROC values
    for i in range(n_classes):
        print(f'AUROC for class {i}: {roc_auc[i]:.2f}')


if __name__ == "__main__":
    main()
