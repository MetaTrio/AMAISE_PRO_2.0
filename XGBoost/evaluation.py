# from helper import *
import pandas as pd
import logging
import click
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


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
@click.option(
    "--logfile",
    "-lf",
    help="Path to log output file (optional)",
    type=click.Path(),
    required=False,
)
@click.help_option("--help", "-h", help="Show this message and exit")
#def main(predfile, truefile):
def main(pred, true, logfile):

    logger = logging.getLogger(f"amaisepro")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    # File handler (for logging to a file, if the logfile option is provided)
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
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

    # Compute and print confusion matrix
    cm = confusion_matrix(true, pred)
    cm_df = pd.DataFrame(cm, index=["Host", "Bacteria", "Virus", "Fungi", "Archaea", "Protozoa"], 
                         columns=["Host", "Bacteria", "Virus", "Fungi", "Archaea", "Protozoa"])
    logger.info(f"\nConfusion Matrix:\n{cm_df}")

    #get precentage values instead of actual values
    
    # Normalize the confusion matrix by dividing each element by the sum of its row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a DataFrame with normalized values and the desired index/columns
    cm_df = pd.DataFrame(cm_normalized, index=["Host", "Bacteria", "Virus", "Fungi", "Archaea", "Protozoa"], 
                        columns=["Host", "Bacteria", "Virus", "Fungi", "Archaea", "Protozoa"])

    # Log the confusion matrix with percentages
    logger.info(f"\nConfusion Matrix (Percentages):\n{cm_df}")


if __name__ == "__main__":
    main()
