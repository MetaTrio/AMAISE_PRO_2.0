from helper import *
import pandas as pd
import logging
import click
import numpy as np
from sklearn.metrics import classification_report


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


if __name__ == "__main__":
    main()
