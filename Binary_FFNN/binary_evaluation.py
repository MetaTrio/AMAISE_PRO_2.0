import pandas as pd
import logging
import click
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

@click.command()
@click.option(
    "--pred",
    "-p",
    help="Path to predicted labels file",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--true",
    "-t",
    help="Path to true labels file",
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

def main(pred, true, logfile):
    # Setup logger
    logger = logging.getLogger("Binary FFNN")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)

    # Formatter for log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler (for logging to console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler (for logging to a file, if the logfile option is provided)
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    # Load predicted and true label files
    pred_df = pd.read_csv(pred)
    true_df = pd.read_csv(true, usecols=['id', 'y_true'])

    # Create a dictionary of predictions
    pred_dict = {row["id"]: row["pred_label"] for _, row in pred_df.iterrows()}

    # Create a list to store true labels by class
    true_list = [[] for _ in range(2)]
    for _, row in true_df.iterrows():
        true_list[row["y_true"]].append(row["id"])
   
    # Prepare true and predicted label lists
    pred = []
    true = []
    for clz in range(2):
        for ele in true_list[clz]:
            pred.append(pred_dict[ele])
        true.extend(np.full(len(true_list[clz]), clz))
    
    # Generate classification report
    logger.info(
        f'\n {classification_report(true, pred, target_names=["Host", "Microbial"])}'
    )

    # Compute and log confusion matrix
    cm = confusion_matrix(true, pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Host", "Microbial"],
        columns=["Host", "Microbial"],
    )
    logger.info(f"\nConfusion Matrix:\n{cm_df}")

    # Normalize confusion matrix and log percentage version
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_df_normalized = pd.DataFrame(
        cm_normalized,
        index=["Host", "Microbial"],
        columns=["Host", "Microbial"],
    )
    logger.info(f"\nConfusion Matrix (Percentages):\n{cm_df_normalized}")


if __name__ == "__main__":
    main()


