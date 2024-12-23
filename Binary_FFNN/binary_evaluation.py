import pandas as pd
import logging
import click
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os 

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
    predicted_labels = []
    true_labels = []
    for clz in range(2):
        for ele in true_list[clz]:
            predicted_labels.append(pred_dict[ele])
        true_labels.extend(np.full(len(true_list[clz]), clz))
    
    # Generate classification report
    logger.info(
        f'\n {classification_report(true_labels, predicted_labels, target_names=["Host", "Microbial"])}'
    )

    # Compute and log confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
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

    # merge true and predicted labels
    merged_df = pd.merge(true_df,pred_df, on='id')
    

    # Save concatenated DataFrame to the same path as the predicted file
    output_path = os.path.join(os.path.dirname(pred), "trueAndPredictedLabels.csv")
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Concatenated CSV saved to {output_path}")

    # Calculate sensitivity and specificity
    # Confusion matrix: [TN, FP], [FN, TP]
    TN, FP, FN, TP = cm.ravel()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Save sensitivity and specificity to logs
    logger.info(f"Sensitivity: {sensitivity:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")

if __name__ == "__main__":
    main()