import pandas as pd
from Bio import SeqIO
import time
import logging
import click
import numpy as np
import psutil
import xgboost as xgb

@click.command()
@click.option(
    "--input_fasta_fastq",
    "-i",
    help="path to input (fasta/fastq) data",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "type_",
    "-t",
    help="type of the input file (fasta or fastq)",
    type=click.Choice(["fasta", "fastq"]),
    required=True,
)
@click.option(
    "--input_kmers",
    "-k",
    help="path to generated k-mers",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--model",
    "-m",
    help="path to the model",
    type=click.Path(exists=True),
    default="models/AMAISE_PRO",
    required=False,
)
@click.option(
    "--output",
    "-o",
    help="path to a folder to save predictions",
    type=str,
    required=True,
)
@click.help_option("--help", "-h", help="Show this message and exit")
def main(input_fasta_fastq, type_, input_kmers, model, output):

    inputset = input_fasta_fastq
    k_mers = input_kmers
    modelPath = model
    resultPath = output
    if resultPath[-1] != "/":
        resultPath = resultPath + "/"

    logger = logging.getLogger(f"amaisepro")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    fileHandler = logging.FileHandler(f"{resultPath}info.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    logger.info("Parsing data...")
    k_mer_arr = pd.read_csv(k_mers, header=None).to_numpy()
    
    input_data = np.array([row.astype(np.float32) for row in k_mer_arr])
    print(input_data.shape)

    accession_numbers = [seq.id for seq in SeqIO.parse(inputset, type_)]

    logger.info("Initializing and loading the model...")
    model = xgb.Booster()  # Initialize an XGBoost model
    model.load_model(modelPath)  # Load the trained model

    start_time = time.time()

    logger.info("Predicting the classes...")
    dtest = xgb.DMatrix(input_data)  # Convert input data to DMatrix
    predictions = model.predict(dtest)  # Get predictions
    print(predictions)
    print(predictions.shape)

    # for i in range(1000):  # Ensure we don't exceed the array length
    #     print(f"Prediction {i+1}: {predictions[i]}")


    # Convert probabilities to class labels
    predicted_labels = predictions
    print(predicted_labels)

    
    logger.info("Processing results...")
    
    # Save predictions to CSV
    class_names = ["host", "bacteria", "virus", "fungi", "archaea", "protozoa"]
    pred_df = pd.DataFrame({"id": accession_numbers, "pred_label": predicted_labels})
    pred_df.to_csv(f"{resultPath}predictions.csv", index=False)

    # Optionally save probabilities for each class
    # probability_columns = [f"prob_{class_name}" for class_name in class_names]
    # pred_df = pd.DataFrame(predictions, columns=probability_columns)
    # pred_df["id"] = accession_numbers
    # pred_df["pred_label"] = predicted_labels
    # pred_df = pred_df[["id", "pred_label"] + probability_columns]
    # pred_df.to_csv(f"{resultPath}predicted_probabilities.csv", index=False)

    end_time = time.time()
    memory = psutil.Process().memory_info()

    logger.info("Total time taken: {:.2f} min".format((end_time - start_time) / 60))
    logger.info(f"Memory usage: {memory}")

if __name__ == "__main__":
    main()
