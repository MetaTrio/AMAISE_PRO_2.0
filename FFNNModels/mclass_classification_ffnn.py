from initialFFNN import *
import torch
import pandas as pd
from Bio import SeqIO
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import logging
import click
import numpy as np
import psutil
from sklearn.metrics import classification_report


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
    type_ = type_
    k_mers = input_kmers
    modelPath = model
    resultPath = output
    if resultPath[-1] != "/":
        resultPath = resultPath + "/"

    logger = logging.getLogger(f"amaisepro")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    fileHandler = logging.FileHandler(f"{resultPath}info.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    k_mer_arr = pd.read_csv(k_mers, header=None).to_numpy()

    input_data = []

    logger.info("parsing data...")

    startTime = time.time()

    i = 0
    for row in k_mer_arr:
        # FFNN expects a flat 1D vector of 32 elements as input, so ensure reshaping is done accordingly
        # input_data.append(np.reshape(row.astype(np.float32), (32,)))  # 32 features
        input_data.append(np.reshape(row.astype(np.float32), (42,)))   #2mer, 3mer
        # input_data.append(np.reshape(row.astype(np.float32), (512,)))  # 5mer
        # input_data.append(np.reshape(row.astype(np.float32), (136,)))  # 4mer
        #input_data.append(np.reshape(row.astype(np.float32), (680,)))  # 3,4,5mer
        i = i + 1

    accession_numbers = []
    for seq in SeqIO.parse(inputset, type_):
        accession_numbers.append(seq.id)

    endTime = time.time()
    encoding_time_diff = (endTime - startTime) / 60
    logger.info(f"Total time taken to parse data: {encoding_time_diff} min")

    logger.info("initializing and loading the model...")

    # Load the new FFNN model
    model = nn.DataParallel(FeedForwardNN())
    model.load_state_dict(torch.load(modelPath, device))
    model = model.to(device)
    model.eval()

    logger.info("predicting the classes...")

    startTime_ = time.time()
    dataLoader = DataLoader(input_data, shuffle=False, batch_size=2048)
    predicted = []
    probabilities = []

    with torch.no_grad():
        for step, test_x in enumerate(dataLoader):
            test_x = test_x.to(device)
            # Get model predictions and apply softmax
            pred = model(test_x)
            # Extract class labels with the highest probabilities as the predicted labels
            _, predicted_labels = torch.max(pred, 1)
            predicted.extend(predicted_labels.cpu().numpy())
            # probabilities.extend(pred.cpu().numpy())
            probabilities.extend(pred.cpu().numpy().tolist())  # Convert to standard Python floats

    endTime = time.time()
    predicting_time_diff = (endTime - startTime_) / 60
    logger.info(f"Time taken to predict the results: {predicting_time_diff} min")

    # Debugging: Print lengths of arrays
    print(f"Length of accession_numbers: {len(accession_numbers)}")
    print(f"Length of predicted: {len(predicted)}")
    print(f"Length of probabilities: {len(probabilities)}")

    # Check if lengths match
    if len(accession_numbers) != len(predicted):
        raise ValueError("The lengths of accession_numbers and predicted arrays do not match.")

    class_names = ["host", "bacteria", "virus", "fungi", "archaea", "protozoa"]

    # pred_df = pd.DataFrame({"id": accession_numbers, "pred_label": predicted})
    # pred_df.to_csv(f"{resultPath}predictions.csv", index=False)

    # # Create a DataFrame with probabilities for each class
    # probability_columns = [f"prob_{class_name}" for class_name in class_names]
    # pred_df = pd.DataFrame(probabilities, columns=probability_columns)
    # pred_df["id"] = accession_numbers
    # pred_df["pred_label"] = predicted

    # # Reorder columns to have 'id' and 'pred_label' as the first two columns
    # pred_df = pred_df[["id", "pred_label"] + probability_columns]

    # # Save DataFrame to CSV
    # pred_df.to_csv(f"{resultPath}predicted_probabilities.csv", index=False)

    # Create a DataFrame with 'id', 'pred_label', and 'probabilities' (vector for each class)
    pred_prob_df = pd.DataFrame({
        "id": accession_numbers,
        "pred_label": predicted,
        "probabilities": [list(prob) for prob in probabilities]  # store as a list for each row
    })

    pred_prob_df.to_csv(f"{resultPath}predictions.csv", index=False)

    # id_label_dict = dict(zip(pred_df["id"], pred_df["pred_label"]))
    # class_seqs = [[], [], [], [], [], []]
    # for seq in SeqIO.parse(inputset, type_):
    #     class_seqs[id_label_dict[seq.id]].append(seq)

    # for i in range(1, 6):
    #     with open(f"{resultPath}{class_names[i]}.{type_}", "w") as file:
    #         SeqIO.write(class_seqs[i], file, type_)

    endTime = time.time()
    memory = psutil.Process().memory_info()

    logger.info("Total time: {:.2f} min".format((endTime - startTime) / 60))

    # Convert memory values from bytes to GB
    rss_gb = memory.rss / (1024 ** 3)
    vms_gb = memory.vms / (1024 ** 3)
    shared_gb = memory.shared / (1024 ** 3)
    text_gb = memory.text / (1024 ** 3)
    data_gb = memory.data / (1024 ** 3)

    # Log the values in GB
    logger.info(f"Memory usage: rss={rss_gb:.2f} GB, vms={vms_gb:.2f} GB, shared={shared_gb:.2f} GB, text={text_gb:.2f} GB, data={data_gb:.2f} GB")


if __name__ == "__main__":
    main()
