import argparse
import pandas as pd
import time
import logging
import click
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import psutil
import xgboost as xgb

@click.command()
@click.option(
    "--input",
    "-i",
    help="path to training data",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--labels",
    "-l",
    help="path to labels of training data",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--model", "-m", help="path to save model", type=str, required=True)
@click.option("--output", "-o", help="path to save output", type=str, required=True)
@click.option(
    "--batch_size",
    "-b",
    help="batch size",
    type=int,
    default=1024,
    show_default=True,
    required=False,
)
@click.option(
    "--epoches",
    "-e",
    help="number of epoches (ignored for XGBoost)",
    type=int,
    default=100,
    show_default=True,
    required=False,
)

@click.option(
    "--learning_rate",
    "-lr",
    help="learning rate",
    type=float,
    default=0.05,
    show_default=True,
    required=False,
)
@click.help_option("--help", "-h", help="Show this message and exit")
def main(input, labels, model, output, batch_size, epoches, learning_rate):

    newModelPath = model
    inputset = input
    labelset = labels
    resultPath = output
    batchSize = batch_size
    epoches = epoches
    learningRate = learning_rate

    logger = logging.getLogger(f"amaisepro")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    fileHandler = logging.FileHandler(f"{resultPath}.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    logger.info(f"Model path: {newModelPath}")
    logger.info(f"Input path: {inputset}")
    logger.info(f"Labels path: {labelset}")
    logger.info(f"Results path: {resultPath}")

    logger.info(f"Learning rate: {learningRate}")
    logger.info(f"Batch size: {batchSize}")

    # Load data
    train_df = pd.read_csv(labelset)
    train_data_arr = pd.read_csv(inputset, header=None).to_numpy()

    X, y = [], []

    logger.info("parsing data...")

    startTime = time.time()

    i = 0
    for row in train_data_arr:
        X.append(row.astype(np.float32))  # Convert to float32
        label = train_df["y_true"][i]  # Assume the label column is 'y_true'
        y.append(label)
        i += 1



    endTime = time.time()
    encoding_time_diff = (endTime - startTime) / 60
    logger.info(f"Total time taken to parse data: {encoding_time_diff:.2f} min")

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y,random_state=0)

    logger.info("Initializing the XGBoost model...")

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    

    # Set XGBoost parameters
    params = {
        "eta": learningRate,  # Learning rate
        "objective": "multi:softmax",  # Multi-class classification
        "num_class": len(set(y)),  # Number of classes
        "max_depth": 16,  # Depth of the trees
        "eval_metric": ["mlogloss", "merror"],  # Logarithmic loss and error rate

    }

    evals = [(dtrain, "train"), (dval, "validation")]
    eval_results = {} 
    # Train the XGBoost model
    bst = xgb.train(params, dtrain, epoches, evals, early_stopping_rounds=10,evals_result = eval_results)
    

    logger.info("Saving the model...")
    bst.save_model(newModelPath)

    endTime = time.time()
    memory = psutil.Process().memory_info()

    logger.info(
        "Total time taken to train the model: {:.2f} min".format(
            (endTime - startTime) / 60
        )
    )
    logger.info(f"Memory usage: {memory}")

    logger.info(f"Model saved at: {newModelPath}")

    # Plot training validation losses vs. epochs
    # eval_results = bst.eval_set(evals)
    # print(eval_results)
    train_losses = eval_results['train']['mlogloss']
    val_losses = eval_results['validation']['mlogloss']
    train_errors = eval_results['train']['merror']
    val_errors = eval_results['validation']['merror']

    epoch_list = [e + 1 for e in range(len(train_losses))]

    plt.plot(epoch_list, train_losses, label="Training loss")
    plt.plot(epoch_list, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(f"{resultPath}_losses.png", dpi=300, bbox_inches="tight")
    plt.clf()


     # Plot Accuracies
    train_accuracies = [1 - error for error in train_errors]
    val_accuracies = [1 - error for error in val_errors]

    plt.figure()
    plt.plot(epoch_list, train_accuracies, label="Training accuracy")
    plt.plot(epoch_list, val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{resultPath}_accuracies.png", dpi=300, bbox_inches="tight")
    plt.clf()

if __name__ == "__main__":
    main()


# python your_script_name.py --input /path/to/your/input.csv --labels /path/to/your/labels.csv --model /path/to/save/model.model --output /path/to/save/output \

