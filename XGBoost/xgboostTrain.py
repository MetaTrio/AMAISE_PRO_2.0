import xgboost as xgb
from sklearn.metrics import accuracy_score,f1_score
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
import torch

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
    # batchSize = batch_size
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading label data...")
    train_df = pd.read_csv(labelset)
    
    logger.info("Loading input data...")
    train_data_arr = pd.read_csv(inputset, header=None).to_numpy()

    logger.info("Parsing data...")

    startTime = time.time()
    
    # X, y = [], []

    # i = 0
    # for row in train_data_arr:
    #     X.append(row.astype(np.float32))  # Convert to float32
    #     label = train_df["y_true"][i]  # Assume the label column is 'y_true'
    #     y.append(label)
    #     i += 1

    # Preallocate arrays based on the shape of the data
    num_samples = train_data_arr.shape[0]
    num_features = train_data_arr.shape[1]
    X = np.zeros((num_samples, num_features), dtype=np.float32)   # Preallocate X
    y = np.zeros(num_samples, dtype=train_df["y_true"].dtype)     # Preallocate y

    # Populate X and y without an explicit loop
    X = train_data_arr.astype(np.float32)  # Directly assign the entire array
    y = train_df["y_true"].to_numpy()       # Convert y_true to a NumPy array

    # for i in range(num_samples):
    #     X[i] = train_data_arr[i].astype(np.float32)  # Directly assign to preallocated array
    #     y[i] = train_df["y_true"][i]  # Directly assign to preallocated array

    # Parse data and move it to the chosen device
    # i = 0
    # for row in train_data_arr:
    #     # Convert row to a torch tensor and move it to the device
    #     X_tensor = torch.tensor(row.astype(np.float32)).to(device)
    #     X.append(X_tensor)
    #     print("here")
    #     label = train_df["y_true"][i]
    #     y_tensor = torch.tensor(label).to(device)  # Move label to device
    #     y.append(y_tensor)
    #     i+=0

    endTime = time.time()
    encoding_time_diff = (endTime - startTime) / 60
    logger.info(f"Total time taken to parse data: {encoding_time_diff:.2f} min")

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    logger.info("Initializing the XGBoost classifier...")
    
    # Set XGBoost parameters
    params = {
        "eta": learningRate,  # Learning rate
        "objective": "multi:softmax",  # Multi-class classification
        "num_class": len(set(y)),  # Number of classes
        "max_depth": 16, 
        "eval_metric": ["merror"],  # Logarithmic loss and error rate
        # "tree_method" :"gpu_hist"
    }

    # Initialize the XGBClassifier with the defined parameters
    xgb_model = xgb.XGBClassifier(
        objective=params["objective"],              # Learning objective 
        num_class=params["num_class"],              # Number of classes 
        eval_metric=params["eval_metric"],          # Evaluation metric (e.g., merror, mlogloss)
        use_label_encoder=False,                    # Avoid label encoder warnings
        #  Hyper parameters
        n_estimators= 200,                            # Number of boosting rounds
        min_child_weight= 1,                          # Minimum sum of instance weight (hessian) needed in a child
        subsample = 1,                                # Subsample ratio of training instances
        max_depth = 10,                               # Depth of the trees 
        learning_rate = 0.1,                          # Learning rate 
        gamma = 0,                                    # Minimum loss reduction to partition
        reg_lambda = 1,                               # L2 regularization term on weights
        reg_alpha = 1,                                # L1 regularization term on weights
        # tree_method="gpu_hist"
    )  

    # Log hyperparameter values
    logger.info("XGBoost Model Hyperparameters:")
    logger.info(f"n_estimators: {xgb_model.n_estimators}")
    logger.info(f"min_child_weight: {xgb_model.min_child_weight}")
    logger.info(f"subsample: {xgb_model.subsample}")
    logger.info(f"max_depth: {xgb_model.max_depth}")
    logger.info(f"learning_rate: {xgb_model.learning_rate}")
    logger.info(f"gamma: {xgb_model.gamma}")
    logger.info(f"reg_lambda: {xgb_model.reg_lambda}")
    logger.info(f"reg_alpha: {xgb_model.reg_alpha}") 

    logger.info("Training the model...")
    
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

    logger.info("Evaluating the model...")

    # # Make predictions on validation set
    # y_pred = xgb_model.predict(X_val)
    # accuracy = accuracy_score(y_val, y_pred)
    # logger.info(f"Validation Accuracy: {accuracy:.4f}")

    # Make predictions on the validation set
    y_val_pred = xgb_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')  # Macro average for balanced data

    # Make predictions on the training set
    y_train_pred = xgb_model.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')

    # Log both accuracies
    logger.info(f"Training Accuracy: {training_accuracy:.4f}")
    logger.info(f"Validation Accuracy: {validation_accuracy:.4f}")

    # Log or print the F1 scores
    logger.info(f"Training F1 Score: {train_f1:.4f}")
    logger.info(f"Validation F1 Score: {val_f1:.4f}")
    
    logger.info("Saving the model...")
    xgb_model.save_model(newModelPath)

    endTime = time.time()
    
    memory = psutil.Process().memory_info()

    logger.info(
        "Total time taken to train the model: {:.2f} min".format(
            (endTime - startTime) / 60
        )
    )

    logger.info(f"Memory usage: {memory}")
    logger.info(f"Model saved at: {newModelPath}")

    # Plot training validation losses vs. epochs if eval_results are available
    eval_results = xgb_model.evals_result()

    # train_losses = eval_results['validation_0']['mlogloss']
    # val_losses = eval_results['validation_1']['mlogloss']
    # train_losses = eval_results['validation_0']['mlogloss']     #train
    # val_losses = eval_results['validation_1']['mlogloss']       #test
    train_errors = eval_results['validation_0']['merror']
    val_errors = eval_results['validation_1']['merror']

    epoch_list = [e + 1 for e in range(len(train_errors))]
    

    plt.plot(epoch_list, train_errors, label="Training loss")
    plt.plot(epoch_list, val_errors, label="Validation loss")
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

    logger.info("Process complete.")

if __name__ == "__main__":
    main()
