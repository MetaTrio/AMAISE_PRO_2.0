from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import xgboost as xgb
import pandas as pd
import time
import logging
import click
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


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


@click.help_option("--help", "-h", help="Show this message and exit")
def main(input, labels, model, output):

    newModelPath = model
    inputset = input
    labelset = labels
    resultPath = output

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

    # logger.info(f"Learning rate: {learningRate}")
    # logger.info(f"Batch size: {batchSize}")

    # Load data
    train_df = pd.read_csv(labelset)
    train_data_arr = pd.read_csv(inputset, header=None).to_numpy()

    X, y = [], []

    logger.info("Parsing data...")

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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],                 # Number of boosting rounds
        'max_depth': [6, 10, 12],                        # Depth of trees
        'min_child_weight': [1, 5],                  # Minimum child weight
        # 'subsample': [0.8, 1.0],                         # Subsample ratio of training instances
        'learning_rate': [0.01, 0.1, 0.2],         # Learning rate
        # 'gamma': [0, 0.1, 0.5],                          # Minimum loss reduction to partition
        # 'reg_lambda': [1, 1.5],                          # L2 regularization term on weights
        # 'reg_alpha': [0.1, 0.5, 1]                       # L1 regularization term on weights
    }

    # Initialize the XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(y)))

    # Define the macro F1 score as the evaluation metric
    scorer = make_scorer(f1_score, average='macro')

    # Initialize GridSearchCV with the XGBoost classifier, hyperparameter grid, and macro F1 score
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,  # 3-fold cross-validation
        verbose=2,
        n_jobs=-1  # Use all available cores
    )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best F1 score
    best_params = grid_search.best_params_
    best_f1 = grid_search.best_score_

    # Log or print the best parameters and score
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best macro F1 score: {best_f1:.4f}")

    # Save the best model
    best_model = grid_search.best_estimator_
    best_model.save_model(newModelPath)

    # You can now evaluate the best model on the validation set as well:
    y_val_pred = best_model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    validation_accuracy = accuracy_score(y_val, y_val_pred)

    logger.info(f"Validation F1 Score with best model: {val_f1:.4f}")
    logger.info(f"Validation Accuracy with best model: {validation_accuracy:.4f}")

if __name__ == "__main__":
    main()
