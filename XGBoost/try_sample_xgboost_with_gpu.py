import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state = 0)

# Convert the data to DMatrix, which is a more efficient format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the parameters
params = {
    'objective': 'binary:logistic',  # for binary classification
    'eval_metric': 'logloss',  # evaluation metric
    'tree_method':"hist",
    'device': "cuda",
}

# Train the model on multiple GPUs
model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])

# Predict and evaluate
y_pred = model.predict(dtest)
y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
