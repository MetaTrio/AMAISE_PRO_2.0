# import torch
# import pickle

# # model = '/home/chadmi.20/models_backup/human_train_kraken_and_nano_len_7k_12k_model_binary_ffnn_2mer_3mer.pkl'
# # Load the complete model with all details
# with open('/home/chadmi.20/models_backup/human_train_kraken_and_nano_len_7k_12k_model_binary_ffnn_2mer_3mer.pkl', 'rb') as f:
#     model = pickle.load(f)

# model.eval()  # Set the model to evaluation model



import torch
import torch.nn as nn
from initialFFNN import *

model_path = '/home/chadmi.20/models_backup/modelInitialFFNN_2mer_3mer.pth'

model = torch.load(model_path)  # Load the model object
model.eval()

model = nn.DataParallel(FeedForwardNN())
model.load_state_dict(torch.load(preModelPath, device))