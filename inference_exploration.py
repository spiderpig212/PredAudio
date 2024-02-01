import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import glob
import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import multiprocessing
import json
import time

import pandas as pd
from functools import partial
from tqdm.notebook import trange, tqdm
from sklearn.decomposition import PCA
# import umap
# import umap.plot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
# sys.path.insert(0, os.path.join(os.path.expanduser('~/Research/MyRepos/'),'SensoryMotorPred'))
# from datasets import WCDataset, WCShotgunDataset, WC3dDataset
import PreCNet as PN
from datasets import AudioDataset

import SETTINGS

import plotly.express as px
import plotly.graph_objects as go

import matplotlib as mpl
mpl.rcParams.update({'font.size':         24,
                     'axes.linewidth':    3,
                     'xtick.major.size':  5,
                     'xtick.major.width': 2,
                     'ytick.major.size':  5,
                     'ytick.major.width': 2,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'font.sans-serif':  "Arial",
                     'font.family':      "sans-serif",
                    })


########## Checks if path exists, if not then creates directory ##########
def check_path(basepath, path):
    if path in basepath:
        return basepath
    elif not os.path.exists(os.path.join(basepath, path)):
        os.makedirs(os.path.join(basepath, path))
        print('Added Directory:' + os.path.join(basepath, path))
        return os.path.join(basepath, path)
    else:
        return os.path.join(basepath, path)

dataRootDir = SETTINGS.DATA_PATH
projectRootDir = SETTINGS.PROJECT_PATH

# Set up partial functions for directory managing
join = partial(os.path.join, dataRootDir)
checkDir = partial(check_path, dataRootDir)
FigurePath = checkDir('Figures')

savefigs = False

# %matplotlib widget
#%%
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Trial    = 8
TimeSize = 7

output_mode = 'error'
save_path   = join('results/Tau01/NotStateful')
fileList    = sorted(glob.glob(os.path.join(save_path, 'PreCNetGRU_T{:03d}_N{:02d}_E*_Tau01_Visual_Netparams.json'.format(TimeSize, Trial))))
filename    = fileList[0]
Netpath = sorted(glob.glob(os.path.join(save_path, 'PreCNetGRU_T{:03d}_N{:02d}_E*_Tau01_Visual*.pt'.format(TimeSize, Trial))))
print(filename)
if len(fileList) > 0 and os.path.exists(fileList[0]):
    with open(fileList[0], 'r') as fp:
        netparams = json.load(fp)

# netparams['Train_paths'] = os.path.expanduser('~/Desktop/Research/Murray/git/PredAudio/Specs_train_mfcc.npy')
# netparams['Test_paths'] = os.path.expanduser('~/Desktop/Research/Murray/git/PredAudio/Specs_test_mfcc.npy')
netparams['Train_paths'] = join("soundFiles/Specs_train.npy")
netparams['Test_paths'] = join("soundFiles/Specs_test.npy")

netparams['save_path'] = save_path

#%%
if netparams['data_format'] == 'channels_first':
    input_shape = (netparams['BatchSize'], netparams['TimeSize'], 1, netparams['height'], netparams['width'])
else:
    input_shape = (netparams['BatchSize'], netparams['TimeSize'], netparams['height'], netparams['width'], 1)

precnet, netparams = PN.getNetwork(netparams)
params = torch.load(Netpath[0], map_location=torch.device('cpu'))  # Need to add map location cpu for load to work on cpu only machine
precnet.load_state_dict(params['Model_state_dict'])
precnet.to(device)
print('Loaded Network')

#%% Load test data
tr_loss = 0.0
sum_trainLoss_in_epoch = 0.0
min_trainLoss_in_epoch = float('inf')

########## Create Datasets and DataLoaders ##########
Dataset_Train = AudioDataset(netparams['Train_paths'], netparams['WindSize'], netparams['Overlap'])
Dataset_Test = AudioDataset(netparams['Test_paths'], netparams['WindSize'], netparams['Overlap'])
num_workers = multiprocessing.cpu_count()//2
DataLoader_Train = DataLoader(Dataset_Train, batch_size=netparams['BatchSize'], shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
DataLoader_Test = DataLoader(Dataset_Test, batch_size=netparams['BatchSize'], shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)

#%% Run model and display state representations
initial_states = precnet.get_initial_states(input_shape)
states = initial_states
h_te = {'h{:d}'.format(n): [] for n in range(len(netparams['R_stack_sizes']))}
with torch.no_grad():
    for batch in tqdm(DataLoader_Test):
        output, states, frame_pred = precnet.grab_states(batch.to(device), states, grab_frame=True)
        for n, state in enumerate(states[:len(netparams['R_stack_sizes'])]):
            h_te['h{:d}'.format(n)].append(output['h{:d}'.format(n)])
        initial_states = precnet.get_initial_states(input_shape)
        states = initial_states
for n in range(len(netparams['R_stack_sizes'])):
     h_te['h{:d}'.format(n)] = np.stack(h_te['h{:d}'.format(n)]).transpose(0, 2, 1, 3, 4, 5).reshape((-1, netparams['TimeSize'],) + h_te['h{:d}'.format(n)][0].shape[-3:])

#%% Load the truths to compare our predictions to
tot_train = len(DataLoader_Train)*netparams['BatchSize']
tot_test = len(DataLoader_Test)*netparams['BatchSize']
# labels_test = pd.read_csv(os.path.join(rootdir,'labels_test_mfcc.csv'))
labels_test = pd.read_csv(join('soundFiles/labels_test.csv'))
labels_test = labels_test.iloc[:tot_test]
# labels_train = pd.read_csv(os.path.join(rootdir,'labels_train_mfcc.csv'))
labels_train = pd.read_csv(join('soundFiles/labels_train.csv'))
labels_train = labels_train.iloc[:tot_train]

FM_Pred = np.stack(frame_pred).transpose(1,0,2).reshape(netparams['BatchSize'], netparams['TimeSize'], netparams['height'], netparams['width'])

FM_Actual = batch.squeeze().numpy()
GT = np.squeeze(batch.detach().numpy())

#%% Grab data to visualize
total_batches = GT.shape[0]
predicted_stacks = np.empty((total_batches, 132, 8))
actual_stacks = np.empty((total_batches, 132, 8))
previous_stacks = np.empty((total_batches, 132, 8))
GT_Label = np.empty(total_batches)

for n in range(total_batches):
    # n = 63  # Item to grab for ground truth and prediction
    plt.rcParams['figure.facecolor'] = 'white'
    # (64 channels, 7 time steps, 128 freqs, 16 segments in the time step [fed first 8 segements, predicts next 8 segments])
    fmGT = torch.FloatTensor(GT[n]).unsqueeze(1)  # The make_grid requires a 4D-tensor, so we unsqueeze to have the grayscale channel back
    fmP = torch.FloatTensor(FM_Pred[n]).unsqueeze(1)  # Should be (batch, color_channels, height, width)
    GT_Grid = torchvision.utils.make_grid(fmGT[:, :, :, 8:], nrow=netparams['TimeSize'])  # Plot from 8 on to avoid plotting overlap in the images
    P_Grid = torchvision.utils.make_grid(fmP[:, :, :, 8:], nrow=netparams['TimeSize'])  #
    # GT_Grid = torchvision.utils.make_grid(fmGT[:, :, :, :], nrow=netparams['TimeSize'])
    # P_Grid = torchvision.utils.make_grid(fmP[:, :, :, :], nrow=netparams['TimeSize'])
    past_truth = GT_Grid[0, :, 2:10]
    current_truth = GT_Grid[0, :, 12:20]
    current_prediction = P_Grid[0, :, 12:20]

    GT_Label[n] = labels_train['digits'][n]

    previous_stacks[n] = past_truth.numpy()
    actual_stacks[n] = current_truth.numpy()
    predicted_stacks[n] = current_prediction.numpy()

    # MSE_Past = ((past_truth - current_prediction)**2)/1
    # MSE_Current = ((current_truth - current_prediction)**2)/1

    # Print truth vs prediction
    # Each chunk is a time step (hence 7 chunks)
    #
    # fig1, axs = plt.subplots(2, 1, figsize=(20, 10))
    # axs[0].imshow(GT_Grid[0, :, 1:21], aspect='auto',  origin='lower') #cmap='gray'
    # axs[0].axis('off')
    # axs[0].set_title('Ground Truth, Digit: {:d}'.format(labels_train['digits'][n]))
    # axs[1].imshow(P_Grid[0, :, 1:21],  aspect='auto',  origin='lower') #cmap='gray'
    # axs[1].axis('off')
    # axs[1].set_title('Prediction')
    # plt.tight_layout()
    # plt.show()
    #
    # fig2, axs = plt.subplots(2, 1, figsize=(20, 10))
    # axs[0].imshow(MSE_Past, aspect='auto',  origin='lower') #cmap='gray'
    # axs[0].axis('off')
    # axs[0].set_title(f"MSE Past , Digit: {labels_train['digits'][n]}, MSE: {MSE_Past.mean():.4f}")
    # axs[1].imshow(MSE_Current,  aspect='auto',  origin='lower') #cmap='gray'
    # axs[1].axis('off')
    # axs[1].set_title(f"MSE Truth , Digit: {labels_train['digits'][n]:}, MSE: {MSE_Current.mean():.4f}")
    # plt.tight_layout()
    # plt.show()

pred_data_frame = pd.DataFrame({
    "number": GT_Label,
    "past_truth": previous_stacks.tolist(),
    "current_truth": actual_stacks.tolist(),
    "current_prediction": predicted_stacks.tolist(),
}, dtype="object")

nums = pred_data_frame.number.unique()

for num in nums:
    filtered_frame = pred_data_frame[pred_data_frame["number"] == num]

    frame_size = filtered_frame.shape[0]

    prev_stack = np.concatenate(filtered_frame["past_truth"].tolist()).reshape((frame_size, 132, 8))
    curr_stack = np.concatenate(filtered_frame["current_truth"].tolist()).reshape((frame_size, 132, 8))
    pred_stack = np.concatenate(filtered_frame["current_prediction"].tolist()).reshape((frame_size, 132, 8))

    MSE_Past = ((prev_stack - pred_stack)**2)/frame_size
    MSE_Present = ((curr_stack - pred_stack)**2)/frame_size

    fig2, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0].imshow(MSE_Past.mean(axis=0), aspect='auto',  origin='lower') #cmap='gray'
    axs[0].axis('off')
    axs[0].set_title(f"MSE Past, Digit: {num}, MSE: {MSE_Past.mean():.4f}")
    axs[1].imshow(MSE_Present.mean(axis=0),  aspect='auto',  origin='lower') #cmap='gray'
    axs[1].axis('off')
    axs[1].set_title(f"MSE Truth, Digit: {num}, MSE: {MSE_Present.mean():.4f}")
    plt.tight_layout()
    plt.show()

