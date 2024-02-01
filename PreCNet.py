import argparse
import glob
import json
import multiprocessing
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from numba import njit
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from datasets import AudioDataset


# from apex import amp


def arg_parser(): 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--BatchSize', type=int, default=32)
    parser.add_argument('--Comp', type=int, default=0)
    parser.add_argument('--source_path', type=str,
                        default='~/Desktop/Research/Murray/git/PredAudio/',
                        help='Path to load files from')
    parser.add_argument('--outputmode', type=str, default='prediction',
                        help='Chose a loss function: error, prediction')
    parser.add_argument('--Tau', type=int, default=1,
                        help='Tau is how many frames in the future to predict')
    parser.add_argument('--UseMotor', type=int, default=0,
                        help='Integrate motor commands in network')
    parser.add_argument('--Nepochs', type=int,default=200,
                       help='Number of epochs (pass through dataset)')
    parser.add_argument('--Trial', type=int, default=99,
                        help='Trial Number')
    parser.add_argument('--RNG', type=int, default=1132,
                        help='Random Seed')
    parser.add_argument('--MotorLayer', type=int, default=0,
                    help='Motor Layer Input')
    parser.add_argument('--ArrayID',type=int,default=0)
    parser.add_argument('--TotTrials',type=int,default=4)
    parser.add_argument('--TotMotorLayer',type=int,default=4)
    parser.add_argument('--Talapas',type=int,default=0)
    parser.add_argument('--Stateful',type=int,default=0)
    parser.add_argument('--Save_Grad',type=int,default=1)
    parser.add_argument('--Apex',type=int,default=1)

    args = parser.parse_args()
    return args

##### Utility Functions #####
def hard_sigmoid(x):
    """
    Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    """
    x = (0.2 * x) + 0.5
    x = F.threshold(-x, -1, -1)
    x = F.threshold(-x, 0, 0)
    return x

def batch_flatten(x):
    """
    equal to the `batch_flatten` in keras.
    x is a Variable in pytorch
    """
    shape = [*x.size()]
    dim = np.prod(shape[1:])
    dim = int(dim)     
    return x.view(-1, dim)

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

########## Define Logging class to create csv document with errors ##########
class log:
    def __init__(self, f, name="", PRINT=True, retrain=False):
        text = ""
        if type(name) == list:
            text = "{}".format(name[0])
            for x in name[1:]:
                text += ",{}".format(x)
        elif type(text) == str:
            text = name

        self.FNAME = f
        if retrain:
            F = open(self.FNAME, "a")
        else: 
            F = open(self.FNAME, "w+")
        if len(text) != 0:
            F.write(text + "\n")
        F.close()
        if PRINT:
            print(text)

    def log(self, data, PRINT=True):
        text = ""
        if type(data) == list:
            text = "{}".format(data[0])
            for x in data[1:]:
                text += ",{}".format(x) 
        elif type(data) == str:
            text = data
        if PRINT:
            print(text)
        F = open(self.FNAME, "a")
        F.write(str(text) + "\n")
        F.close()

########## Create Figure for Tensorboard ##########
def plot_imshow(batch, frame_pred, netparams):
    ##### GT Vs Pred frames #####
    frame_pred = np.stack(frame_pred).transpose(1, 0, 2).reshape(netparams['BatchSize'], netparams['TimeSize'], netparams['height'], netparams['width'])
    GT = np.squeeze(batch.detach().numpy())
    fmGT = torch.FloatTensor(GT[0]).unsqueeze(1)
    fmP = torch.FloatTensor(frame_pred[0]).unsqueeze(1)
    GT_Grid = torchvision.utils.make_grid(fmGT, nrow=netparams['TimeSize'])
    P_Grid = torchvision.utils.make_grid(fmP, nrow=netparams['TimeSize'])

    fig1, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0].imshow(GT_Grid[0], aspect='auto',  origin='lower')  # cmap='gray'
    axs[1].imshow(P_Grid[0],  aspect='auto',  origin='lower')  # cmap='gray'
    axs[0].axis('off')
    axs[1].axis('off')
    plt.tight_layout()

    ##### Delta t Plot #####
    diffrange = 3
    diffloss = parallel_diff(frame_pred.astype(float), GT.astype(float), diffrange=diffrange)
    STD = np.nanstd(diffloss, axis=0)
    DiffAvg = np.nanmean(diffloss, axis=0)
    fig2, ax2 = plt.subplots(1, figsize=(5, 5))
    x = np.arange(-diffrange, diffrange+1)+1
    ax2.plot(x, DiffAvg, 'k', linewidth=3)
    ax2.fill_between(x, DiffAvg-STD, DiffAvg+STD, alpha=.5, color='k')
    plt.tight_layout()

    # torch.cat((GT_Grid,P_Grid),dim=1)
    return fig1, fig2

# Calculate the error between the prediction and actual fram separated by delta t
@njit(parallel=True)
def parallel_diff(FM_Pred, FM_Actual, diffrange=10):
    BatchSize, tlength, width, height = FM_Actual.shape
    # Assumes FM_Pred and FM_Actual have same dimensions.
    assert FM_Pred.shape == FM_Actual.shape, 'Dimensions must be equal'
    # Diff in frame loss
    diffloss = np.zeros((tlength, diffrange*2+1))
    diffloss[:] = np.nan
    
    for fm in range(tlength):# trange(0,tlength, desc='Frame Num',leave=False): #
        for ind, tau in enumerate(np.arange(-diffrange, diffrange+1)):
            if ((fm+tau) <= (tlength-tau)) & ((fm+tau) >= 0):
                diffloss[fm, ind] = np.nanmean(np.abs(FM_Pred[:, fm, :, :] - FM_Actual[:, fm+tau, :, :]))
    return diffloss

class PreCNet(nn.Module):
    def __init__(self, stack_sizes, R_stack_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1., error_activation='relu', stateful=True,
                 GRU_activation='tanh', GRU_inner_activation='hard_sigmoid',
                 output_mode='error', extrap_start_time=None, data_format='channels_first',
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=.003, optimizer='Adam', **kwargs):
        super(PreCNet, self).__init__()
        self.stack_sizes            = stack_sizes
        self.nb_layers              = len(stack_sizes)
        assert len(R_stack_sizes)   == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes          = R_stack_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes        = Ahat_filt_sizes
        assert len(R_filt_sizes)    == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes           = R_filt_sizes
        self.num_layers             = len(stack_sizes)
        self.pixel_max              = pixel_max
        self.stateful               = stateful
        
        self.optimizer = optimizer
        self.lr = lr
        default_output_modes = ['prediction', 'error', 'all']
        layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['Rd', 'Ed', 'Ad', 'Ahatd','Ru', 'Eu', 'Au', 'Ahatu']]
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None
        self.extrap_start_time = extrap_start_time

        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {channels_last, channels_first}'
        self.data_format = data_format
        self.channel_axis = -3 if data_format == 'channels_first' else -1
        self.row_axis = -2 if data_format == 'channels_first' else -3
        self.column_axis = -1 if data_format == 'channels_first' else -2
        self.device = device
        self.build()
    
    def get_initial_states(self, input_shape):
        '''
        input_shape is like: (batch_size, timeSteps, Height, Width, 3)
                         or: (batch_size, timeSteps, 3, Height, Width)
        '''
        init_height = input_shape[self.row_axis]     # equal to `init_nb_rows` in original version
        init_width  = input_shape[self.column_axis]     # equal to `init_nb_cols` in original version

        base_initial_state = np.zeros(input_shape)
        non_channel_axis = -1 if self.data_format == 'channels_first' else -2
        for _ in range(2):
            base_initial_state = np.sum(base_initial_state, axis=non_channel_axis)
        base_initial_state = np.sum(base_initial_state, axis=1)   # (batch_size, 3)

        initial_states = []
        states_to_pass = ['R', 'E']    # R is `representation`, c is Cell state in GRU, E is `error`.
        layerNum_to_pass = {sta: self.num_layers for sta in states_to_pass}
        if self.extrap_start_time is not None:
            states_to_pass.append('Ahat')   # pass prediction in states so can use as actual for t+1 when extrapolating
            layerNum_to_pass['Ahat'] = 1

        for sta in states_to_pass:
            for lay in range(layerNum_to_pass[sta]):
                downSample_factor = 2 ** lay            
                row = init_height // downSample_factor
                col = init_width  // downSample_factor
                if sta in ['R']:
                    stack_size = self.R_stack_sizes[lay]
                elif sta == 'E':
                    stack_size = self.stack_sizes[lay] * 2
                elif sta == 'Ahat':
                    stack_size = self.stack_sizes[lay]
                output_size = stack_size * row * col    # flattened size
                reducer = np.zeros((input_shape[self.channel_axis], output_size))   # (3, output_size)
                initial_state = np.dot(base_initial_state, reducer)                 # (batch_size, output_size)

                if self.data_format == 'channels_first':
                    output_shape = (-1, stack_size, row, col)
                else:
                    output_shape = (-1, row, col, stack_size)
                # initial_state = torch.from_numpy(np.reshape(initial_state, output_shape)).float().to(device)
                initial_state = Variable(torch.from_numpy(np.reshape(initial_state, output_shape)).float().to(self.device), requires_grad = True)
                initial_states += [initial_state]

        if self.extrap_start_time is not None:
            initial_states += [Variable(torch.IntTensor(1).zero_().to(self.device))]   # the last state will correspond to the current timestep

        return initial_states

        
    def build(self):
        self.conv_layers = {c: [] for c in ['hd', 'zd', 'od', 'hu', 'zu', 'ou', 'ahat']}
        self.parms = []
        for c in sorted(self.conv_layers.keys()):
            for l in range(self.nb_layers):
                if c == 'ahat':
                    nb_channels = self.R_stack_sizes[l]
                    self.conv_layers['ahat'].append(nn.Conv2d(in_channels  = nb_channels, 
                                                              out_channels = self.stack_sizes[l],
                                                              stride       = (1, 1),
                                                              kernel_size  = self.Ahat_filt_sizes[l],
                                                              padding      =(-1 + self.Ahat_filt_sizes[l]) // 2)
                    )
                elif c in ['hd', 'zd', 'od']:
                    if l == self.nb_layers-1:
                        nb_channels = 2 * self.stack_sizes[l] + self.R_stack_sizes[l]
                    else:
                        nb_channels = 2 * self.stack_sizes[l+1] + self.R_stack_sizes[l]
                    self.conv_layers[c].append(nn.Conv2d(in_channels  = nb_channels,
                                                         out_channels = self.R_stack_sizes[l],
                                                         kernel_size  = self.R_filt_sizes[l],
                                                         stride       = (1, 1),
                                                         padding      = (-1 + self.R_filt_sizes[l]) // 2)
                                                         )
                
                elif c in ['hu', 'zu', 'ou']:
                    nb_channels = 2 * self.stack_sizes[l] + self.R_stack_sizes[l]
                    if l < self.nb_layers - 1:
                        self.conv_layers[c].append(nn.Conv2d(in_channels  = nb_channels,
                                                             out_channels = self.R_stack_sizes[l],
                                                             kernel_size  = self.R_filt_sizes[l],
                                                             stride       = (1, 1),
                                                             padding      = (-1 + self.R_filt_sizes[l]) // 2)
                                                             )

        for name, layerList in self.conv_layers.items():
            self.conv_layers[name] = nn.ModuleList(layerList)
            setattr(self, name, self.conv_layers[name])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        if self.extrap_start_time is not None:
            self.t_extrap = Variable(torch.IntTensor(1).zero_().to(self.device), requires_grad=False)

    def step(self, a, states):
        h_tm1 = states[:self.nb_layers]
        e_tm1 = states[self.nb_layers:2*self.nb_layers]
        
        if self.extrap_start_time is not None:
            t = states[-1]
        
        a0 = a[:]

        h = []
        e = []
        ahat_list=[]

        ########## Update R units starting from the top ##########
        for l in reversed(range(self.nb_layers)):
            if l == self.nb_layers - 1:
                inputs = [h_tm1[l], e_tm1[l]]
            else: 
                inputs = [h_tm1[l], ed]
                
            inputs = torch.cat(inputs, dim=self.channel_axis)
            if not isinstance(inputs, Variable):
                inputs = Variable(inputs, requires_grad=True)

            z = hard_sigmoid(self.conv_layers['zd'][l](inputs))
            o = hard_sigmoid(self.conv_layers['od'][l](inputs))
            _o = torch.tanh(o * self.conv_layers['hd'][l](inputs))
            _h = (1-z) * h_tm1[l] + z * _o
            h.insert(0, _h)

            ahat = self.conv_layers['ahat'][l](h[0])
            if l == 0:
                ahat[ahat > self.pixel_max] = self.pixel_max        # passed through a saturating non-linearity set at the maximum pixel value
                frame_prediction = ahat
            ahat_list.insert(0, ahat)
            
            if l > 0:
                a = self.pool(h_tm1[l-1])
            else:
                if self.extrap_start_time is not None:
                    if t >= self.t_extrap:
                        a = ahat
                    else:
                        a = a0
                else:
                    a = a0
            
            ########## compute errors ##########
            e_up = F.relu(ahat - a)
            e_down = F.relu(a - ahat)

            e.insert(0, torch.cat((e_up, e_down), dim=self.channel_axis))

            if l > 0:
                ed = self.upsample(e[0])
            
            if self.output_layer_num == l:
                if self.output_layer_type == 'Ad':
                    output = a
                elif self.output_layer_type == 'Ahatd':
                    output = ahat
                elif self.output_layer_type == 'Hd':
                    output = h[l]
                elif self.output_layer_type == 'Ed':
                    output = e[l]

        ########## Update feedforward path starting from the bottom ##########
        for l in range(self.nb_layers):
            if l == 0:
                pass
            else:
                a = self.pool(h[l-1])
                ahat = ahat_list[l]
                e_up = F.relu(ahat - a)
                e_down = F.relu(a - ahat)
                e[l] = torch.cat((e_up, e_down), axis=self.channel_axis)

            if l < self.nb_layers - 1:
                inputs = [h[l], e[l]]
                inputs = torch.cat(inputs, dim=self.channel_axis)
                if not isinstance(inputs, Variable):
                    inputs = Variable(inputs, requires_grad=True)

                z = hard_sigmoid(self.conv_layers['zu'][l](inputs))
                o = hard_sigmoid(self.conv_layers['ou'][l](inputs))
                _o = torch.tanh(o * self.conv_layers['hu'][l](inputs))
                _h = (1-z) * h[l] + z * _o
                h[l] = _h

            if self.output_layer_num == l:
                if self.output_layer_type == 'Au':
                    output = a
                elif self.output_layer_type == 'Ahatu':
                    output = ahat
                elif self.output_layer_type == 'Hu':
                    output = h[l]
                elif self.output_layer_type == 'Eu':
                    output = e[l]

        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction
            else:
                for l in range(self.num_layers):
                    layer_error = torch.mean(batch_flatten(e[l]), dim=-1, keepdim=True)
                    all_error = layer_error if l == 0 else torch.cat((all_error, layer_error), dim=-1)
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = torch.cat((batch_flatten(frame_prediction), all_error), dim=-1)

        states = h + e
        return output, states, batch_flatten(frame_prediction)


    def forward(self, A0_withTimeStep, initial_states, grab_frame=False):
        num_timesteps = A0_withTimeStep.size()[1]
        hidden_states = initial_states    
        
        output_list = [] 
        frame_pred_list = [] 
        for t in range(num_timesteps):
            A0 = A0_withTimeStep[:, t]
            output, hidden_states, frame_pred = self.step(A0, hidden_states)
            output_list.append(output)
            if grab_frame:
                frame_pred_list.append(frame_pred.detach().cpu().numpy())
        hidden_states = [hidden_state.detach() for hidden_state in hidden_states]
        if self.output_mode == 'error':
            return output_list, hidden_states, frame_pred_list
        elif self.output_mode == 'prediction':
            return output_list, hidden_states, frame_pred_list
        elif self.output_mode == 'all':
            pass
        else:
            raise(RuntimeError('Unknown output mode!'))
    
    def grab_states(self, A0_withTimeStep, initial_states, grab_frame=False):
        num_timesteps = A0_withTimeStep.size()[1]
        hidden_states = initial_states    
        
        ht = {'h{:d}'.format(n): [] for n in range(len(self.R_stack_sizes))}
        frame_pred_list = [] 
        with torch.no_grad():
            for t in range(num_timesteps):
                A0 = A0_withTimeStep[:, t]
                _, hidden_states, frame_pred = self.step(A0, hidden_states)
                output = [hidden_state.detach().clone().cpu().numpy() for hidden_state in hidden_states]
                for n in range(len(self.R_stack_sizes)):
                    ht['h{:d}'.format(n)].append(output[n])
                if grab_frame:  
                    frame_pred_list.append(frame_pred.detach().cpu().numpy())
            hidden_states = [hidden_state.detach() for hidden_state in hidden_states]
            for key in ht.keys():
                ht[key] = np.stack(ht[key])

            return ht, hidden_states, frame_pred_list
           

########## Dataset Generation ##########
def GenerateDatasetsContiuous(Frame_Train, Train_Names, Frame_Test, Test_Names, netparams):
    """ Generates dataset that is continuous along batch dimensions"""

    BTime = np.round(netparams['VidLength']/netparams['TimeSize']).astype(int)
    # Create Dataset from Numpy Array, With Motor Commands
    Frame_Train2 = Frame_Train.reshape(-1, BTime, netparams['TimeSize'], netparams['width'], netparams['height'],1)
    
    Vrange_Train = np.arange(0, len(netparams['Train_Names'])+netparams['BatchSize'], netparams['BatchSize'])
    n = 0
    FM_Train, Ag_Train = [], []
    for num in range(len(Vrange_Train)-1):
        while n < BTime:
            FM_Train.append(Frame_Train2[Vrange_Train[num]:Vrange_Train[num+1], n, :, :, :])
            n += 1
        n = 0
        num += 1
    FM_Train = np.vstack(FM_Train)
    
    Frame_Test2 = Frame_Test.reshape(-1, BTime, netparams['TimeSize'], netparams['width'], netparams['height'], 1)
    Vrange_Test = np.arange(0, len(netparams['Test_Names'])+netparams['BatchSize'],netparams['BatchSize'])
    n=0
    FM_Test, Ag_Test = [], []
    for num in range(len(Vrange_Test)-1):
        while n < BTime:
            FM_Test.append(Frame_Test2[Vrange_Test[num]:Vrange_Test[num+1], n, :, :, :])
            n += 1
        n = 0
        num += 1
    FM_Test = np.vstack(FM_Test)
    print('Orig. Train Dataset Shape: ', Frame_Train.shape, 'Orig. Test Dataset Shape: ', Frame_Test.shape)
    print('Cont. Train Dataset Shape: ', FM_Train.shape,    'Cont. Test Dataset Shape: ', FM_Test.shape)
    return FM_Train, FM_Test


########## Function to get and initialze network ##########
def getNetwork(netparams):
    # Create Model Object
    model = PreCNet(netparams['stack_sizes'], netparams['R_stack_sizes'],
                    netparams['Ahat_filt_sizes'], netparams['R_filt_sizes'], 
                    pixel_max=1, output_mode=netparams['output_mode'], return_sequences=True,)    
    if netparams['use_motor']:
        netparams['filename'] = 'PreCNetGRU_T{:03d}_N{:02d}_E{:04d}_Tau{:02d}_Motor'.format(
            netparams['TimeSize'],netparams['Trial'], netparams['Nepochs'], netparams['Tau'])
    else:
        netparams['filename'] = 'PreCNetGRU_T{:03d}_N{:02d}_E{:04d}_Tau{:02d}_Visual'.format(
            netparams['TimeSize'], netparams['Trial'], netparams['Nepochs'], netparams['Tau'])
    if netparams['Stateful']:
        netparams['save_path'] = check_path(netparams['save_path'], os.path.join('Tau{:02d}'.format(netparams['Tau']),'Stateful'))
    else:
        netparams['save_path'] = check_path(netparams['save_path'], os.path.join('Tau{:02d}'.format(netparams['Tau']),'NotStateful'))

    return model, netparams

def main(args):
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Trial      = 8    # Trial number
    n_channels = 1
    img_height = 128
    img_width  = 16
    stack1, stack2, stack3, stack4 = (4, 16, 64, 256)  # (32, 64, 128, 256) #
    stack_sizes       = (n_channels, stack1, stack2, stack3)
    R_stack_sizes     = (stack1, stack2, stack3, stack4)
    FiltSizes         = 3
    Ahat_filt_sizes   = tuple([FiltSizes for _ in range(len(stack_sizes))])  # (FiltSizes, FiltSizes, FiltSizes, FiltSizes)
    R_filt_sizes      = tuple([FiltSizes for _ in range(len(stack_sizes))])  # (FiltSizes, FiltSizes, FiltSizes, FiltSizes)

    BatchSize   = 64    # Size of minibatches
    TestSize    = 37    # Size of the test dataset, # of videos
    TimeSize    = 7    # How many recurrent time steps to take 
    WindSize    = 16
    Overlap     = 8
    Nepochs     = 1500   # Number of epochs (full run through the dataset)
    Tau         = 1     # Number steps to predict ##### Only implemented Tau =1 for now
    Stateful    = 0     # Whether or not to keep state from previous batch ##### Need To Implement, detach hiddenstates
    Save_Grad   = 0     # Save Gradient  ##### Need To Implement 
    output_mode = 'error' # Type of output, 'error', 'prediction', 'all' 
    rootdir     = os.path.expanduser(args.source_path)
    save_path   = check_path(rootdir, 'results')
    fileList    = sorted(glob.glob(os.path.join(save_path, '*Netparams.json')))

    # Set random seed
    # RNG = Trial+100 
    # random.seed(RNG)

    # ########## Load Parameters to keep consistent across trials ##########
    # if (len(fileList)>0) & (os.path.exists(fileList[0])):
    #     with open(fileList[0], 'r') as fp:
    #         netparams = json.load(fp)
    #     Train_Names = netparams['Train_Names']
    #     Test_Names  = netparams['Test_Names']
    #     assert Test_Names == netparams['Test_Names'], 'Test Names are not equal'
    #     print('Restored Train List')
    #     Frame_Train, Agent_Train, Train_Names, Frame_Test,Agent_Test, Test_Names = GD.GrabFramesFromVid([source_path],netparams)
    # else:
    #     Frame_Train, Agent_Train, Train_Names, Frame_Test,Agent_Test, Test_Names = GD.GrabFramesFromVid([source_path],TestSize=TestSize)

    ########## Define Network Parameters ##########
    netparams = {'width': img_width,
                 'height': img_height,
                 'BatchSize': int(BatchSize),
                 'TimeSize': int(TimeSize),
                 'WindSize': int(WindSize),
                 'Overlap': int(Overlap),
                 'Tau': int(Tau),
                 'FiltNum': stack_sizes[1],
                 'KSize': R_filt_sizes[0],  # Filter size for Recurrent Layer
                 'stack_sizes': stack_sizes,
                 'R_stack_sizes': R_stack_sizes,
                 'Ahat_filt_sizes': Ahat_filt_sizes,
                 'R_filt_sizes': R_filt_sizes,
                 'layer_loss_weightsMode': 'L_0',
                 'lr': 0.005,
                 'Nepochs': int(Nepochs),
                 'output_mode': output_mode,
                 'data_format': 'channels_first',
                 'ImageSize': img_width*img_height,
                 'log_freq': 10,
                 # 'source_path': source_path,
                 'save_path': save_path,
                 'input_shape': (BatchSize, TimeSize, img_height, img_width, 1),
                 'Train_paths': os.path.join(rootdir, 'Specs_train.npy'),
                 'Test_paths': os.path.join(rootdir, 'Specs_test.npy'),
                 'Trial': int(Trial),
                 'use_motor': 0,
                 'Stateful': int(Stateful),
                 'Save_Grad': Save_Grad,
                 }

    ########## Input Shape for building Network ##########
    if netparams['data_format'] == 'channels_first':
        input_shape = (netparams['BatchSize'], netparams['TimeSize'], 1, netparams['height'], netparams['width'])
    else:
        input_shape = (netparams['BatchSize'], netparams['TimeSize'], netparams['height'], netparams['width'], 1)

    # torch.autograd.set_detect_anomaly(True)

    ########## Create Network ##########
    precnet, netparams = getNetwork(netparams)
    precnet.to(device)

    ########## Preprocess Images ##########
    # FM_Train, FM_Test = GenerateDatasetsContiuous(Frame_Train, Train_Names, Frame_Test, Test_Names, netparams)
    # FM_Train = np.clip(FM_Train.transpose(0,1,4,2,3) / 255, 2e-30, 1).astype(float32) # Clip and channel first
    # FM_Test  = np.clip(FM_Test.transpose(0,1,4,2,3) / 255, 2e-30, 1).astype(float32)

    ########## Create Datasets and DataLoaders ##########
    Dataset_Train = AudioDataset(netparams['Train_paths'], netparams['WindSize'], netparams['Overlap'])
    Dataset_Test = AudioDataset(netparams['Test_paths'], netparams['WindSize'], netparams['Overlap'])
    num_workers = multiprocessing.cpu_count()//2
    DataLoader_Train = DataLoader(Dataset_Train, batch_size=netparams['BatchSize'], shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)
    DataLoader_Test = DataLoader(Dataset_Test, batch_size=netparams['BatchSize'], shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(precnet.parameters(), lr=netparams['lr'])

    ########## Use Apex float16 support for larger batchsize ##########
    # if args.Apex:
    #     precnet, optimizer = amp.initialize(precnet, optimizer, opt_level='O2')

    ########## Set up learning rate schedule ##########
    lr_maker = lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.5)  #0.5  decay the lr every step_size epochs by a factor of gamma

    ########## Initialize Logger and Tensorboard Writer ##########
    logf = log(os.path.join(netparams['save_path'], netparams['filename']+'_log.csv'), name=['Epoch', 'Train_Loss', 'Test_Loss'])
    logGrad = log(os.path.join(netparams['save_path'], netparams['filename']+'_log_Gradient.csv'), name=['_'.join(name.split('.')) for name, p in precnet.named_parameters()])
    netparams['LogDir'] = check_path(netparams['save_path'],'Logs/Trial_{:02d}'.format(Trial))
    writer = SummaryWriter(netparams['LogDir'])

    ########## Pre-save json in case of error ##########
    with open(os.path.join(netparams['save_path'], netparams['filename']+'_Netparams.json'), 'w') as fp:
        json.dump(netparams, fp, sort_keys=True, indent=4)

    ##### Epoch Training Loop #####
    totsteps = 0
    min_trainLoss_in_epoch = float('inf')
    for Epoch in range(netparams['Nepochs']+1):
        tr_loss = 0.0
        sum_trainLoss_in_epoch = 0.0
        sum_testLoss_in_epoch = 0.0
        startTime_epoch = time.time()

        initial_states = precnet.get_initial_states(input_shape)
        states = initial_states
        ##### Gradient Step Loop #####
        for step, batch in enumerate(DataLoader_Train):
            mini_batch = batch.to(device)
            if (step==(len(DataLoader_Train)-1)):
                output, states, frame_pred = precnet(mini_batch, states, grab_frame=True)
                GTVPred, DiffPlot = plot_imshow(batch, frame_pred, netparams)
                writer.add_figure('Actual vs. Predicted', GTVPred, global_step=Epoch)
                writer.add_figure('Delta t Plot', DiffPlot, global_step=Epoch)
                plt.close(GTVPred)
                plt.close(DiffPlot)
            else:
                output, states, _ = precnet(mini_batch, states)

            num_layer = len(netparams['stack_sizes'])
            # weighting for each layer in final loss
            if netparams['layer_loss_weightsMode'] == 'L_0':        # e.g., [1., 0., 0., 0.]
                layer_weights = np.array([0. for _ in range(num_layer)])
                layer_weights[0] = 1.
                layer_weights = torch.from_numpy(layer_weights)
            elif netparams['layer_loss_weightsMode'] == 'L_all':    # e.g., [1., 1., 1., 1.]
                layer_weights = np.array([0.1 for _ in range(num_layer)])
                layer_weights[0] = 1.
                layer_weights = torch.from_numpy(layer_weights)
            else:
                raise(RuntimeError('Unknown loss weighting mode! Please use `L_0` or `L_all`.'))
            layer_weights = Variable(layer_weights.float().to(device))

            # Weighting Loss for each time step of RNN
            num_timeSteps = netparams['TimeSize']
            time_loss_weight = (1. / (num_timeSteps - 1))
            time_loss_weight = Variable(torch.from_numpy(np.array([time_loss_weight])).float().to(device))
            time_loss_weights = [time_loss_weight for _ in range(num_timeSteps - 1)]
            time_loss_weights.insert(0, Variable(torch.from_numpy(np.array([0.])).float().to(device)))

            # Compute Batch Loss
            error_list = [batch_x_numLayer__error * layer_weights for batch_x_numLayer__error in output] # Layer weights
            error_list = [error_at_t.sum() for error_at_t in error_list] # Sum across layer
            total_error = error_list[0] * time_loss_weights[0]  # Time Weights
            for err, time_weight in zip(error_list[1:], time_loss_weights[1:]):
                total_error = torch.cat((total_error, err * time_weight), axis=0)

            loss = total_error.mean()
            optimizer.zero_grad()
    #         if args.Apex:
    #             with amp.scale_loss(loss, optimizer) as scaled_loss:                      
    #                 scaled_loss.backward()
    #         else:
            loss.backward()
            optimizer.step()

            # Save Loss, Iterate Total Step
            tr_loss = loss.item()
            sum_trainLoss_in_epoch += loss.item()
            totsteps += 1

            ##### Log Gradients #####
            total_norm = 0; 
            gradnorms = []
            for name, p in precnet.named_parameters():
                param_norm = p.grad.data.norm(2).item()
                gradnorms.append(param_norm)
                total_norm += param_norm ** 2
            total_norm = total_norm ** (1. / 2)
            logGrad.log(gradnorms, PRINT=False)

            ##### Log Step Loss and Frame Prediction #####
            writer.add_scalar('Loss/Train_Loss', tr_loss, global_step=totsteps)

            ##### Reset States when new video #####
    #         if (netparams['Stateful'] == 1)  & ((step % (netparams['VidLength'] // netparams['TimeSize']))==0)  & (step>0): # 
            states = precnet.get_initial_states(input_shape)

        ########## Calculate Test Loss ##########
        with torch.no_grad():
            initial_states = precnet.get_initial_states(input_shape)
            states = initial_states
            for step, batch in enumerate(DataLoader_Test):
                mini_batch = batch.to(device)
                output, states, _ = precnet(mini_batch, states)

                error_list = [batch_x_numLayer__error * layer_weights for batch_x_numLayer__error in output] # Layer weights
                error_list = [error_at_t.sum() for error_at_t in error_list]  # Sum across layer
                total_error = error_list[0] * time_loss_weights[0]  # Time Weights
                for err, time_weight in zip(error_list[1:], time_loss_weights[1:]):
                    total_error = torch.cat((total_error, err * time_weight), axis=0)

                Test_loss = total_error.mean()
                sum_testLoss_in_epoch += Test_loss.item()
                # if (netparams['Stateful'] == 1)  & ((step % (netparams['VidLength'] // netparams['TimeSize']))==0)  & (step>0): # 
                states = precnet.get_initial_states(input_shape)

        lr_maker.step() # Iterate Learning Rate Scheduler
        endTime_epoch    = time.time()
        EpochTime        = endTime_epoch - startTime_epoch
        Train_Epoch_Loss = sum_trainLoss_in_epoch/len(DataLoader_Train)
        Test_Epoch_Loss  = sum_testLoss_in_epoch/len(DataLoader_Test)
        print('Epoch: {:04d} Epoch_Time: {:.3f} Train_Epoch_Loss: {:.5f}  Test_Epoch_Loss: {:.5f}'.format(Epoch, EpochTime, Train_Epoch_Loss, Test_Epoch_Loss))

        ########## log data ##########
        logf.log([Epoch, Train_Epoch_Loss, Test_Epoch_Loss], PRINT=False)
        writer.add_scalar('Loss/Train_Epoch_Loss', Train_Epoch_Loss, Epoch)
        writer.add_scalar('Loss/Test_Epoch_Loss', Test_Epoch_Loss, Epoch)

        ##### Save Model #####
        if sum_testLoss_in_epoch < min_trainLoss_in_epoch:
            min_trainLoss_in_epoch = sum_testLoss_in_epoch
            torch.save({'Model_state_dict': precnet.state_dict(),
                        'Optimizer_state_dict': optimizer.state_dict(),
                        'Epoch': Epoch,
                        'Loss': loss},
                        os.path.join(netparams['save_path'], netparams['filename']+'.pt'))
            print('Saved New Model')

    ##### Close Tensorboard #####
    writer.flush()
    writer.close()


if __name__ == '__main__':
    args = arg_parser()
    main(args)
    
