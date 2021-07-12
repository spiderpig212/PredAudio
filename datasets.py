import numpy as np
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_path, wind_size=8, overlap=4, transform=None):
        
        self.data = np.load(data_path)
        self.wind_size = wind_size
        self.transform = transform
        self.overlap = overlap

    def __len__(self):
        return(self.data.shape[0])
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = np.lib.stride_tricks.sliding_window_view(self.data[idx,0],window_shape=self.wind_size, axis=-1)[:,::self.overlap,:].transpose(1,0,2).copy()

        # sample = np.zeros((self.data.shape[-1]//self.wind_size,self.data.shape[-2],self.wind_size))
        # for n in range(self.data.shape[-1]//self.wind_size):
        #     sample[n] = self.data[idx,:,:,(n*self.wind_size):(n*self.wind_size+self.wind_size)]
        
        sample = (sample - np.min(sample))/(np.max(sample)-np.min(sample))
        sample = np.clip(sample, 2e-30, 1).astype(np.float32)
        sample = torch.Tensor(sample[:,np.newaxis])
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


