import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler


'''
GOAL: label is the Class column, features are the other columns, ignore Case #

RESOURCES
https://discuss.pytorch.org/t/data-processing-as-a-batch-way/14154
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

'''

# Create dummy csv data
nb_samples = 110
a = np.arange(nb_samples)
df = pd.DataFrame(a, columns=['data'])
df.to_csv('data.csv', index=False)

# Create Dataset
class CSVDataset(Dataset):
    def __init__(self, path, chunksize, nb_samples):
        self.path = path
        self.chunksize = chunksize
        self.len = nb_samples / self.chunksize

    def __getitem__(self, index):
        x = next(
            pd.read_csv(
                self.path,
                skiprows=index * self.chunksize + 1,  #+1, since we skip the header
                chunksize=self.chunksize,
                names=['data']))
        x = torch.from_numpy(x.data.values)
        return x

    def __len__(self):
        return self.len


dataset = CSVDataset('data.csv', chunksize=10, nb_samples=nb_samples)
loader = DataLoader(dataset, batch_size=10, num_workers=1, shuffle=False)

