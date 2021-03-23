import numpy as np
import torch
import torch.utils.data as data

class Coefficient_dataset(data.Dataset):

    def __init__(self, filename, is_train=True):
        super(Coefficient_dataset, self).__init__()
        self.filename = filename
        self.is_train = is_train

        self.data = np.load(self.filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        output = self.data[index, :]
        output = (output).astype(np.float32)*10

        return torch.from_numpy(output)
