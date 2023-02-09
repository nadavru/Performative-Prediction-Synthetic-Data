from torch.utils.data import Dataset

class Supervised(Dataset):
    def __init__(self, x, y, n_examples, return_indexes=False):
        super().__init__()
        self.x = x
        self.y = y
        self.n_examples = n_examples
        self.return_indexes = return_indexes

    def __getitem__(self, ind):
        if self.return_indexes:
            return self.x[ind], self.y[ind], ind
        else:
            return self.x[ind], self.y[ind]

    def __len__(self):
        return self.n_examples

class Unsupervised(Dataset):
    def __init__(self, x, n_examples):
        super().__init__()
        self.x = x
        self.n_examples = n_examples

    def __getitem__(self, ind):
        return self.x[ind]

    def __len__(self):
        return self.n_examples
