from torch.utils.data import Dataset

class dam_data(Dataset):
    def __init__(self, x_train=None, y_train=None, train=True, transforms=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transforms = transforms
        self.train = train

    def __getitem__(self, idx):
        x = self.x_train[idx]

        if self.transforms:
            x = self.transforms(x)

        if self.train:
            y = self.y_train[idx]
            return x, y

        else:
            return x

    def __len__(self):
        return len(self.x_train)