from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        super().__init__()
        self.data = data
        if mode == "train":
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                 tv.transforms.RandomHorizontalFlip(0.15),
                                                 tv.transforms.RandomVerticalFlip(0.15),
                                                    tv.transforms.ToTensor(),
                                                    tv.transforms.Normalize(train_mean, train_std),
                                                 ])
        else:
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                     tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(train_mean, train_std)
                                                     ])


    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        assert False

    def __len__(self):
            return len(self.data)

    def __getitem__(self, index):
        index += self.data.first_valid_index()
        path = self.data['filename'][index]
        img = gray2rgb(imread(path))
        # swap axes so shape has the correct format
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        img_trans = self.transform(torch.from_numpy(img))

        return img_trans, torch.tensor([self.data.crack[index], self.data.inactive[index]], dtype=torch.float)
