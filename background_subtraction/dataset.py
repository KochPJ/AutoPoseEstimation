"""The SegmenationTrainingDataset is used to load the training data during the training of a model facing a
segmenation problem.
The SegmenationTrainingDataset is used to load the test data during the evaluation of a model facing a
segmenation problem.
"""
from torch.utils.data.dataset import Dataset
from background_subtraction.utils import *


class SegmentationDataset(Dataset):
    def __init__(self,  mode, root, dirs, classes, mean=None, std=None, show_plots=False):
        super(SegmentationDataset, self).__init__()

        self.root = root
        self.classes = classes
        self.dirs = dirs
        self.show_plots = show_plots
        self.n_samples = len(self.dirs[list(self.dirs.keys())[0]])

        self.n_classes = 2
        self.Resize = transforms.Resize([480, 640])
        if mode == 'train':
            self.rotation = transforms.functional.rotate
            self.hflip = transforms.functional.hflip
            self.vflip = transforms.functional.vflip
            self.ColorJitter = transforms.ColorJitter(brightness=0.05,
                                                      contrast=0.05,
                                                      saturation=0.05,
                                                      hue=0.02)
        else:
            self.rotation = None
            self.hflip = None
            self.vflip = None
            self.ColorJitter = None

        self.toTensor = transforms.ToTensor()

        if not mean or not std:
            print('__________________________________________________')
            print('getting mean and std')
            self.mean = []
            self.std = []
            for key in self.dirs:
                for idx in range(23):
                    x, _ = load_subtraction(self.root, key, idx)
                    l = x.shape[2]
                    x = self.toTensor(x)
                    self.mean.append([torch.mean(x[:, :, i]).numpy() for i in range(l)])
                    self.std.append([torch.std(x[:, :, i]).numpy() for i in range(l)])

            self.mean = list(np.mean(np.array(self.mean), axis=0))
            self.std = list(np.mean(np.array(self.std), axis=0))
            print('mean: {}\n std: {}'.format(self.mean, self.std))

        else:
            self.mean = mean
            self.std = std
        self.normalize = transforms.Normalize(self.mean, self.std)

    def __getitem__(self, index):

        cls = index//self.n_samples
        key = list(self.classes)[cls]
        idx = int(index - (cls * self.n_samples))

        x, y = load_subtraction(self.root,
                                key,
                                idx,
                                resize=self.Resize,
                                rotate=self.rotation,
                                colorJitter=self.ColorJitter,
                                hflip=self.hflip,
                                vflip=self.vflip,
                                plot=self.show_plots)

        y[y != 0] = 1

        # to tensor
        x = self.toTensor(x)
        y = torch.from_numpy(np.array(y, dtype=np.float64))
        y = y.long()

        # normalize
        x = self.normalize(x)

        return x, y

    def __len__(self):
        return int(self.n_samples*len(self.dirs.keys()))

