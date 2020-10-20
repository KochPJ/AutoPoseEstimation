"""The SegmenationTrainingDataset is used to load the training data during the training of a model facing a
segmenation problem.
The SegmenationTrainingDataset is used to load the test data during the evaluation of a model facing a
segmenation problem.
"""
from torch.utils.data.dataset import Dataset
import os
from pathlib import Path
from segmentation.utils import *
import matplotlib.pyplot as plt
import copy


class SegmentationDataset(Dataset):
    def __init__(self,  data_set_name, mode, mean=None, std=None, label_mode='pred', plot=False):
        super(SegmentationDataset, self).__init__()

        self.classes = []
        self.dirs = []

        pkg_path = Path(__file__).resolve().parent.parent
        self.root = os.path.join(pkg_path, 'data_generation', 'data')
        self.label_root = os.path.join(pkg_path, 'label_generator', 'data')
        self.label_mode = label_mode
        self.mode = mode
        self.plot = plot
        path = os.path.join(pkg_path,
                            'label_generator',
                            'data_sets',
                            'segmentation',
                            data_set_name,
                            '{}_data_list.txt'.format(mode))
        input_file = open(path)
        while 1:
            input_line = input_file.readline()[:-1]
            if not input_line:
                break

            self.dirs.append(input_line)

        path = os.path.join(pkg_path,
                            'label_generator',
                            'data_sets',
                            'segmentation',
                            data_set_name,
                            'classes.txt')

        input_file = open(path)
        while 1:
            input_line = input_file.readline()[:-1]
            if not input_line:
                break

            self.classes.append(input_line)

        self.n_classes = len(self.classes)+1
        self.labels = []
        for d in self.dirs:
            found = False
            for i, cls in enumerate(self.classes):
                if cls in d:
                    found = True
                    self.labels.append(i+1)
                    break
            if not found:
                raise ValueError

        if not std or not mean:
            print('compute mean and std')
            self.toTensor = transforms.ToTensor()
            self.mean = []
            self.std = []
            for d in self.dirs:
                image = Image.open('{0}/{1}.color.png'.format(self.root, d))
                image = self.toTensor(image)
                self.mean.append([torch.mean(image[:, :, i]).numpy() for i in range(3)])
                self.std.append([torch.std(image[:, :, i]).numpy() for i in range(3)])


            self.mean = list(np.mean(np.array(self.mean), axis=0))
            self.std = list(np.mean(np.array(self.std), axis=0))
            print('mean = {}'.format(self.mean))
            print('std = {}'.format(self.std))
        else:
            self.mean = mean
            self.std = std

        if mode == 'train':
            self.augmentations = transforms.Compose([colorJitter(),
                                                     rotate(),
                                                     CropAndZoom()])
        else:
            self.augmentations = None

        self.preprocess = transforms.Compose([toTensor(),
                                              normalize(self.mean, self.std)])

    def __getitem__(self, index):
        img = Image.open('{0}/{1}.color.png'.format(self.root, self.dirs[index]))
        #depth = Image.open('{0}/{1}.depth.png'.format(self.root, self.dirs[index]))
        label = Image.open('{0}/{1}.{2}.label.png'.format(self.label_root, self.dirs[index], self.label_mode))

        if self.mode == 'train':
            img, label = self.augmentations([img, label])

        label = np.array(label)
        label[label != 0] = self.labels[index]
        if self.plot:
            self.plot_data(img, label, index)

        img, label = self.preprocess([img, label])
        return img, label

    def __len__(self):
        return len(self.dirs)


    def plot_data(self, image, label, index):
        label = np.array(copy.deepcopy(label), dtype=np.uint8)
        plt.cla()
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(image, dtype=np.uint8))
        plt.title('Image {}'.format(index))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(label)
        cls_id = self.labels[index]
        plt.title('Label: {}, {}'.format(cls_id, self.classes[cls_id-1]))
        plt.axis('off')

        added = np.array(copy.deepcopy(image), dtype=np.float)
        red = np.zeros(added.shape)
        red[:, :, 0] = 255
        #print(np.unique(label))
        added[label != 0] = added[label != 0] * 0.7 + red[label != 0] * 0.3
        added[added < 0] = 0
        added[added > 255] = 255
        added = np.array(added, dtype=np.uint8)
        plt.subplot(1, 3, 3)
        plt.imshow(added)
        plt.title('Added')
        plt.axis('off')
        plt.show()