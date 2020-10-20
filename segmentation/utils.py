import numpy as np
import torch
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
import segmentation_models_pytorch as smp
import random
import copy
from PIL import Image

class HFlipDefault:
    def __init__(self):
        self.p = 0.5
        self.hflip = transforms.functional.hflip

    def __call__(self, data):
        img, label = data
        if np.random.rand() <= self.p:
            img = self.hflip(img)
            label = self.hflip(label)

        return [img, label]

class rotate:
    def __init__(self):
        self.rotation = transforms.functional.rotate
        self.range = [-180, 180]

    def __call__(self, data):
        image, label = data
        angle = random.uniform(self.range[0], self.range[1])
        image = self.rotation(image, angle)
        label = self.rotation(label, angle)
        return image, label

class colorJitter:
    def __init__(self):
        self.ColorJitter = transforms.ColorJitter(brightness=0.2,
                                                  contrast=0.2,
                                                  saturation=0.2,
                                                  hue=0.05)
    def __call__(self, data):
        img, label = data
        img = self.ColorJitter(img)
        return [img, label]

class normalize:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, data):
        img, label = data
        img = self.normalize(img)
        return [img, label]

class toTensor:
    def __init__(self):
        self.toTensor = transforms.ToTensor()

    def __call__(self, data):
        img, label = data
        img = self.toTensor(img)
        label = torch.from_numpy(np.array(label, dtype=np.float64))
        label = label.long()
        return [img, label]




def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    # unique = torch.unique(true)[1:]
    unique = torch.unique(true)
    #print('unique', unique)
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    #jacc_loss = (intersection / (union + eps)).mean()
    jacc_loss = (intersection / (union + eps))
    #print('jacc_loss', jacc_loss, jacc_loss.shape)
    #print(jacc_loss.mean(dim=1), jacc_loss.mean(dim=1).shape, jacc_loss.mean())
    jacc_loss = jacc_loss[unique]
    jacc_loss = jacc_loss.mean()
    #print('mean', jacc_loss)
    #print('out', (1 - jacc_loss))
    return (1 - jacc_loss)


class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf



class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.
        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou[1:])

def animate(i, fig, axs, path, mean_cca=True):


    with open(path) as json_file:
        config = json.load(json_file)

    plt.cla()
    fig.suptitle('Best mIou: {} in epoch: {}'.format(np.round(config['best_iou_score'], 4), config['best_iou_score_epoch']), fontsize=16)
    axs[0].plot(config['losses'], c='b')
    axs[0].set_title('jaccard loss')
    axs[0].set_ylabel('1-mIoU')
    axs[0].set_xlabel('Epochs')

    axs[1].plot(config['iou_scores'], c='r')
    axs[1].set_title('mean IoU')
    axs[1].set_ylabel('mIoU')
    axs[1].set_xlabel('Epochs')

    if mean_cca:
        axs[2].plot(config['iou_cca_scores'], c='r')
        axs[2].set_title('mean cca IoU')
        axs[2].set_ylabel('cca mIoU')
        axs[2].set_xlabel('Epochs')

def animate2(i, fig, axs, path):


    with open(path) as json_file:
        config = json.load(json_file)

    plt.cla()
    fig.suptitle('Best mIou: {} in epoch: {}'.format(np.round(config['best_iou_score'], 4), config['best_iou_score_epoch']), fontsize=16)
    axs[0, 0].plot(config['train_losses'], c='b')
    axs[0, 0].set_title('train jaccard loss')
    axs[0, 0].set_ylabel('1-mIoU')
    axs[0, 0].set_xlabel('Epochs')

    axs[0, 1].plot(config['train_iou_scores'], c='r')
    axs[0, 1].set_title('train mean IoU')
    axs[0, 1].set_ylabel('mIoU')
    axs[0, 1].set_xlabel('Epochs')

    axs[1, 0].plot(config['valid_losses'], c='b')
    axs[1, 0].set_title('valid jaccard loss')
    axs[1, 0].set_ylabel('1-mIoU')
    axs[1, 0].set_xlabel('Epochs')

    axs[1, 1].plot(config['valid_iou_scores'], c='r')
    axs[1, 1].set_title('valid mean IoU')
    axs[1, 1].set_ylabel('mIoU')
    axs[1, 1].set_xlabel('Epochs')



nets = {'Unet': smp.Unet,
        'PsPNet': smp.PSPNet,
        'LinkNet': smp.Linknet}

def get_model(name, segmentation_config):
    model = nets[name]
    model = model(**segmentation_config)
    return model

class CropAndZoom():
    def __init__(self):
        self.bbox_increase = 1.1
        self.to_small = 0.8
        self.to_big = 1.2
        self.size = False
        self.output_size = 480
        self.max_zoom = 2
        self.max_l = 480
        self.min_l = int(float(self.output_size) / self.max_zoom)

    def __call__(self, data):

        image_PIL, label_PIL = data
        label = np.array(copy.deepcopy(label_PIL))
        # get some variables
        self.size = label.shape  # height, width
        extreme_points = self.get_extreme_points(label)
        h, w, c = self.get_size(extreme_points)
        h_ratio = float(h) / float(self.output_size)
        w_ratio = float(w) / float(self.output_size)
        h_w_ratio = h_ratio / w_ratio
        ls = [h, w]
        bigger = 0
        if w_ratio > h_ratio:
            bigger = 1

        # create bbox
        bbox = self.get_bbox(c, ls[bigger] * self.bbox_increase)
        bbox = self.random_zoom(bbox)
        bbox_h, bbox_w, bbox_c = self.get_size(bbox)  # height = width

        # adapt bbox
        if h_w_ratio <= self.to_big and h_w_ratio >= self.to_small:
            # case: square
            if bbox_h <= self.size[0] and bbox_w <= self.size[0]:
                # if the bbox is not to big, ensure that it is inside the image
                bbox = self.move_bbox_inside(bbox)
            else:
                # if the box is to big, slide random along the horizontal axis of the bbox and then create it as big as
                # possible and ensure that it is inside the image
                bbox_c[1] = int(bbox_c[1] - (w / 2)) + np.random.randint(0, w)
                bbox = self.get_bbox(bbox_c, self.size[0] - 2)
                bbox = self.move_bbox_inside(bbox)
        else:
            # case: rectangular
            # slide the bbox randomly along the bigger axis
            bbox_c[bigger] = int(bbox_c[bigger] - (ls[bigger] / 2)) + np.random.randint(0, ls[bigger])
            bbox = self.get_bbox(bbox_c, bbox_h)
            bbox_h, bbox_w, bbox_c = self.get_size(bbox)  # height = width

            if bbox_h <= self.size[0] and bbox_w <= self.size[0]:
                # if the bbox is not to big, ensure that it is inside the image
                bbox = self.move_bbox_inside(bbox)
            else:
                bbox = self.get_bbox(bbox_c, self.size[0] - 2)
                bbox = self.move_bbox_inside(bbox)

        # create patch and set label id
        image_PIL = image_PIL.crop(box=[bbox[2], bbox[0], bbox[3], bbox[1]])
        label_PIL = label_PIL.crop(box=[bbox[2], bbox[0], bbox[3], bbox[1]])

        image_PIL = image_PIL.resize(size=(self.output_size, self.output_size))
        label_PIL = label_PIL.resize(size=(self.output_size, self.output_size), resample=Image.NEAREST)

        '''
            cv2.resize(image[bbox[0]:bbox[1], bbox[2]:bbox[3]], dsize=(self.output_size, self.output_size),
                       interpolation=cv2.INTER_NEAREST), dtype=np.float)
        target_out = np.array(
            cv2.resize(label[bbox[0]:bbox[1], bbox[2]:bbox[3]], dsize=(512, 512), interpolation=cv2.INTER_NEAREST),
            dtype=np.uint8)
        '''

        return [image_PIL, label_PIL]

    def resize_bbox_to_max_zoom(self, bbox):
        bbox_h, bbox_w, bbox_c = self.get_size(bbox)
        if bbox_h > self.max_l:
            bbox = self.get_bbox(bbox_c, self.max_l)
        elif bbox_h < self.min_l:
            bbox = self.get_bbox(bbox_c, self.min_l)
        return bbox

    def random_zoom(self, bbox):
        bbox_h, bbox_w, bbox_c = self.get_size(bbox)
        h = int(random.uniform(self.min_l, self.max_l))
        bbox = self.get_bbox(bbox_c, h)
        return bbox

    def get_extreme_points(self, label):
        label_pos = np.where(label == 255)
        label_x = label_pos[0]
        label_y = label_pos[1]
        arg_max_x = np.argmax(label_x)
        arg_max_y = np.argmax(label_y)
        arg_min_x = np.argmin(label_x)
        arg_min_y = np.argmin(label_y)
        extreme_points = np.array(
            [label_x[arg_min_x], label_x[arg_max_x],
             label_y[arg_min_y], label_y[arg_max_y]])  # used for plotting [up, down, left, right]

        return extreme_points

    def get_size(self, extreme_points):
        h = extreme_points[1] - extreme_points[0]
        w = extreme_points[3] - extreme_points[2]
        c = [extreme_points[0] + int(h / 2), extreme_points[2] + int(w / 2)]  # [height, width] (x,y)
        return h, w, c

    def get_bbox(self, c, l):
        half = int(l / 2)
        bbox = [c[0] - half, c[0] + half, c[1] - half, c[1] + half]  # [up, down, left, right]
        return bbox

    def move_bbox_inside(self, bbox):
        move = [0, 0]
        if bbox[0] < 0:
            move[0] = bbox[0]
        elif bbox[1] > self.size[0]:
            move[0] = bbox[1] - self.size[0]

        if bbox[2] < 0:
            move[1] = bbox[2]
        elif bbox[3] > self.size[1]:
            move[1] = bbox[3] - self.size[1]
        bbox = [bbox[0] - move[0], bbox[1] - move[0], bbox[2] - move[1], bbox[3] - move[1]]
        return bbox


