import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
import segmentation_models_pytorch as smp
import os
from PIL import Image
import random
import copy
import cv2


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
    jacc_loss = (intersection / (union + eps))

    jacc_loss = jacc_loss[1:].mean()

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

def do_cca(predicted, cuda=True):
    predicted = F.softmax(predicted, dim=1)
    if cuda:
        predicted = predicted.cpu()
    predicted = predicted.numpy()
    new_pred = []
    for i, pred in enumerate(predicted):
        pred = pred.transpose(1, 2, 0)
        mask = np.array(np.argmax(pred, axis=2), dtype=np.uint8)
        mask2 = np.array(np.max(pred, axis=2))
        ret, labels = cv2.connectedComponents(mask, connectivity=8)
        biggest = 1
        biggest_score = 0
        for u in np.unique(labels)[1:]:
            score = np.sum(mask2[labels == u])
            if score > biggest_score:
                biggest_score = score
                biggest = u

        out = np.zeros(mask.shape)
        out[labels == biggest] = 1
        new_pred.append(np.expand_dims(out, axis=0))
    new_pred = np.concatenate(new_pred, axis=0)

    return new_pred

class IoU_cca(Metric):
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

        new_pred = do_cca(predicted)
        new_pred = torch.from_numpy(new_pred)

        self.conf_metric.add(new_pred.view(-1), target.view(-1))

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
                conf_matrix[:, index] = 0
                conf_matrix[index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou[1:])


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
                conf_matrix[:, index] = 0
                conf_matrix[index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou[1:])


def animate(i, fig, axs, path):

    with open(path) as json_file:
        config = json.load(json_file)

    plt.cla()
    fig.suptitle('Best mIou: {} in epoch: {}'.format(np.round(config['best_iou_score'], 3), config['best_iou_score_epoch']), fontsize=16)
    axs[0].plot(config['losses'], c='b')
    axs[0].set_title('jaccard loss')
    axs[0].set_ylabel('1-mIoU')
    axs[0].set_xlabel('Epochs')

    axs[1].plot(config['iou_scores'], c='r')
    axs[1].set_title('mean IoU')
    axs[1].set_ylabel('mIoU')
    axs[1].set_xlabel('Epochs')



nets = {'Unet': smp.Unet,
        'PsPNet': smp.PSPNet,
        'LinkNet': smp.Linknet}

def get_model(name, segmentation_config):
    model = nets[name]
    model = model(**segmentation_config)
    return model

def load_subtraction(root,
                     key,
                     idx,
                     resize=None,
                     rotate=None,
                     colorJitter=None,
                     hflip=None,
                     vflip=None,
                     plot=False,
                     abs=True):

    if plot:
        plt.clf()
        plt.cla()

    # make sure augmentation is for all the same
    angle = 0
    if rotate:
        angle = random.uniform(-180, 180)

    if hflip:
        if np.random.rand() <= 0.5:
            hflip = None

    if vflip:
        if np.random.rand() <= 0.5:
            vflip = None

    # get background and foreground, also augment them
    b = Image.open(os.path.join(root,
                                key,
                                'background',
                                'img{:06d}.png'.format(idx))
                   ).convert('RGB')
    b = augment(b,
                angle=angle,
                resize=resize,
                rotate=rotate,
                colorJitter=colorJitter,
                hflip=hflip,
                vflip=vflip)


    f = Image.open(os.path.join(root,
                                key,
                                'foreground',
                                'img{:06d}.png'.format(idx))
                   ).convert('RGB')

    if plot:
        plt.subplot(3, 4, 1)
        plt.imshow(np.array(f))
        plt.title('RGB foreground')
        plt.axis('off')

        plt.subplot(3, 4, 2)
        plt.imshow(np.array(copy.deepcopy(f).convert('HSV')))
        plt.title('HSV foreground')
        plt.axis('off')

    f = augment(f,
                angle=angle,
                resize=resize,
                rotate=rotate,
                colorJitter=colorJitter,
                hflip=hflip,
                vflip=vflip)

    # copy rgb images and convert them to hsv
    b_hsv = copy.deepcopy(b).convert('HSV')
    f_hsv = copy.deepcopy(f).convert('HSV')

    if plot:
        plt.subplot(3, 4, 5)
        plt.imshow(np.array(f))
        plt.title('RGB augmented foreground')
        plt.axis('off')

        plt.subplot(3, 4, 6)
        plt.imshow(np.array(f_hsv))
        plt.title('HSV augmented foreground')
        plt.axis('off')


    # load depth and augment them
    b_depth = Image.open(os.path.join(root,
                                      key,
                                      'background',
                                      'depth{:06d}.png'.format(idx))
                         )

    b_depth = augment(b_depth,
                      angle=angle,
                      resize=resize,
                      rotate=rotate,
                      colorJitter=None,
                      hflip=hflip,
                      vflip=vflip)

    f_depth = Image.open(os.path.join(root,
                                      key,
                                      'foreground',
                                      'depth{:06d}.png'.format(idx))
                         )

    if plot:
        plt.subplot(3, 4, 3)
        plt.imshow(np.array(f_depth))
        plt.title('Depth foreground')
        plt.axis('off')

    f_depth = augment(f_depth,
                      angle=angle,
                      resize=resize,
                      rotate=rotate,
                      colorJitter=None,
                      hflip=hflip,
                      vflip=vflip)

    if plot:
        plt.subplot(3, 4, 7)
        plt.imshow(np.array(f_depth))
        plt.title('Depth augmented foreground')
        plt.axis('off')


    # conver to numpy
    b = np.array(b, dtype=np.float)
    f = np.array(f, dtype=np.float)
    b_hsv = np.array(b_hsv, dtype=np.float)
    f_hsv = np.array(f_hsv, dtype=np.float)
    b_depth = np.array(b_depth, dtype=np.float)
    f_depth = np.array(f_depth, dtype=np.float)

    # eliminate measuring errors
    f_depth[b_depth == 0] = 0
    b_depth[f_depth == 0] = 0


    # subtract
    x = f - b
    x_hsv = f_hsv - b_hsv
    x_depth = f_depth - b_depth

    # take absolute difference if wanted
    if abs:
        x = np.abs(x)
        x_hsv = np.abs(x_hsv)
        x_depth = np.abs(x_depth)


    if plot:
        plt.subplot(3, 4, 9)
        plt.imshow(np.array(x, dtype=np.uint8))
        plt.title('RGB subtracted')
        plt.axis('off')

        plt.subplot(3, 4, 10)
        plt.imshow(np.array(x_hsv, dtype=np.uint8))
        plt.title('HSV subtracted')
        plt.axis('off')

        plt.subplot(3, 4, 11)
        plt.imshow(np.array(x_depth, dtype=np.uint8))
        plt.title('Depth subtracted')
        plt.axis('off')


    #  concatenate channels
    x = np.concatenate((x, x_hsv), axis=2)
    x = np.concatenate((x, np.expand_dims(x_depth, axis=2)), axis=2)

    # convert to uint8
    x = np.array(x, dtype=np.uint8)

    if plot:
        plt.subplot(3, 4, 12)
        plt.imshow(np.array(np.sum(x, axis=2)/x.shape[2], dtype=np.uint8))
        plt.title('Average summed difference')
        plt.axis('off')


    # load label and augment like the image data
    y = Image.open(os.path.join(root,
                                key,
                                'groundtruth',
                                'img{:06d}.mask.0.png'.format(idx))
                   )
    if plot:
        plt.subplot(3, 4, 4)
        plt.imshow(np.array(y))
        plt.title('Label')
        plt.axis('off')

    y = augment(y,
                angle=angle,
                resize=resize,
                rotate=rotate,
                colorJitter=None,
                hflip=hflip,
                vflip=vflip)
    if plot:
        plt.subplot(3, 4, 8)
        plt.imshow(np.array(y))
        plt.title('Label')
        plt.axis('off')

        plt.show()

    # convert to float
    y = np.array(y, dtype=np.float)

    return x, y


def augment(x, angle=0, resize=None, rotate=None, colorJitter=None, hflip=None, vflip=None):

    if resize:
        x = resize(x)

    if rotate:
        x = rotate(x, angle)

    if colorJitter:
        x = colorJitter(x)

    if hflip:
        x = hflip(x)

    if vflip:
        x = vflip(x)

    return x

def get_default_model(root):
    segmentation_config = {'encoder_name': 'resnet34',
                           'encoder_weights': None,
                           'activation': 'softmax',
                           'in_channels': 7,
                           'classes': 2}
    name = 'Unet'
    model = get_model(name, segmentation_config)

    cp = torch.load(os.path.join(root,
                                 'background_subtraction',
                                 'trained_models',
                                 '{}_{}.ckpt'.format(name, segmentation_config['encoder_name'])))
    model.load_state_dict(cp['state_dict'])

    return model


def get_mask_prediction(object_name, root, mean=None, std=None, reference_point=np.array([]), plot=False,
                        use_cuda=True):

    if std is None:
        std = [0.059689723, 0.05965291, 0.056203008, 0.05619316, 0.054657422, 0.054514673, 0.05377024]
    if mean is None:
        mean = [0.040278014, 0.04060352, 0.038310923, 0.0381776, 0.03656849, 0.03636289, 0.03556486]

    object_path = os.path.join(root, 'data_generation/data', object_name)
    dirs = os.listdir(object_path)
    background_path = os.path.join(object_path, 'background')
    n = int(len(os.listdir(background_path)) / 3)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)

    if torch.cuda.is_available() and use_cuda:
        device = torch.device('cuda:0')
        cuda = True
    else:
        cuda = False
        device = torch.device('cpu')

    model = get_default_model(root)
    model.to(device)
    model.eval()

    try:
        i = dirs.index('background')
        del dirs[i]
    except:
        raise ValueError('background does not exist in object_path: {}'.format(object_path))

    try:
        i = dirs.index('extra')
        del dirs[i]
    except:
        pass

    if len(dirs) < 1:
        raise ValueError('no foreground')

    ns = n * len(dirs)
    counter = 0
    for d in dirs:
        foreground_path = os.path.join(object_path, d)
        save_dir = os.path.join(root, 'label_generator/data', object_name, d)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        indexes = list(range(n))
        # indexes = [32,33]

        for idx in indexes:
            counter += 1
            print('number = {}/{}'.format(counter, ns))  # prints the progress in the terminal
            # load for the given index the background, object frame and ground truth

            with open(os.path.join(background_path, '{:06d}.color.png'.format(idx)), 'rb') as f:
                b_rgb = np.array(Image.open(f).convert('RGB'), dtype=np.float)

            with open(os.path.join(foreground_path, '{:06d}.color.png'.format(idx)), 'rb') as f:
                f_rgb = np.array(Image.open(f).convert('RGB'), dtype=np.float)

            with open(os.path.join(background_path, '{:06d}.color.png'.format(idx)), 'rb') as f:
                b_hsv = np.array(Image.open(f).convert('RGB').convert('HSV'), dtype=np.float)

            with open(os.path.join(foreground_path, '{:06d}.color.png'.format(idx)), 'rb') as f:
                f_hsv = np.array(Image.open(f).convert('RGB').convert('HSV'), dtype=np.float)

            with open(os.path.join(background_path, '{:06d}.depth.png'.format(idx)), 'rb') as f:
                b_depth = np.array(Image.open(f), dtype=np.float)

            with open(os.path.join(foreground_path, '{:06d}.depth.png'.format(idx)), 'rb') as f:
                f_depth = np.array(Image.open(f), dtype=np.float)

            if reference_point != np.array([]):
                with open(os.path.join(foreground_path, '{:06d}.meta.json'.format(idx))) as f:

                    meta = json.load(f)
                    robot2endEff_tf = np.array(meta.get('robot2endEff_tf')).reshape(4,4)
                    hand_eye_calibration = np.array(meta.get('hand_eye_calibration')).reshape(4, 4)

                    robot2cam = np.dot(robot2endEff_tf, hand_eye_calibration)
                    pos = robot2cam[:3, 3]
                    measure_dist = np.linalg.norm(reference_point-pos)
            else:
                measure_dist = None


            if not measure_dist:
                max_measure_dist = int(1500)  # 1.5 m
                min_measure_dist = 0
            else:
                max_measure_dist = measure_dist + 150
                min_measure_dist = measure_dist - 150

            f_depth[f_depth > max_measure_dist] = 0
            b_depth[b_depth > max_measure_dist] = 0
            f_depth[f_depth < min_measure_dist] = 0
            b_depth[b_depth < min_measure_dist] = 0

            # eliminate measuring errors
            f_depth[b_depth == 0] = 0
            b_depth[f_depth == 0] = 0

            # subtract
            x = f_rgb - b_rgb
            x_hsv = f_hsv - b_hsv
            x_depth = f_depth - b_depth


            # take absolute difference if wanted
            x = np.abs(x)
            x_hsv = np.abs(x_hsv)
            x_depth = np.abs(x_depth)

            if plot:
                plt.cla()
                plt.subplot(3, 4, 1)
                plt.imshow(np.array(b_rgb, dtype=np.uint8))
                plt.title('RGB foreground')
                plt.axis('off')

                plt.subplot(3, 4, 2)
                plt.imshow(np.array(f_rgb, dtype=np.uint8))
                plt.title('RGB foreground')
                plt.axis('off')

                plt.subplot(3, 4, 3)
                plt.imshow(np.array(b_hsv, dtype=np.uint8))
                plt.title('HSV foreground')
                plt.axis('off')

                plt.subplot(3, 4, 4)
                plt.imshow(np.array(f_hsv, dtype=np.uint8))
                plt.title('HSV foreground')
                plt.axis('off')

                plt.subplot(3, 4, 5)
                plt.imshow(np.array(b_depth/max_measure_dist*255, dtype=np.uint8))
                plt.title('Depth background')
                plt.axis('off')

                plt.subplot(3, 4, 6)
                plt.imshow(np.array(f_depth/max_measure_dist*255, dtype=np.uint8))
                plt.title('Depth foreground')
                plt.axis('off')

                plt.subplot(3, 4, 7)
                plt.imshow(np.array(x, dtype=np.uint8))
                plt.title('RGB subtracted')
                plt.axis('off')

                plt.subplot(3, 4, 8)
                plt.imshow(np.array(x_hsv, dtype=np.uint8))
                plt.title('HSV subtracted')
                plt.axis('off')

                plt.subplot(3, 4, 9)
                plt.imshow(np.array(x_depth, dtype=np.uint8))
                plt.title('Depth subtracted')
                plt.axis('off')

            #  concatenate channels
            x = np.concatenate((x, x_hsv), axis=2)
            x = np.concatenate((x, np.expand_dims(x_depth, axis=2)), axis=2)

            # convert to uint8
            x = np.array(x, dtype=np.uint8)

            if plot:
                plt.subplot(3, 4, 10)
                plt.imshow(np.array(np.sum(x, axis=2) / x.shape[2], dtype=np.uint8))
                plt.title('Average summed difference')
                plt.axis('off')

            x = to_tensor(x)
            x = normalize(x)
            x = x.to(device)
            x = x.unsqueeze(0)

            pred = model.predict(x)
            pred_cca = do_cca(pred, cuda=cuda)[0]
            pred_cca[pred_cca != 0] = 255
            pred_cca = np.array(pred_cca, dtype=np.uint8)

            if plot:
                if cuda:
                    pred = pred.cpu()[0]
                else:
                    pred = pred[0]
                pred = torch.argmax(pred, dim=0).numpy()
                pred[pred != 0] = 255
                pred = np.array(pred, dtype=np.uint8)

                plt.subplot(3, 4, 11)
                plt.imshow(pred)
                plt.title('Predicted Label')
                plt.axis('off')

                plt.subplot(3, 4, 12)
                plt.imshow(pred_cca)
                plt.title('Predicted Label with CCA')
                plt.axis('off')
                plt.show()

            label = Image.fromarray(pred_cca)
            label.save(os.path.join(save_dir, '{:06d}.pred.label.png'.format(idx)))



