import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from segmentation.dataset import SegmentationDataset
from segmentation.utils import jaccard_loss, IoU, animate
import numpy as np
import os
import json
from torch import nn
import matplotlib.pyplot as plt

nets = {'Unet': smp.Unet,
        'PsPNet': smp.PSPNet,
        'LinkNet': smp.Linknet}

class CP:
    def __init__(self):
        self.train_losses = []
        self.train_iou_scores = []
        self.valid_losses = []
        self.valid_iou_scores = []
        self.train_iou_per_class = []
        self.valid_iou_per_class = []



def segmentation_training(training_config, segmentation_config):

    print('create paths')
    save_path = os.path.join(Path(__file__).resolve().parent, 'trained_models', training_config['dataset_name'])
    logs_path = os.path.join(Path(__file__).resolve().parent, 'logs', training_config['dataset_name'])
    logs_images = os.path.join(logs_path, 'images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(logs_images):
        os.makedirs(logs_images)

    for img in os.listdir(logs_images):
        os.remove(os.path.join(logs_images, img))

    if segmentation_config['encoder_weights'] == 'imagenet':
        print('use imagenet mean and std')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif training_config['dataset_name'] == 'full_12_classes':
        print('use full_12_classes mean and std')
        mean = [0.7423757, 0.74199075, 0.7420199]
        std = [0.1662702, 0.16652738, 0.16721568]
    else:
        mean = None
        std = None

    print('create datasets')
    train_dataset = SegmentationDataset(training_config['dataset_name'],
                                        'train',
                                        mean=mean,
                                        std=std,
                                        plot=False)
    test_dataset = SegmentationDataset(training_config['dataset_name'],
                                       'test',
                                       mean=train_dataset.mean,
                                       std=train_dataset.std)
    segmentation_config['classes'] = train_dataset.n_classes

    print('create model')
    name = segmentation_config['name']
    del segmentation_config['name']
    model = utils.get_model(name, segmentation_config)
    if torch.cuda.device_count() > 1:
        multi_gpu = True
    else:
        multi_gpu = False

    if torch.cuda.is_available():
        pin_memory = True
        device = torch.device('cuda:0')
    else:
        pin_memory = False
        device = torch.device('cpu')

    if multi_gpu:
        model = nn.DataParallel(model)

    model.to(device)

    print('create optimizer, dataloader and metric')

    if training_config.get('optimizer') == 'Adam':
        print('use Adam optimizer: lr = {}'.format(training_config['lr']))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=training_config['lr'],
                                     weight_decay=training_config['weight_decay'])

    else:
        print('use SGD optimizer: lr = {}'.format(training_config['lr']))
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=training_config['lr'],
                                    momentum=training_config['momentum'],
                                    weight_decay=training_config['weight_decay'],
                                    nesterov=True)



    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=training_config['batch_size'],
                                                   shuffle=training_config['shuffle'],
                                                   num_workers=training_config['num_workers'],
                                                   pin_memory=pin_memory)


    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=training_config['batch_size'],
                                                  shuffle=training_config['shuffle'],
                                                  num_workers=training_config['num_workers'],
                                                  pin_memory = pin_memory)

    print('class names: {}'.format(train_dataset.classes))
    print('n classes: {}'.format(train_dataset.n_classes))
    print('n train batches: {}'.format(len(train_dataloader)))
    print('n valid batches: {}'.format(len(test_dataloader)))


    metric = IoU(num_classes=train_dataset.n_classes)

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=50, verbose=True, factor=0.1)

    best_iou_score = 0
    best_epoch = 0
    cp = CP()

    print('start training')
    for i in range(training_config['epochs']):
        print('__________________________________________________')
        print('Epoch {}/{}'.format(i, training_config['epochs']-1))
        currentloss = []
        model.train()
        metric.reset()
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = jaccard_loss(label, pred)
            currentloss.append(float(loss.data))
            metric.add(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iou_per_class, iou_score = metric.value()
        cp.train_losses.append(np.mean(currentloss))
        cp.train_iou_scores.append(iou_score)
        cp.train_iou_per_class.append(iou_per_class)
        print('train Loss: {}'.format(cp.train_losses[-1]))
        print('train mIoU: {}'.format(iou_score))
        per_class_mean = np.array(cp.train_iou_per_class)
        per_class_mean = np.round(np.mean(per_class_mean, axis=0), 3)
        print('train per class mIOU: {}'.format(per_class_mean))

        currentloss = []
        model.eval()
        metric.reset()
        first_valid = False
        with torch.no_grad():
            for img, label in test_dataloader:
                img = img.to(device)
                label = label.to(device)
                pred = model(img)
                loss = jaccard_loss(label, pred)
                currentloss.append(float(loss.data))
                metric.add(pred, label)

                if first_valid:
                    if torch.cuda.is_available():
                        pred = pred.cpu()
                        label = label.cpu()

                    pred = pred.numpy()
                    label = label.numpy()
                    plt.cla()
                    fig, axs = plt.subplots(2, 4, constrained_layout=True, figsize=(25, 12))
                    for j in range(4):
                        axs[0, j].imshow(label[j])
                        axs[0, j].set_title('label {}'.format(list(np.unique(label[j]))))
                        axs[0, j].set_axis_off()

                        axs[1, j].imshow(np.argmax(pred[j].transpose(1, 2, 0), axis=2))
                        axs[1, j].set_title('prediction {}'.format(list(np.unique(pred[j]))))
                        axs[1, j].set_axis_off()


                    fig.suptitle('epoch {}'.format(i), fontsize=16)
                    plt.savefig(os.path.join(logs_images, 'valid_{}.png'.format(i)))

                    first_valid = False

        iou_per_class, iou_score = metric.value()
        cp.valid_iou_scores.append(iou_score)
        cp.valid_iou_per_class.append(iou_per_class)
        cp.valid_losses.append(np.mean(currentloss))
        print('valid Loss: {}'.format(cp.valid_losses[-1]))
        print('valid mIoU: {}'.format(iou_score))

        per_class_mean = np.array(cp.valid_iou_per_class)
        per_class_mean = np.round(np.mean(per_class_mean, axis=0), 3)
        print('valid per class mIOU: {}'.format(per_class_mean))

        #scheduler.step(iou_score)


        if cp.valid_iou_scores[-1] > best_iou_score:
            best_iou_score = cp.valid_iou_scores[-1]
            best_epoch = i
            if multi_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            checkpoint = {'state_dict': state_dict,
                          'epoch': i,
                          'iou': best_iou_score,
                          'train_iou_scores': cp.train_iou_scores,
                          'train_losses': cp.train_losses,
                          'train_loss': cp.train_losses[-1],
                          'valid_iou_scores': cp.valid_iou_scores,
                          'valid_losses': cp.valid_losses,
                          'training_config': training_config,
                          'name': name,
                          'segmentation_config': segmentation_config}
            torch.save(checkpoint, os.path.join(save_path, '{}_{}.ckpt'.format(name,
                                                                               segmentation_config['encoder_name'])))


        print('best iou: {}'.format(best_iou_score))
        print('best_epoch: {}'.format(best_epoch))

        logs = {'best_iou_score': best_iou_score,
                'best_iou_score_epoch': best_epoch,
                'train_iou_scores': cp.train_iou_scores,
                'train_losses': cp.train_losses,
                'valid_iou_scores': cp.valid_iou_scores,
                'valid_losses': cp.valid_losses}

        with open(os.path.join(logs_path, '{}_{}.json'.format(name, segmentation_config['encoder_name'])), 'w') as file:
            json.dump(logs, file)






if __name__ == '__main__':
    segmentation_config = {'name': 'Unet',
                           'encoder_name': 'resnet34',
                           'encoder_weights': 'imagenet',
                           'activation': 'softmax'}
    training_config = {
        'epochs': 20,
        'batch_size': 4,
        'lr': 1e-3,
        'weight_decay': 0.1,
        'shuffle': True,
        'num_workers': 0,
        'momentum': 0.9,
        'dataset_name': 'bluedude_solo'}

    segmentation_training(training_config, segmentation_config)
