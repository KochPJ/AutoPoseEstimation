import torch
import torchvision
import segmentation_models_pytorch as smp
from pathlib import Path
from background_subtraction.dataset import SegmentationDataset
from background_subtraction.utils import jaccard_loss, IoU, animate, get_model, IoU_cca, do_cca
import numpy as np
import os
import json
import matplotlib.pyplot as plt


nets = {'Unet': smp.Unet,
        'PsPNet': smp.PSPNet,
        'LinkNet': smp.Linknet}

class CP:
    def __init__(self):
        self.losses = []
        self.iou_scores = []
        self.iou_cca_scores = []



def segmentation_training(training_config, segmentation_config):
    root = str(Path(__file__).resolve().parent)
    save_path = os.path.join(root, 'trained_models')
    logs_path = os.path.join(root, 'logs')
    logs_images = os.path.join(logs_path, 'images')
    data_path = os.path.join(root, 'data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(logs_images):
        os.makedirs(logs_images)

    for img in os.listdir(logs_images):
        os.remove(os.path.join(logs_images, img))

    classes = os.listdir(data_path)
    train_dirs = {}
    test_dirs = {}
    n_samples = 23
    cut_cls = int(len(classes)*0.8)
    for i, cls in enumerate(classes):
        if i > int(cut_cls):
            test_dirs[cls] = list(range(n_samples))
        else:
            train_dirs[cls] = list(range(n_samples))



    #mean = None
    #std = None
    mean = [0.040278014, 0.04060352, 0.038310923, 0.0381776, 0.03656849, 0.03636289, 0.03556486]
    std = [0.059689723, 0.05965291, 0.056203008, 0.05619316, 0.054657422, 0.054514673, 0.05377024]
    show_plots = False
    random_test = False
    if show_plots:
        training_config['num_workers'] = 0
    train_dataset = SegmentationDataset(mode='train',
                                        root=data_path,
                                        dirs=train_dirs,
                                        classes=classes,
                                        mean=mean,
                                        std=std,
                                        show_plots=show_plots)
    test_dataset = SegmentationDataset(mode='test',
                                       root=data_path,
                                       dirs=test_dirs,
                                       classes=classes,
                                       mean=train_dataset.mean,
                                       std=train_dataset.std)
    segmentation_config['classes'] = train_dataset.n_classes

    name = segmentation_config['name']
    del segmentation_config['name']
    model = get_model(name, segmentation_config)
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=training_config['lr'],
                                momentum=training_config['momentum'],
                                weight_decay=training_config['weight_decay'],
                                nesterov=True)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=training_config['batch_size'],
                                                   shuffle=training_config['shuffle'],
                                                   num_workers=training_config['num_workers'])


    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=training_config['batch_size'],
                                                   shuffle=training_config['shuffle'],
                                                   num_workers=training_config['num_workers'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    metric = IoU(num_classes=train_dataset.n_classes)
    metric_cca = IoU_cca(num_classes=train_dataset.n_classes)

    test_metric1 = IoU(num_classes=train_dataset.n_classes)
    test_metric2 = IoU(num_classes=train_dataset.n_classes)
    test_metric3 = IoU_cca(num_classes=train_dataset.n_classes)
    test_metric4 = IoU_cca(num_classes=train_dataset.n_classes)
    # test random test
    model.eval()

    if random_test:
        print('__________________________________________________')
        print('do random test')
        with torch.no_grad():
            for img, label in test_dataloader:
                img = img.cuda()
                label = label.cuda()
                pred = model(img)
                test_metric1.add(pred, label)
                test_metric3.add(pred, label)

                random_pred = torch.rand(pred.shape)
                test_metric2.add(random_pred, label)
                test_metric4.add(random_pred, label)
        _, test_score1 = test_metric1.value()
        _, test_score2 = test_metric2.value()
        _, test_score3 = test_metric3.value()
        _, test_score4 = test_metric4.value()
        print('test model result: {}'.format(float(test_score1)))
        print('test random result: {}'.format(float(test_score2)))
        print('test cca model result: {}'.format(float(test_score3)))
        print('test cca random result: {}'.format(float(test_score4)))
        del test_metric1
        del test_metric2
        del test_metric3
        del test_metric4



    best_iou_score = 0
    best_iou_cca_score = 0
    best_epoch = 0
    cp = CP()
    print('__________________________________________________')
    print('len train_dataloader: {}'.format(len(train_dataloader)))
    print('len test_dataloader: {}'.format(len(test_dataloader)))
    for i in range(training_config['epochs']):
        print('__________________________________________________')
        print('Epoch {}/{}'.format(i, training_config['epochs']-1))
        currentloss = []
        model.train()
        first_train = True
        for img, label in train_dataloader:
            bs = len(img)
            img = img.cuda()
            label = label.cuda()
            pred = model(img)
            loss = jaccard_loss(label, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            currentloss.append(float(loss.data/bs))

            if first_train:
                new_pred_cca = do_cca(pred.detach())
                new_pred = pred.detach().cpu().numpy()
                new_label = label.cpu().numpy()
                plt.cla()
                fig, axs = plt.subplots(3, 5, constrained_layout=True, figsize=(25, 12))
                for j in range(5):
                    axs[0, j].imshow(new_label[j])
                    axs[0, j].set_title('label {}'.format(j))
                    axs[0, j].set_axis_off()

                    axs[1, j].imshow(np.argmax(new_pred[j].transpose(1, 2, 0), axis=2))
                    axs[1, j].set_title('prediction {}'.format(j))
                    axs[1, j].set_axis_off()

                    axs[2, j].imshow(new_pred_cca[j])
                    axs[2, j].set_title('prediction cca {}'.format(j))
                    axs[2, j].set_axis_off()

                fig.suptitle('epoch {}'.format(i), fontsize=16)
                plt.savefig(os.path.join(logs_images, 'train_{}.png'.format(i)))
                first_train = False


        cp.losses.append(np.mean(currentloss))
        print('Loss: {}'.format(cp.losses[-1]))

        model.eval()
        metric.reset()
        metric_cca.reset()
        first_valid = True
        with torch.no_grad():
            for img, label in test_dataloader:
                img = img.cuda()
                label = label.cuda()
                pred = model(img)
                metric.add(pred, label)
                metric_cca.add(pred, label)
                if first_valid:
                    new_pred_cca = do_cca(pred)
                    new_pred = pred.cpu().numpy()
                    new_label = label.cpu().numpy()
                    plt.cla()
                    fig, axs = plt.subplots(3, 5, constrained_layout=True, figsize=(25, 12))
                    for j in range(5):
                        axs[0, j].imshow(new_label[j])
                        axs[0, j].set_title('label {}'.format(j))
                        axs[0, j].set_axis_off()

                        axs[1, j].imshow(np.argmax(new_pred[j].transpose(1, 2, 0), axis=2))
                        axs[1, j].set_title('prediction {}'.format(j))
                        axs[1, j].set_axis_off()

                        axs[2, j].imshow(new_pred_cca[j])
                        axs[2, j].set_title('prediction cca {}'.format(j))
                        axs[2, j].set_axis_off()


                    fig.suptitle('epoch {}'.format(i), fontsize=16)
                    plt.savefig(os.path.join(logs_images, 'valid_{}.png'.format(i)))

                    first_valid = False

        _, iou_score = metric.value()
        _, iou_score_cca = metric_cca.value()

        scheduler.step(iou_score)

        cp.iou_scores.append(iou_score)
        cp.iou_cca_scores.append(iou_score_cca)
        print('mIoU: {}'.format(iou_score))
        print('mIoU cca: {}'.format(iou_score_cca))


        if cp.iou_scores[-1] > best_iou_score:
            best_iou_cca_score = cp.iou_cca_scores[-1]
            best_iou_score = cp.iou_scores[-1]
            best_epoch = i
            checkpoint = {'state_dict': model.state_dict(),
                          'epoch': i,
                          'iou': best_iou_score,
                          'iou_scores': cp.iou_scores,
                          'losses': cp.losses,
                          'loss': cp.losses[-1],
                          'iou_cca': best_iou_cca_score,
                          'iou_cca_scores': cp.iou_cca_scores,
                          'training_config': training_config,
                          'name': name,
                          'segmentation_config': segmentation_config}
            torch.save(checkpoint, os.path.join(save_path, '{}_{}.ckpt'.format(name,
                                                                               segmentation_config['encoder_name'])))


        print('best iou: {}'.format(best_iou_score))
        print('best iou_cca: {}'.format(best_iou_cca_score))
        print('best_epoch: {}'.format(best_epoch))

        logs = {'best_iou_score': best_iou_score,
                'best_iou_score_epoch': best_epoch,
                'iou_scores': cp.iou_scores,
                'iou_cca_scores': cp.iou_cca_scores,
                'losses': cp.losses}

        with open(os.path.join(logs_path, '{}_{}.json'.format(name, segmentation_config['encoder_name'])), 'w') as file:
            json.dump(logs, file)






if __name__ == '__main__':
    segmentation_config = {'name': 'Unet',
                           'encoder_name': 'resnet34',
                           'encoder_weights': None,
                           'activation': 'softmax',
                           'in_channels': 7}
    training_config = {
        'epochs': 100,
        'batch_size': 5,
        'lr': 5e-3,
        'weight_decay': 0.0,
        'shuffle': True,
        'num_workers': 4,
        'momentum': 0.9,
        'dataset_name': 'bluedude_solo'}

    segmentation_training(training_config, segmentation_config)
