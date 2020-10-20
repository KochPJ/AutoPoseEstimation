# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import argparse
import os
import random
import time
import numpy as np
import torch
from pathlib import Path
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from DenseFusion.datasets.myDatasetAugmented.dataset import PoseDataset
from DenseFusion.lib.network import PoseNet, PoseRefineNet
from DenseFusion.lib.loss import Loss
from DenseFusion.lib.loss_refiner import Loss_refine
from matplotlib import pyplot as plt
import pc_reconstruction.open3d_utils as pc_utils
import json
from DenseFusion.tools.utils import *
from DenseFusion.lib.transformations import quaternion_matrix


def main(data_set_name, root, save_extra='', load_pretrained=True, load_trained=False, load_name='',
         label_mode='new_pred', p_extra_data=0.0, p_viewpoints=1.0, show_sample=False, plot_train=False, device_num=0):

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--lr', default=0.0001, help='learning rate')
    parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
    parser.add_argument('--w', default=0.015, help='learning rate')
    parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
    parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
    parser.add_argument('--refine_margin', default=0.010, help='margin to start the training of iterative refinement')
    parser.add_argument('--noise_trans', default=0.03,
                        help='range of the random noise of translation added to the training data')
    parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
    parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
    parser.add_argument('--refine_epoch_margin', type=int, default=400, help='max number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
    opt = parser.parse_args()

    opt.manualSeed = random.randint(1, 10000)

    torch.cuda.set_device(device_num)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    print('bs', opt.batch_size, 'it', opt.iteration)

    opt.refine_start = False
    opt.num_points = 1000 #number of points on the input pointcloud
    opt.outf = os.path.join(root, 'DenseFusion/trained_models', data_set_name+save_extra) #folder to save trained models
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    opt.log_dir = os.path.join(root, 'DenseFusion/experiments/logs', data_set_name+save_extra) #folder to save logs
    opt.log_dir_images = os.path.join(root, 'DenseFusion/experiments/logs', data_set_name+save_extra, 'images')
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.exists(opt.log_dir_images):
        os.makedirs(opt.log_dir_images)


    opt.repeat_epoch = 1 #number of repeat times for one epoch training
    print('create datasets')
    dataset = PoseDataset('train',
                          opt.num_points,
                          True,
                          0.0,
                          opt.refine_start,
                          data_set_name,
                          root,
                          show_sample=show_sample,
                          label_mode=label_mode,
                          p_extra_data=p_extra_data,
                          p_viewpoints=p_viewpoints)

    test_dataset = PoseDataset('test',
                               opt.num_points,
                               False,
                               0.0,
                               opt.refine_start,
                               data_set_name,
                               root,
                               show_sample=show_sample,
                               label_mode=label_mode,
                               p_extra_data=p_extra_data,
                               p_viewpoints=p_viewpoints)


    opt.num_objects = dataset.num_classes #number of object classes in the dataset
    print('n classes: {}'.format(dataset.num_classes))

    print('create models')
    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects)
    refiner.cuda()

    if load_pretrained:
        # load the pretrained estimator model on the ycb dataset, leave the last layer due to mismatch
        init_state_dict = estimator.state_dict()
        pretrained_dict = torch.load(os.path.join(root, 'DenseFusion/trained_models/pose_model.pth'))
        pretrained_dict['conv4_r.weight'] = init_state_dict['conv4_r.weight']
        pretrained_dict['conv4_r.bias'] = init_state_dict['conv4_r.bias']
        pretrained_dict['conv4_t.weight'] = init_state_dict['conv4_t.weight']
        pretrained_dict['conv4_t.bias'] = init_state_dict['conv4_t.bias']
        pretrained_dict['conv4_c.weight'] = init_state_dict['conv4_c.weight']
        pretrained_dict['conv4_c.bias'] = init_state_dict['conv4_c.bias']
        estimator.load_state_dict(pretrained_dict)

        del init_state_dict
        del pretrained_dict

        # load the pretrained refiner model on the ycb dataset, leave the last layer due to mismatch
        init_state_dict = refiner.state_dict()
        pretrained_dict = torch.load(os.path.join(root, 'DenseFusion/trained_models/pose_refine_model.pth'))
        pretrained_dict['conv3_r.weight'] = init_state_dict['conv3_r.weight']
        pretrained_dict['conv3_r.bias'] = init_state_dict['conv3_r.bias']
        pretrained_dict['conv3_t.weight'] = init_state_dict['conv3_t.weight']
        pretrained_dict['conv3_t.bias'] = init_state_dict['conv3_t.bias']
        refiner.load_state_dict(pretrained_dict)

        del init_state_dict
        del pretrained_dict
    elif load_trained:
        loading_path = os.path.join(root, 'DenseFusion/trained_models/{}/pose_model.pth'.format(load_name))
        pretrained_dict = torch.load(loading_path)
        estimator.load_state_dict(pretrained_dict)

        loading_path = os.path.join(root, 'DenseFusion/trained_models/{}/pose_refine_model.pth'.format(load_name))
        pretrained_dict = torch.load(loading_path)
        refiner.load_state_dict(pretrained_dict)
        del pretrained_dict


    print('create optimizer and dataloader')
    #opt.refine_start = False
    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=opt.workers,
    #                                         collate_fn=collate_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}'
          '\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
        len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf
    best_test_epoch = 0
    best_train = np.Inf
    best_train_epoch = 0
    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            if log !='images':
                os.remove(os.path.join(opt.log_dir, log))
        for img in os.listdir(opt.log_dir_images):
            os.remove(os.path.join(opt.log_dir_images, img))

    train_dists = []
    test_dists = []
    losses = []
    refiner_losses = []
    best_loss = np.inf
    best_loss_epoch = 0
    elapsed_times = 0.0

    for epoch in range(opt.start_epoch, opt.nepoch):
        start_time = time.time()
        train_count = 0
        train_dis_avg = 0.0

        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()

        optimizer.zero_grad()

        epoch_losses = []
        epoch_losses_refiner = []
        for rep in range(opt.repeat_epoch):
            #for batch in dataloader:
                #points, choose, img, target, model_points, idx = batch
                #print(points.shape, choose.shape, img.shape, target.shape, model_points.shape)
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target, model_points, idx = data

                #print(points.shape, choose.shape, img.shape, target.shape, model_points.shape)
                points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(target).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                loss, dis, new_points, new_target, pred = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
                epoch_losses.append(loss.item())
                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target, pred = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()
                    epoch_losses_refiner.append(dis.item())
                else:
                    loss.backward()
                    epoch_losses_refiner.append(0)
                train_dis_avg += dis.item()
                train_count += 1

                # make step after one epoch
                if train_count % opt.batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # make last step of epoch if something is remaining
            if train_count % opt.batch_size != 0:
                optimizer.step()
                optimizer.zero_grad()

        refiner_losses.append(np.mean(epoch_losses_refiner))
        losses.append(np.mean(epoch_losses))
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_loss_epoch = epoch

        train_dists.append(train_dis_avg/train_count)
        if train_dists[-1] < best_train:
            best_train_epoch = epoch
            best_train = train_dists[-1]

        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        if plot_train:
            # plot randomly selected validation preds
            jj = 0
            x_axis = 0
            fig_x = 4
            fig_y = 4
            log_indexes = sorted(list(np.random.choice(list(range(len(testdataloader))), int(fig_x*(fig_y/2)), replace=False)))

            plt.cla()
            plt.close('all')
            fig, axs = plt.subplots(fig_x, fig_y, constrained_layout=True, figsize=(25, 15))

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx, intr, np_img = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)

            if plot_train:
                if j in log_indexes:
                    my_pred, my_r, my_t = my_estimator_prediction(pred_r, pred_t, pred_c, opt.num_points, 1, points)

            _, dis, new_points, new_target, pred = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)

                    if plot_train:
                        if j in log_indexes:
                            my_pred, my_r, my_t = my_refined_prediction(pred_r, pred_t, my_r, my_t)

                    dis, new_points, new_target, pred = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            if plot_train:
                if j in log_indexes:
                    if jj == 4:
                        jj = 0
                        x_axis += 1

                    my_r = quaternion_matrix(my_r)[:3, :3]
                    np_pred = np.dot(model_points[0].data.cpu().numpy(), my_r.T)
                    np_pred = np.add(np_pred, my_t)
                    np_target = target[0].data.cpu().numpy()
                    np_img = np_img[0].data.numpy()

                    image_target = pc_utils.pointcloud2image(np_img.copy(), np_target, 3, intr)
                    image_prediction = pc_utils.pointcloud2image(np_img.copy(), np_pred, 3, intr)

                    axs[x_axis, jj].imshow(image_target)
                    axs[x_axis, jj].set_title('target {}'.format(j))
                    axs[x_axis, jj].set_axis_off()
                    jj += 1

                    axs[x_axis, jj].imshow(image_prediction)
                    axs[x_axis, jj].set_title('prediction {}'.format(j))
                    axs[x_axis, jj].set_axis_off()
                    jj += 1

            test_dis += dis.item()
            test_count += 1

        test_dis = test_dis / test_count

        test_dists.append(test_dis)

        if plot_train:
            fig.suptitle('epoch {}, with a average dist: {}'.format(epoch, test_dis), fontsize=16)
            plt.savefig(os.path.join(opt.log_dir_images, 'test_images_epoch_{}.png'.format(epoch)))

            if epoch > 1:
                plt.close('all')
                plt.cla()
                fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(30, 20))
                axs[0, 0].plot(losses)
                axs[0, 0].set_title('Training estimator loss')
                axs[0, 0].set_xlabel('Epochs')
                axs[0, 0].set_ylabel('Loss')

                axs[0, 1].plot(refiner_losses)
                axs[0, 1].set_title('Training refiner loss')
                axs[0, 1].set_xlabel('Epochs')
                axs[0, 1].set_ylabel('Loss')

                axs[1, 0].plot(train_dists)
                axs[1, 0].set_title('Training Avg. distance')
                axs[1, 0].set_xlabel('Epochs')
                axs[1, 0].set_ylabel('Avg. distance [m]')

                axs[1, 1].plot(test_dists)
                axs[1, 1].set_title('Test Avg. distance')
                axs[1, 1].set_xlabel('Epochs')
                axs[1, 1].set_ylabel('Avg. distance [m]')

                plt.savefig(os.path.join(opt.log_dir_images, 'losses.png'))

        out_dict = {
            'losses': losses,
            'refiner_losses': refiner_losses,
            'train_dists': train_dists,
            'test_dists': test_dists
        }
        with open(os.path.join(opt.log_dir, 'losses.json'), 'w') as outfile:
            json.dump(out_dict, outfile)
        del out_dict


        print('>>>>>>>>----------Epoch {0} finished---------<<<<<<<<'.format(epoch))
        if test_dis <= best_test:
            best_test = test_dis
            best_test_epoch = epoch
            if opt.refine_start:
                state_dict = refiner.state_dict()
                torch.save(state_dict, '{0}/pose_refine_model.pth'.format(opt.outf))
                del state_dict
            else:
                state_dict = estimator.state_dict()
                torch.save(state_dict, '{0}/pose_model.pth'.format(opt.outf))
                del state_dict

            print('>>>>>>>>----------MODEL SAVED---------<<<<<<<<')

        t_elapsed = time.time() - start_time
        elapsed_times += t_elapsed/3600
        print('elapsed time: {} min, total elapsed time: {} hours'.format(
            np.round(t_elapsed/60, 2), np.round(elapsed_times), 2))

        print('Train loss           : {}'.format(losses[-1]))
        print('Best train loss {}   : {}'.format(best_loss_epoch, best_loss))
        print('Train dist           : {}'.format(train_dists[-1]))
        print('Best train dist {}   : {}'.format(best_train_epoch, best_train))
        print('Test dist            : {}'.format(test_dists[-1]))
        print('Best test dist {}    : {}'.format(best_test_epoch, best_test))



        # changing stuff during training if...
        if best_test < opt.decay_margin and not opt.decay_start:
            print('decay lr')
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)


        if (best_test < opt.refine_margin or epoch >= opt.refine_epoch_margin) and not opt.refine_start:
            #print('train refiner')
            opt.refine_start = True
            print('bs', opt.batch_size, 'it', opt.iteration)
            opt.batch_size = int(opt.batch_size / opt.iteration)
            print('new bs', opt.batch_size)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            #testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            
            #opt.sym_list = dataset.get_sym_list()
            #opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------train refiner!---------<<<<<<<<')
            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)


if __name__ == '__main__':
    data_set_name = 'bluedude_solo'
    save_extra = '_test4'
    root = Path(__file__).resolve().parent.parent.parent
    main(data_set_name, root, save_extra=save_extra)
