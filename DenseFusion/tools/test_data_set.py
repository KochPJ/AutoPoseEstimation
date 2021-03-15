# --------------------------------------------------------

import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from DenseFusion.datasets.myDatasetAugmented2.dataset import PoseDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()

def main():

    data_set_name = 'bluedude_solo'
    opt.num_points = 1000  # number of points on the input pointcloud

    opt.refine_start = False

    dataset = PoseDataset('train',
                          opt.num_points,
                          True,
                          opt.noise_trans,
                          opt.refine_start,
                          data_set_name,
                          show_sample=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, data in enumerate(dataloader, 0):
        print(i)
        points, choose, img, target, model_points, idx = data


if __name__ == '__main__':
    main()
