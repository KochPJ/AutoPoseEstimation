import torch
import numpy as np
from DenseFusion.lib.transformations import quaternion_matrix, quaternion_from_matrix
import copy


def my_estimator_prediction(pred_r, pred_t, pred_c, num_points, bs, cloud):
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)
    points = cloud.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)
    return my_pred, my_r, my_t

def my_refined_prediction(pred_r, pred_t, my_r, my_t):
    my_mat = quaternion_matrix(my_r)
    my_mat[0:3, 3] = my_t
    pred_r = pred_r.view(1, 1, -1)
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
    my_r_2 = pred_r.view(-1).cpu().data.numpy()
    my_t_2 = pred_t.view(-1).cpu().data.numpy()
    my_mat_2 = quaternion_matrix(my_r_2)

    my_mat_2[0:3, 3] = my_t_2

    my_mat_final = np.dot(my_mat, my_mat_2)
    my_r_final = copy.deepcopy(my_mat_final)
    my_r_final[0:3, 3] = 0
    my_r_final = quaternion_from_matrix(my_r_final, True)
    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

    my_pred = np.append(my_r_final, my_t_final)
    my_r = my_r_final
    my_t = my_t_final
    return my_pred, my_r, my_t#


def get_new_points(pred_r, pred_t, pred_c, points):
    bs, num_p, _ = pred_c.size()
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                                num_p,
                                                                                                                1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                                num_p,
                                                                                                                1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)),
                     dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)

    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)

    t = ori_t[which_max[0]] + points[which_max[0]]
    points = points.view(1, bs * num_p, 3)

    ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()


    return new_points.detach()