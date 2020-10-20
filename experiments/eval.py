from pipeline.utils import *
from DenseFusion.lib.loss import Loss
from DenseFusion.lib.loss_refiner import Loss_refine
from pathlib import Path
from torch.autograd import Variable
from DenseFusion.datasets.myDatasetAugmented.dataset import PoseDataset



root = str(Path(__file__).resolve().parent.parent)

def main():
    pass
    data_set_name = 'exp12'
    _, estimator, refiner, classes, _, _, _, _, _ = get_prediction_models(
        root, data_set_name)
    print(classes)
    num_points = 1000
    refine_start = True
    show_sample = False
    label_mode = 'new_pred'
    p_extra_data = '0.0'
    p_viewpoints = '1.0'
    workers = 8
    iteration = 2
    w = 0.015

    eval(num_points, refine_start, data_set_name, show_sample, label_mode, p_extra_data, p_viewpoints,
         estimator, w, refiner, iteration, workers, classes)


def eval(num_points, refine_start, data_set_name, show_sample, label_mode, p_extra_data, p_viewpoints,
         estimator, w, refiner, iteration, workers, classes):

    results = {cls: {'<2': 0, '>=2': 0, 'dis': []} for cls in classes}
    dists = []
    test_dataset = PoseDataset('test',
                               num_points,
                               False,
                               0.0,
                               refine_start,
                               data_set_name,
                               root,
                               show_sample=show_sample,
                               label_mode=label_mode,
                               p_extra_data=p_extra_data,
                               p_viewpoints=p_viewpoints)

    sym_list = test_dataset.get_sym_list()
    num_points_mesh = test_dataset.get_num_points_mesh()

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers)

    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    for j, data in enumerate(testdataloader, 0):
        points, choose, img, target, model_points, idx, intr, np_img = data
        cls_key = classes[idx]
        #print(cls_key)
        points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                         Variable(choose).cuda(), \
                                                         Variable(img).cuda(), \
                                                         Variable(target).cuda(), \
                                                         Variable(model_points).cuda(), \
                                                         Variable(idx).cuda()
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)



        _, dis, new_points, new_target, pred = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points,
                                                         w, refine_start)
        if refine_start:
            for ite in range(0, iteration):
                pred_r, pred_t = refiner(new_points, emb, idx)
                dis, new_points, new_target, pred = criterion_refine(pred_r, pred_t, new_target, model_points, idx,
                                                                     new_points)

        dists.append(dis.item())
        if dists[-1] < 0.02:
            results[cls_key]['<2'] += 1
        else:
            results[cls_key]['>=2'] += 1
        results[cls_key]['dis'].append(dis.item())

        total_less = 0
        total_more = 0
        for k, v in results.items():
            total_less += v['<2']
            total_more += v['>=2']
        total_less = np.round(total_less/(total_more+total_less)*100, 2)

        print('sample {}/{}| dis: {}, average ADD-s: {}, total <2: {}'.format(j, len(testdataloader), np.round(dists[-1], 5),
              np.round(np.mean(dists), 5), total_less))

    for key, v in results.items():
        results[key]['p'] = np.round(v['<2']/(v['>=2']+v['<2'])*100, 2)
        results[key]['dis'] = np.round(np.mean(v['dis']), 5)
    print(results)






if __name__ == '__main__':
    main()



