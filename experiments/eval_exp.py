from pipeline.utils import *
from DenseFusion.lib.loss import Loss
from DenseFusion.lib.loss_refiner import Loss_refine
from pathlib import Path
from DenseFusion.datasets.myDatasetAugmented.dataset import PoseDataset
import json


root = str(Path(__file__).resolve().parent.parent)

def main():
    pass
    data_set_name = 'exp12'
    exp_name = 'full_12_classes'
    classes = get_classes(data_set_name)
    print(classes)
    results_path = os.path.join(root, 'experiments', 'data', '{}_exp_eval_results.json'.format(exp_name))

    num_objects = len(classes)

    exp_path = os.path.join(root, 'DenseFusion', 'trained_models', data_set_name, exp_name)
    exps = sorted(list(os.listdir(exp_path)))

    exps = ['full_12_classes_pw1.0_pe1.0_new_pred']
    print(exps)
    results = {}
    for i, exp in enumerate(exps):
        run = 'run: {}/{}'.format(i+1, len(exps))

        estimator, refiner = get_models(data_set_name, exp_name, exp, num_objects)

        num_points = 1000
        refine_start = True
        show_sample = False
        label_mode = 'new_pred'
        p_extra_data = '0.0'
        p_viewpoints = '1.0'
        workers = 8
        iteration = 2
        w = 0.015

        results[exp] = eval(num_points, refine_start, data_set_name, show_sample, label_mode, p_extra_data,
                            p_viewpoints, estimator, w, refiner, iteration, workers, classes, run)

        with open(results_path, 'w') as file:
            json.dump(results, file)


def eval(num_points, refine_start, data_set_name, show_sample, label_mode, p_extra_data, p_viewpoints,
         estimator, w, refiner, iteration, workers, classes, run):

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


    total_less = 0
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

        print('{} | sample {}/{} | dis: {}, average ADD-s: {}, total <2: {}'.format(run,
                                                                                    j+1,
                                                                                    len(testdataloader),
                                                                                    np.round(dists[-1], 5),
                                                                                    np.round(np.mean(dists), 5),
                                                                                    total_less))

    for key, v in results.items():
        results[key]['p'] = float(np.round(v['<2']/(v['>=2']+v['<2'])*100, 2))
        results[key]['dis'] = float(np.round(np.mean(v['dis']), 5))
    results['average_add'] = float(np.round(np.mean(dists), 5))
    results['total_less_then_two'] = float(total_less)
    print(results)
    return results


def get_classes(data_set_name):
    class_file = open(os.path.join(root, 'label_generator', 'data_sets', 'segmentation', data_set_name, 'classes.txt'))
    classes = []
    while 1:
        class_input = class_file.readline()[:-1]
        if not class_input:
            break
        classes.append(class_input)
    return classes

def get_models(data_set_name, exp_name, exp, num_objects):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        cuda = True
    else:
        cuda = False
        device = torch.device('cpu')
    print(cuda, device)

    print('create estimator and refiner models')
    pose_path = os.path.join(root, 'DenseFusion', 'trained_models', data_set_name, exp_name, exp)
    num_points = 1000
    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    loading_path = os.path.join(pose_path, 'pose_model.pth')
    pretrained_dict = torch.load(loading_path, map_location=torch.device('cpu'))
    estimator.load_state_dict(pretrained_dict)
    loading_path = os.path.join(pose_path, 'pose_refine_model.pth')
    pretrained_dict = torch.load(loading_path, map_location=torch.device('cpu'))
    refiner.load_state_dict(pretrained_dict)

    estimator.to(device)
    refiner.to(device)
    estimator.eval()
    refiner.eval()
    return estimator, refiner

if __name__ == '__main__':
    main()



