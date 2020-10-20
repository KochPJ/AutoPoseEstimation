import os
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

root = str(Path(__file__).resolve().parent.parent)

def main():
    exp_name = 'full_12_classes'
    path = os.path.join(root, 'DenseFusion', 'experiments', 'logs', exp_name)
    dirs = sorted(list(os.listdir(path)))
    show_plot = True
    results = {}
    for d in dirs:
        path2 = os.path.join(path, d, 'losses.json')

        with open(path2) as jfile:
            results[d] = json.load(jfile)

    exp_results = {'bs pred': {'pw': {}, 'pe': {}},
                   'sm pred': {'pw': {}, 'pe': {}}}
    for key, logs in results.items():
        print(key)
        name = key.split('_')
        pw = float(name[3][2:])*100
        pe = float(name[4][2:])*100

        if show_plot:

            fig, axs = plt.subplots(4, 1, constrained_layout=True)
            if pe != 0:
                pe2 = float(np.round(pe/(100+pe)*100, 2))
                fig.suptitle(
                    'Training Results of the pose estimation with {}% extra data.'.format(pe2))
            else:
                fig.suptitle(
                    'Training Results of the pose estimation with {}% view points'.format(pw))
            axs[0].plot(logs['losses'], c='b')
            axs[0].set_title('Estimator Loss')
            axs[0].set_ylabel('Dense Fusion Loss')
            axs[0].set_xlabel('Epochs')

            axs[1].plot(logs['refiner_losses'], c='b')
            axs[1].set_title('Refiner Loss')
            axs[1].set_ylabel('Dense Fusion Loss')
            axs[1].set_xlabel('Epochs')

            axs[2].plot(logs['train_dists'], c='r')
            axs[2].set_title('Train ADD')
            axs[2].set_ylabel('ADD [m]')
            axs[2].set_xlabel('Epochs')

            axs[3].plot(logs['test_dists'], c='r')
            axs[3].set_title('Validation ADD')
            axs[3].set_ylabel('ADD [m]')
            axs[3].set_xlabel('Epochs')
            plt.show()

        start_ref = len(logs['refiner_losses'])
        for i in range(len(logs['refiner_losses'])):
            if logs['refiner_losses'][i] != 0:
                start_ref = i
                break

        print(start_ref, len(logs['refiner_losses']))
        if len(logs['refiner_losses']) < 499:
            print('logs to small, continuing')
            continue

        best_estimator = np.round(np.min(logs['test_dists'][:start_ref]), 4)
        best_estimator_epoch = np.argmin(logs['test_dists'][:start_ref])

        best_refiner = np.round(np.min(logs['test_dists'][start_ref:]), 4)
        best_refiner_epoch = np.argmin(logs['test_dists'][start_ref:])+start_ref

        if 'new' in name:
            name = 'sm pred'
        else:
            name = 'bs pred'

        if pe != 0:

            exp_results[name]['pe'][str(pe)] = {'best_estimator': best_estimator,
                                       'best_estimator_epoch': best_estimator_epoch,
                                       'best_refiner': best_refiner,
                                       'best_refiner_epoch': best_refiner_epoch}
        else:
            exp_results[name]['pw'][str(pw)] = {'best_estimator': best_estimator,
                                       'best_estimator_epoch': best_estimator_epoch,
                                       'best_refiner': best_refiner,
                                       'best_refiner_epoch': best_refiner_epoch}
    print(exp_results)

if __name__ == '__main__':
    main()
