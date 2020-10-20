import os
from pathlib import Path
import DenseFusion.tools.train as pose_estimation
import copy
import numpy as np
import json

root = str(Path(__file__).resolve().parent.parent)
import time


def main():
    data_set = 'full_12_classes'

    stats_path = os.path.join(root, 'experiments', 'data')
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    stats_path = os.path.join(stats_path, '{}_stats.json'.format(data_set))
    default_run = {'pw': 1.0, 'pe': 0.0, 'lm': 'new_pred'}
    runs = [{'pw': 1.0, 'pe': 0.0, 'lm': 'pred'}]
    #runs = []
    #pws = [0.25, 0.5, 0.75, 1.0]
    pws = []
    for pw in pws:
        runs.append(copy.deepcopy(default_run))
        runs[-1]['pw'] = pw

    pes = [0.75, 1.0]
    for pe in pes:
        runs.append(copy.deepcopy(default_run))
        runs[-1]['pe'] = pe

    stats = {}
    for i, run in enumerate(runs):
        print('run {}/{}: {}'.format(i+1, len(runs), run))
        print('________________________________________')

        save_extra = '_pw{}_pe{}_{}'.format(run['pw'], run['pe'], run['lm'])

        t_start = time.time()
        pose_estimation.main(data_set, root, p_viewpoints=run['pw'], p_extra_data=run['pe'], label_mode=run['lm'],
                             show_sample=False, save_extra=save_extra, device_num=1)
        t_elapsed = time.time()-t_start
        stats['run{}'.format(i+1)] = {'run': run, 't_elapsed': t_elapsed}

        print('elapsed time: {}, total elapsed time: {}'.format(t_elapsed, np.mean([stats[key]['t_elapsed'] for key in stats])))
        print('________________________________________')
        with open(stats_path, 'w') as jfile:
            json.dump(stats, jfile)



if __name__ == '__main__':
    main()