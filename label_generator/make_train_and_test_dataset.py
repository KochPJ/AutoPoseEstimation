from pathlib import Path
import os
import numpy as np


def make_train_and_test_dataset(object_names,
                                data_set_type,
                                save_name,
                                p_test=0.2,
                                mode='pred',
                                use_extra_data=False):
    train_samples = []
    test_samples = []
    extra_train_samples = []
    root = Path(__file__).resolve().parent.parent

    given_mode = mode

    save_dir = os.path.join(root, 'label_generator/data_sets', data_set_type, save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for object_name in object_names:
        object_path = os.path.join(root, 'label_generator/data', object_name)
        dirs = os.listdir(object_path)
        if 'extra' in dirs:
            if data_set_type == 'segmentation':
                i = dirs.index('extra')
                del dirs[i]
            else:
                if not use_extra_data:
                    i = dirs.index('extra')
                    del dirs[i]

        for d in dirs:
            dir_path = os.path.join(object_path, d)
            samples = sorted(os.listdir(dir_path))
            if samples:

                if d == 'extra':
                    mode = 'new_pred'
                else:
                    mode = given_mode

                tag = '.{}.label.png'.format(mode)
                l = len(tag)
                samples = [s[:-l] for s in samples if tag in s]

                if d != 'extra':
                    step = int(np.round(len(samples) / (len(samples) * p_test), 0))
                    iii = []
                    for i, s in enumerate(samples):
                        if i % step == 0:
                            test_samples.append(os.path.join(object_name, d, s))
                        else:
                            train_samples.append(os.path.join(object_name, d, s))
                        if object_name == 'Disk':
                            iii.append(i)
                else:
                    for s in samples:
                        extra_train_samples.append(os.path.join(object_name, d, s))

    print('number of train samples: {}'.format(len(train_samples)))
    print('number of train samples: {}'.format(len(test_samples)))
    print('number of train samples: {}'.format(len(extra_train_samples)))

    with open(os.path.join(save_dir, 'train_data_list.txt'), 'w') as f:
        for item in train_samples:
            f.write("%s\n" % item)

    with open(os.path.join(save_dir, 'test_data_list.txt'), 'w') as f:
        for item in test_samples:
            f.write("%s\n" % item)

    if use_extra_data:
        with open(os.path.join(save_dir, 'extra_train_data_list.txt'), 'w') as f:
            for item in extra_train_samples:
                f.write("%s\n" % item)

    with open(os.path.join(save_dir, 'classes.txt'), 'w') as f:
        for item in object_names:
            f.write("%s\n" % item)


if __name__ == '__main__':
    object_names = ['bluedude3']
    save_name = 'bluedude_solo'
    make_train_and_test_dataset(object_names, save_name)
