import os
from PIL import Image
from pathlib import Path
import numpy as np
import json


root = str(Path(__file__).resolve().parent.parent)
def main():
    p = 0.2
    path = os.path.join(root, 'data_generation', 'data')
    classes = list(os.listdir(path))

    gt_path = os.path.join(root, 'experiments', 'data', 'gt_test')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)

    json_path = os.path.join(gt_path, 'meta.json')
    if not os.path.exists(json_path):
        meta = {'p': p}
    else:
        with open(json_path) as f:
            meta = json.load(f)


    rotations = ['foreground', 'foreground180']
    for i, cls in enumerate(classes):
        cls_path = os.path.join(path, cls)
        for j, rot in enumerate(rotations):
            print('cls {}/{}, rot {}/{}'.format(i, len(classes), j, len(rotations)))
            rot_path = os.path.join(cls_path, rot)
            name = '{}_{}'.format(cls, rot)
            if not name in meta:
                samples = np.array([s for s in sorted(list(os.listdir(rot_path))) if '.color.png' in s])
                np.random.shuffle(samples)
                meta[name] = list(samples)

            save_path = os.path.join(gt_path, cls, rot)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            n = int(len(meta[name])*meta['p'])
            for sample in meta[name][:n]:
                with open(os.path.join(rot_path, sample), 'rb') as f:
                    x = Image.open(f).convert('RGB')

                x.save(os.path.join(save_path, sample))


    with open(json_path, 'w') as f:
        json.dump(meta, f)











if __name__ == '__main__':
    main()