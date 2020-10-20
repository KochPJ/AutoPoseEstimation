import os
import json

'''
use only for data with foreground and foreground 180
'''

path = './data'
classes = list(os.listdir(path))
tag = '.meta.json'
l = len(tag)
for cls in classes:
    print('_________________')
    print('class: ', cls)
    cls_path = os.path.join(path, cls)
    dirs = list(os.listdir(cls_path))
    for d in dirs:
        dir_path = os.path.join(cls_path, d)
        samples = [s for s in list(os.listdir(dir_path)) if '.meta.json' in s]

        for sample in samples:
            try:
                with open(os.path.join(dir_path, sample)) as f:
                    meta = json.load(f)
                    meta['symmetric'] = 0

                with open(os.path.join(dir_path, sample), 'w') as f:
                    json.dump(meta, f)
            except:
                print(os.path.join(dir_path, sample))
                input()


