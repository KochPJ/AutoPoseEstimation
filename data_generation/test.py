import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

for i in range(157):
    path = './data/bluedude/background/{:06d}.color.png'.format(i)
    path2 = './data/bluedude/background/{:06d}.depth.png'.format(i)
    path3 = './data/bluedude/foreground/{:06d}.color.png'.format(i)
    path4 = './data/bluedude/foreground/{:06d}.depth.png'.format(i)
    with open(path, 'rb') as f:
        image = np.array(Image.open(f).convert('RGB'))

    with open(path3, 'rb') as f:
        image2 = np.array(Image.open(f).convert('RGB'))

    with open(path2, 'rb') as f:
        depth = np.array(Image.open(f))
    with open(path4, 'rb') as f:
        depth2 = np.array(Image.open(f))

    print(image[269, 387], depth[269, 387])
    depth = depth/10
    depth[depth>255] = 0


    depth2 = depth2/10
    depth2[depth2>255] = 0

    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.subplot(2,2,2)
    plt.imshow(depth, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(image2)
    plt.subplot(2,2,4)
    plt.imshow(depth2, cmap='gray')
    plt.show()