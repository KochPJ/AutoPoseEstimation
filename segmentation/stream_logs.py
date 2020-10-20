from segmentation.utils import animate2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os


if __name__ == '__main__':
    dataset = 'exp12'
    filename = 'Unet_resnet34.json'
    mean_cca = False
    path = os.path.join(Path(__file__).resolve().parent, 'logs', dataset, filename)
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    ani = animation.FuncAnimation(fig,
                                  animate2,
                                  fargs=(fig, axs, path),
                                  interval=1000)

    plt.show()
    del ani