from segmentation.utils import animate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os


if __name__ == '__main__':
    filename = 'Unet_resnet34.json'
    path = os.path.join(Path(__file__).resolve().parent, 'logs', filename)
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  fargs=(fig, axs, path),
                                  interval=1000)

    plt.show()