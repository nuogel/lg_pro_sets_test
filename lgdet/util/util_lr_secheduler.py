import math
from matplotlib import pyplot as plt
import numpy as np


def cosine_decay(init_lr, final_lr, step, decay_steps):
    alpha = final_lr / init_lr
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return init_lr * decayed


if __name__ == '__main__':
    epochs = 100
    lr_list = []
    for i in range(epochs):
        lr = cosine_decay(1e-3, 1e-5, i, epochs)
        lr_list.append(lr)
        print(lr)

    x = np.arange(100)
    y = np.asarray(lr_list)
    plt.plot(x, y)
    plt.show()
