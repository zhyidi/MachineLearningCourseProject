import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch

from config import parse_args


def draw_learning_curve(num_epochs, train_loss, dev_loss=[]):
    x = np.linspace(1, num_epochs, num_epochs)
    plt.plot(x, train_loss, scalex=True, label="training loss")
    plt.plot(x, dev_loss, label="development loss")
    plt.legend()
    plt.show()


def draw_prediction_curve(timestep_x, timestep_y, sample_plot):
    true_x, true_y, pred_y = sample_plot
    idx = 1
    true_x, true_y, pred_y = true_x[idx, :, -1], true_y[idx, :, -1], pred_y[idx, :, -1] # shape: (96)

    x = np.linspace(1, timestep_x + timestep_y, timestep_x + timestep_y)
    true_y_plot = np.concatenate([true_x, true_y], axis=0)
    pred_y_plot = np.empty_like(true_y_plot)
    pred_y_plot[:] = numpy.nan
    pred_y_plot[timestep_x:] = pred_y

    plt.plot(x, true_y_plot, label="GroundTruth", linewidth=2.0)
    plt.plot(x, pred_y_plot, label="Prediction", linewidth=2.0)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # draw_learning_curve(5, [4, 2, 3, 1, 1])

    x = np.linspace(0, 10, 100)
    y = 4 + 2 * np.sin(2 * x)
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))
    ax.set()
    plt.show()
