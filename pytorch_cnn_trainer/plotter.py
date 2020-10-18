import matplotlib.pyplot as plt

__all__ = ['plot_results']


def plot_results(history, train_metric, val_metric):

    """
    Plots the graphs of train vs validation metric for history dictionary.
    Args:
        history: A dictionary returned from fit() function.
        train_metric: Training Metric to plot. One of loss, top1_acc, top1_acc
        val_matric: Validation Metric to plot. One of loss, top1_acc, top1_acc
    """

    plt.plot(history["train"][train_metric])
    plt.plot(history["train"][val_metric])
    plt.plot(history["val"][train_metric])
    plt.plot(history["val"][val_metric])
    plt.title('train v/s val')
    plt.ylabel('metrics')
    plt.xlabel('epoch')
    plt.legend(['train-' + train_metric, 'train-' + val_metric, 'val-' + train_metric, 'val-' + val_metric], loc='upper left')
    plt.show()

