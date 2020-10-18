import matplotlib.pyplot as plt 

__all__ = ['plot_results']

def plot_results(history,metric1,metric2):
    plt.plot(history["train"][metric1])
    plt.plot(history["train"][metric2])
    plt.plot(history["val"][metric1])
    plt.plot(history["val"][metric2])
    plt.title('train v/s val')
    plt.ylabel('metrics')
    plt.xlabel('epoch')
    plt.legend(['train-'+ metric1,'train-'+metric2, 'val-'+metric1,'val-'+metric2], loc='upper left')
    plt.show()
    return None
