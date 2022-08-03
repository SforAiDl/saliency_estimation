"""Utilities for model metrics"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

from .paths import create_folders

def plot_confusion_matrix(conf_matrix,
                          target_names,
                          save_path,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (conf_matrix), make a nice plot

    Arguments
    ---------
    conf_matrix:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(conf_matrix  = conf_matrix,         # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(conf_matrix) / float(np.sum(conf_matrix))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        # settings for adding ticks on top and right side too
        ax.tick_params( direction = 'out' )
        ax_r = ax.secondary_yaxis('right')
        ax_t = ax.secondary_xaxis('top')
        ax_r.tick_params(axis='y', direction='in')
        ax_t.tick_params(axis='x', direction='inout')
        # original ticks
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        # adding secondary ticks
        ax_t.set_xticks(tick_marks)
        ax_t.set_xticklabels(target_names)
        ax_r.set_yticks(tick_marks)
        ax_r.set_yticklabels(target_names)

    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]


    # thresh = conf_matrix.max() / 1.5 if normalize else conf_matrix.max() / 2
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(conf_matrix[i, j]),
                     horizontalalignment="center", fontsize=8,
                     color="black")
        else:
            plt.text(j, i, "{:0.2f}".format(conf_matrix[i, j]),
                     horizontalalignment="center", fontsize=8,
                     color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    create_folders(save_path)
    save_path = f"{save_path}/confusion_matrix.png"
    plt.savefig(save_path)
    # print(f"Confusion Matrix saved at {save_path}")
