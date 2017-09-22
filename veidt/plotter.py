# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.



import matplotlib.pyplot as plt
from prettyplotlib import brewer2mpl

# To Do: 1) plot from a model object 2)write test


def draw_neural_net(ax, left, right, bottom, top, layer_sizes, features,
                    linealpha=1, circlealpha=1):
    """
    Draw a neural network scheme plot using matplotilb.
    This implementation is based on draw_neural_net.py by craffel (https://gist.github.com/craffel/2d727968c3aaebd10359.js)
    example:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :param ax: matplotlib.axes.AxesSubplot
            The axes on which to plot the plt (get e.g. by plt.gca())
    :param left: float
            The center of the leftmost node(s) will be placed here
    :param right: float
            The center of the rightmost node(s) will be placed here
    :param bottom: float
            The center of the bottommost node(s) will be placed here
    :param top: float
            The center of the topmost node(s) will be placed here
    :param layer_sizes: list of int
            List of layer sizes, including input and output dimensionality
    :param features: list of str
            List of features representation in string
    :param linealpha: float in 0:1
            Alpha (opacity) of the connecting line
    :param circlealpha: float in 0:1
            Alpha (opacity) of the neurons
    :return: matplotlib.axes.AxesSubplot
            The axes on which to plot the plt
    """

    # check input size to be the same as input layer size
    assert (len(features) == layer_sizes[0]), " Number of input features not compatible with input layer size"

    colors = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for layer, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        color = colors[layer]
        for node in range(layer_size):
            x = layer * h_spacing + left
            y = layer_top - node * v_spacing
            radius = v_spacing / 2.
            circle = plt.Circle((x, y), radius,
                                color='w', ec=color, zorder=4, linewidth=3, alpha=circlealpha)
            if layer == 0:
                ax.text(x, y, features[node], zorder=5, fontsize=15)
            ax.add_artist(circle)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        color = colors[n]
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c=color, alpha=linealpha)

                ax.add_artist(line)

    return ax


def draw_neuron(ax, activate_func, x=0.5, y=0.5, r=0.4):
    """
    Draw a neuron of neural network using matplotilb.
    example:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neuron(fig.gca(), 'tanh',x=0.5,y=0.5,r=1)
    :param ax: matplotlib.axes.AxesSubplot
            The axes on which to plot the plt (get e.g. by plt.gca())
    :param activate_func: string
            string name of the activation function (e.g. 'relu','sigmoid','tanh')
    :param x: float
            x coordinate of the neuron center
    :param y: float
            r coordinate of the neuron center
    :param r: float
            radius of the neuron
    :return: matplotlib.axes.AxesSubplot
            The axes on which to plot the plt
    """

    colors = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
    # arrow length is the r, text sits at the middle of the arrow
    para_size = 20
    arrow_len = r
    text_xpos = r / 2
    text_ypos = 0.001 * para_size

    # text in the neuron
    plt.text(x=x - 0.7 * r, y=y + 0.2 * r,
             s=r'$z^{[l]}_{i} = \sum_{j=1}^{n^{[l-1]}}{w}_{i,j}^{[l]} {a}_{j}^{[l-1]} + b^{[l]}_{i} $',
             fontsize=20, zorder=10)
    plt.text(x=x - 0.7 * r, y=y - 0.2 * r, s=r'$a^{[l]}_{i} = \sigma(z^{[l]}_{i})$', fontsize=20, zorder=10)
    plt.text(x=x - 0.7 * r, y=y - 0.4 * r, s=r'$\sigma : {} $'.format(activate_func), fontsize=20, zorder=9)
    plt.text(x=x - 0.7 * r, y=y - 0.6 * r, s=r'$l: layer; i,j: node $', fontsize=20, zorder=9)

    # arrows and text on the left
    ax.arrow(x=x - 2 * r - 0.1, y=y + r, dx=r, dy=-0.5 * r,
             head_width=0.05, head_length=0.1, linewidth=3, fc=colors[1], ec=colors[1], alpha=1, zorder=5)
    plt.text(x=x - 2 * r - 0.1 - text_xpos, y=y + r + 2 * text_ypos,
             s=r'$a_{j=1}^{[l-1]}$',
             fontsize=para_size, zorder=6)
    plt.text(x=x - 1.75 * r - 0.05, y=y + 0.5 * r,
             s=r'$w_{i,j=1}^{[l]}$',
             fontsize=para_size, zorder=6)

    ax.arrow(x=x - 2 * r - 0.1, y=y, dx=r, dy=0,
             head_width=0.05, head_length=0.1, linewidth=3, fc=colors[1], ec=colors[1], alpha=1, zorder=5)

    ax.arrow(x=x - 2 * r - 0.1, y=y - r, dx=r, dy=0.5 * r,
             head_width=0.05, head_length=0.1, linewidth=3, fc=colors[1], ec=colors[1], alpha=1, zorder=5)
    plt.text(x=x - 2 * r - 0.1 - text_xpos, y=y - r - 2 * text_ypos,
             s=r'$a_{j=n^{[l-1]}}^{[l-1]}$',
             fontsize=para_size, zorder=6)
    plt.text(x=x - 1.75 * r - 0.05, y=y - 0.5 * r,
             s=r'$w_{i,j=n^{[l-1]}} ^{[l]}$',
             fontsize=para_size, zorder=6)

    ax.text(x=x - 2 * r - text_xpos, y=y,
            s=r'$\vdots$',
            fontsize=para_size, zorder=6)

    # arrows on the right
    ax.arrow(x=x + r, y=y, dx=arrow_len, dy=0,
             head_width=0.05, head_length=0.1, linewidth=3, fc=colors[1], ec=colors[1], alpha=1)
    plt.text(x=x + r + text_xpos, y=y + text_ypos,
             s=r'$a^{[l]}_{i}$',
             fontsize=para_size, zorder=6)

    # plot circle
    circle = plt.Circle(xy=(x, y), radius=r, fc='w', ec=colors[1], zorder=4, linewidth=5)

    ax.add_artist(circle)

    return ax
