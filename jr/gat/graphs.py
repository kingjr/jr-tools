import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

black_green = mcol.LinearSegmentedColormap(
    'black_green',
    {'red':   ([0.] * 3, (1.0, 0.0, 0.0)),
     'green': ([0.] * 3, [1.] * 3),
     'blue':  ([0.] * 3, (1.0, 0.0, 0.0))}, 256)

white_red = mcol.LinearSegmentedColormap(
    'white_red',
    {'red':   ((0.0, 1.0, 1.0), [1.] * 3),
     'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
     'blue':  ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))}, 256)

white_black = mcol.LinearSegmentedColormap(
    'white_red',
    {'red':   ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
     'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
     'blue':  ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))}, 256)


def plot_graph(X, directional=False, prune=None, negative_weights=True,
               weights_scale=10, iterations=1000, fixed=None, init_pos=None,
               node_size=100, node_color=None, node_alpha=.5,
               edge_curve=False, edge_width=None, edge_width_scale=1,
               edge_color=None, pos=None,
               edge_alpha=.5, self_edge=False, wlim=[.1, 2], clim=None,
               ax=None, final_pos='auto', arrowstyle='-'):
    """
    Parameters
    ----------
    X : connectivity matrix shape(n_nodes, n_nodes)
    prune : significant connections (p_values < .05)

    Returns
    -------
    G : the network
    nodes: Paths Collection of all nodes
    """
    import copy
    import networkx as nx
    from sklearn.decomposition import PCA
    X = copy.deepcopy(X)

    # default parameters
    n_nodes = len(X)
    if not directional:
        np.fill_diagonal(X, np.diag(X) / 2)
        X = (X + X.T) / 2.
        # for ii in range(n_nodes - 1):
        #     for jj in range(ii + 1, n_nodes):
        #         X[ii, jj] = 0.
    if negative_weights:
        weights = np.abs(X * weights_scale)
    else:
        # only use positive connections
        weights = X * weights_scale
        weights *= weights > 0.

    # --- network shape
    # # ----- TODO first and last nodes need to be empty
    if directional:
        G = nx.from_numpy_matrix(weights, create_using=nx.DiGraph())
    else:
        G = nx.from_numpy_matrix(weights, create_using=nx.MultiGraph())
    # ---- bias for t0 left
    if init_pos is None:
        r = np.linspace(-np.pi, np.pi, n_nodes)
        init_pos = np.vstack((np.cos(r), np.sin(r)))
        # init_pos += np.random.randn(*init_pos.shape) / 1000.
    init_pos = dict(zip(range(n_nodes), init_pos.T))
    # ---- compute graph
    if pos is None:
        pos = nx.spring_layout(G, pos=init_pos, iterations=iterations,
                               fixed=fixed)

    # ATTRIBUTES
    # ---- nodes color
    if node_color is None:
        node_color = plt.cm.rainbow
    if isinstance(node_color, (mcol.LinearSegmentedColormap,
                               mcol.ListedColormap)):
        node_color = plt.get_cmap(node_color)
        node_color = np.array([node_color(float(ii) / n_nodes)
                              for ii in range(n_nodes)])
    elif np.ndim(node_color) == 1 or isinstance(node_color, str):
        node_color = [node_color] * n_nodes

    # ---- node size
    if isinstance(node_size, (float, int)):
        node_size = [node_size] * n_nodes

    # ---- edge width
    if edge_width is None:
        edge_width = np.abs(weights) / weights_scale
        edge_width[edge_width < wlim[0]] = wlim[0]
        edge_width[edge_width > wlim[1]] = wlim[1]
    if isinstance(edge_width, (float, int)):
        edge_width = edge_width * np.ones_like(weights)

    # ---- edge color
    if clim is None:
        clim = np.min(X), np.max(X)
    if edge_color is None:
        edge_color = white_black
    if isinstance(edge_color, mcol.LinearSegmentedColormap):
        cmap = plt.get_cmap(edge_color)
        edge_color = (X - float(clim[0])) / float(np.ptp(clim))
        edge_color[edge_color > 1.] = 1.
        edge_color[edge_color < 0.] = 0.
        edge_color = cmap(edge_color)
    elif isinstance(edge_color, str) or np.ndim(edge_color) == 1:
        edge_color = np.tile(edge_color, weights.shape)
    else:
        raise ValueError('unknown edge color')

    # ---- add attributes to graph
    for ii in G.nodes():
        G.node[ii]['pos'] = pos[ii]
        G.node[ii]['color'] = node_color[ii]
        G.node[ii]['size'] = node_size[ii]

    for (ii, jj) in G.edges():
        if directional:
            G.edges[ii, jj]['width'] = edge_width[ii, jj]
            G.edges[ii, jj]['color'] = edge_color[ii, jj]
        else:
            G.edges[ii, jj][0]['width'] = edge_width[ii, jj]
            G.edges[ii, jj][0]['color'] = edge_color[ii, jj]

    # ---- prune graph for plotting
    if prune is None:
        prune = np.zeros_like(X)
    for (ii, jj) in G.edges():
        if prune[ii, jj]:
            G.remove_edge(ii, jj)

    try:
        outdeg = G.degree()
        to_remove = [n for n in outdeg if outdeg[n] == 0]
        G.remove_nodes_from(to_remove)
    except:
        pass

    # ---- Rotate graph for horizontal axis
    xy = np.squeeze([pos[xy] for xy in pos if xy not in to_remove])
    if n_nodes > 1:
        if final_pos == 'auto':
            pca = PCA(whiten=False)
            pca.fit(xy)
            xy = pca.transform(xy)
        elif final_pos == 'horizontal':
            # center
            center = np.tile(xy[0, :], (xy.shape[0], 1))
            xy -= center
            # flip
            xy[:, 0] *= -1
            # polar coordinate
            angles = np.arctan2(xy[:, 1], xy[:, 0])
            radius = np.sqrt(np.sum(xy ** 2, axis=1))
            angles = angles - angles[-1]
            xy = np.vstack((np.cos(angles) * radius,
                            np.sin(angles) * radius)).T
            xy += center
    xy_ = np.zeros((n_nodes, 2))
    xy_[np.array([ii for ii in range(n_nodes) if ii not in to_remove]), :] = xy
    pos = dict(zip(range(n_nodes), xy_))

    # update G nodes pos
    for ii, xy in zip(G.nodes(), xy_):
        G.node[ii]['pos'] = xy

    # plot
    if ax is None:
        fig, ax = plt.subplots(1)

    node_color = [G.node[node]['color'] for node in G.nodes()]
    node_size = [G.node[node]['size'] for node in G.nodes()]
    if directional:
        edge_color = [G.edge[ii][jj]['color'] for (ii, jj) in G.edges()]
        edge_width = [G.edge[ii][jj]['width'] for (ii, jj) in G.edges()]
    else:
        edge_color = [G.edge[ii][jj][0]['color'] for (ii, jj) in G.edges()]
        edge_width = [G.edge[ii][jj][0]['width'] for (ii, jj) in G.edges()]

    draw_net = draw_curve_network if edge_curve else nx.draw_networkx_edges
    if self_edge is True:
        self_edge = np.max(node_size)
    edges = draw_net(G, pos, ax=ax, edge_color=edge_color,
                     width=np.array(edge_width) * edge_width_scale,
                     self_edge=self_edge, edge_alpha=edge_alpha,
                     arrowstyle=arrowstyle)
    if edge_alpha is not None and not edge_curve:
        edge_colors = edges.get_edgecolors()
        edge_colors[:, 3] = edge_alpha
        edges.set_edgecolors(edge_colors)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, alpha=node_alpha,
                                   node_color=node_color, node_size=node_size)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_axis_off()

    return G, nodes, edges


def draw_curve_network(G, pos, edge_color=None, width=None, ax=None,
                       self_edge=None, edge_alpha=None, arrowstyle='-'):
    from matplotlib.patches import FancyArrowPatch
    seen = {}

    objects = list()
    for edge, (ii, jj) in enumerate(G.edges()):
        # default alpha: .5
        try:
            # find out whether it's directional or not
            color = G.edge[ii][jj][0]['color'] if edge_color is None\
                else edge_color[edge]
            width_ = G.edge[ii][jj][0]['width'] if width is None\
                else width[edge]
        except Exception:
            color = G.edge[ii][jj]['color'] if edge_color is None\
                else edge_color[edge]
            width_ = G.edge[ii][jj]['width'] if width is None else width[edge]
        if edge_alpha is not None or len(color) == 3:
            alpha = edge_alpha
        # reverse angle is arrow already exists
        rad = 0.2
        if (ii, jj) in seen:
            rad = seen.get((ii, jj))
            rad = (rad + np.sign(rad) * 0.1) * -1
        seen[(ii, jj)] = rad
        # plot arrow
        if ii != jj:
            e = FancyArrowPatch(pos[ii], pos[jj],
                                arrowstyle=arrowstyle,
                                connectionstyle='arc3,rad=%s' % rad,
                                mutation_scale=10.0,
                                lw=width_, alpha=alpha, color=color)
            ax.add_patch(e)
        else:
            verts = [[self_edge / 2 + self_edge * np.cos(kk),
                      self_edge / 2 + self_edge * np.sin(kk)]
                     for kk in np.linspace(-np.pi, np.pi)]
            ax.scatter(pos[ii][0], pos[ii][1], self_edge, marker=(verts, 0),
                       facecolor='none', edgecolor=color, alpha=alpha,
                       linewidth=width_)
            # ax.add_artist(e)
        # objects.append(e)
    return objects


def annotate_graph(X, pos, keep, times, sel_times=np.arange(0, 700, 100),
                   ax=None):
    """
    Parameters
    ----------
    X
    pos
    keep
    times
    sel_times
    ax

    Returns
    -------
    ax
    """

    colors = np.array([plt.cm.rainbow(float(ii) / X.shape[1])
                       for ii in range(X.shape[1])])
    G = nx.Graph()
    data_nodes = []
    ano_nodes = []
    init_pos = {}
    # Reconstruct original graph
    for j, b in enumerate(range(len(pos))):
        x, y = pos[b]
        data_str = 'data_{0}'.format(j)
        G.add_node(data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (x, y)
    # add node where want to annotate
    for time in sel_times:
        idx = np.argmin(np.abs(times - time))
        idx = keep[np.argmin(np.abs(keep - idx))]
        if ((times[idx] - time) < 20):
            ano_str = 'ano_{0}'.format(time)
            data_str = 'data_{0}'.format(idx)
            G.add_node(ano_str, color=colors[idx, :],
                       string='%sms.' % int(time))
            G.add_edge(data_str, ano_str, weight=100)
            ano_nodes.append(ano_str)
            init_pos[ano_str] = pos[idx]
    # recompute graph
    new_pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes,
                               iterations=500)
    # FIXME fix init_pos bug??
    for node in new_pos:
        new_pos[node] -= new_pos[data_nodes[0]] - pos[0]

    # add text
    if ax is None:
        fig, ax = plt.subplots(1)
    for anod in ano_nodes:
        ax.text(new_pos[anod][0], new_pos[anod][1], fontsize=12,
                s=G.node[anod]['string'], color=G.node[anod]['color'],
                horizontalalignment='center')
    return ax


def animate_graph(dynamics, G, nodes, times=None, cmap=white_red, clim=None):
    from matplotlib import animation
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    ax = nodes.get_axes()
    fig = ax.get_figure()
    if times is None:
        times = range(len(dynamics))
    if clim is None:
        clim = np.min(dynamics), np.max(dynamics)

    def animate(nframe):
        dynamic = dynamics[nframe, :]
        colors = list()
        for ii in G.nodes():
            color = (dynamic[ii] - clim[0]) / np.ptp(clim)
            color = color if color < 1. else 1.
            color = color if color > 0. else 0.
            colors.append(cmap(color))
        nodes.set_facecolors(colors)
        ax.set_title('%s ms' % (int(times[nframe])),
                     fontdict=dict(horizontalalignment='left'))
    anim = animation.FuncAnimation(fig, animate, frames=len(dynamics))
    return anim
