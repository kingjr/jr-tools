import numpy as np
import matplotlib.pyplot as plt
import itertools
from mne.decoding import GeneralizationAcrossTime
from sklearn.cross_validation import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from jr.plot import pretty_plot
from jr.gat import scorer_spearman, plot_graph
from jr.meg import mat2mne

# ################################################################
# Neuronal Encoding


def encode_neurons(y, n_neuron=None, distributed=False, angle=False,
                   heterogeneous=False):
    if (y < 0) or (y > 1):
        raise ValueError
    if n_neuron is None:
        n_neuron = len(y)
    if distributed:
        neurons = np.linspace(0, 1., n_neuron)
        if np.isnan(y):
            y = neurons
        if angle:
            neurons = np.linspace(0, 1., n_neuron + 1)[:-1]
            response = 1. - 2 * np.abs((neurons - y + .5) % 1 - .5)
        else:
            response = 1. - np.abs(neurons - y)
    else:
        neurons = np.ones(n_neuron)
        if np.isnan(y):
            y = 0.
        if angle:
            response = np.zeros_like(neurons)
            half = int(n_neuron / 2)
            response[:half] = neurons[:half] * np.cos(y * 2 * np.pi)
            response[half:] = neurons[half:] * np.sin(y * 2 * np.pi)
        else:
            response = neurons * y
    if heterogeneous is not False:
        response *= heterogeneous
    return response


# ################################################################
# Network architecture

def make_column(wait_init=.01, wait_acc=2.):
    """This generate a predictive coding column based on 3 neurons
        0 : lower region => removed at network level
        1-3 : column
        1 : entry layer
        2 : waiter
        3: predicter
    """
    column = np.zeros((4, 4))
    column[0, 1] = 1.   # from lower region to column
    column[0, 1] = 1.
    column[0, 2] = -1  # stop waiting
    column[1, 2] = wait_init  # initiate waiting
    column[2, 2] = wait_acc  # acceleration of increase
    column[2, 1] = -1
    return column


def make_network(network, n_nodes=10):
    connectivity = np.zeros((n_nodes, n_nodes))
    if network == 'maintain':
        # input
        connectivity[0, 1] = 1
        connectivity[1, 1:n_nodes] = 1
        # first node connected to itself and to other
    elif network == 'serial':
        # all nodes are connected to the next one
        for ii in range(n_nodes - 1):
            connectivity[ii, ii + 1] = 1
    elif network == 'serial_accumulators':
        # all nodes are connected to the next one
        connectivity[0, 1] = .5
        for ii in range(1, n_nodes - 1):
            connectivity[ii, ii + 1] = 1.
            connectivity[ii, ii] = .5
    elif network == 'cumulative':
        for ii in range(n_nodes - 1):
            # all nodes are connected to the next one
            connectivity[ii, ii + 1] = 1
            # all nodes are connected to itself
            connectivity[ii, ii] = 1
    elif network == 'loop':
        # all nodes are connected to the next one
        for ii in range(n_nodes - 1):
            connectivity[ii, ii + 1] = 1
        # final node feedbacks first node
        connectivity[n_nodes - 1, 1] = 1
    elif network == 'inhib_loop':
        # all nodes are connected to the next one
        for ii in range(n_nodes - 1):
            connectivity[ii, ii + 1] = 1
        # final node feedbacks first node
        connectivity[n_nodes - 1, 1] = -1
    elif network == 'instant_feedback':
        # all nodes are connected to the next one
        for ii in range(n_nodes - 1):
            connectivity[ii, ii + 1] = 1
        # final node feedbacks all nodes
        for ii in range(1, n_nodes - 1):
            connectivity[n_nodes - 1, ii] = 1
        # final node feedbacks to itself
        connectivity[n_nodes - 1, n_nodes - 1] = 1
    elif network == 'instant_inhib_feedback':
        # all nodes are connected to the next one
        for ii in range(n_nodes - 1):
            connectivity[ii, ii + 1] = 1
        # final node feedbacks all nodes
        for ii in range(1, n_nodes - 1):
            connectivity[n_nodes - 1, ii] = -1
        # final node feedbacks to itself
        connectivity[n_nodes - 1, n_nodes - 1] = 1
    elif network == 'local_loops':
        # all nodes are connected to the next one
        for ii in range(n_nodes - 1):
            connectivity[ii, ii + 1] = 1
        # all nodes are connected to the preceeding one
        for ii in range(1, n_nodes - 2):
            connectivity[ii + 2, ii] = .5
    else:
        connectivity = np.zeros((2 * n_nodes, 2 * n_nodes))
        if network == 'loop_2':
            # all nodes are connected to the next one
            for ii in range(n_nodes - 1):
                connectivity[ii, ii + 1] = 1
            connectivity[n_nodes - 1, n_nodes] = 1
            for ii in range(n_nodes + 1, 2 * n_nodes):
                connectivity[ii - 1, ii] = 1
        elif network == 'reverse':
            for ii in range(n_nodes - 1):
                connectivity[ii, ii + 1] = 1
            connectivity[n_nodes - 1, 2 * n_nodes - 1] = 1
            for ii in range(n_nodes + 1, 2 * n_nodes):
                connectivity[ii, ii - 1] = 1
        elif network == 'instant_feedback_2':
            # all nodes are connected to the next one
            for ii in range(n_nodes):
                connectivity[ii, ii + 1] = 1
            connectivity[n_nodes, n_nodes - 1] = 1
            for ii in range(n_nodes + 1, 2 * n_nodes):
                connectivity[n_nodes - 1, ii] = 1
    return connectivity


def make_hierarchical_net(column, n_regions=4):
    """this generates a serial network of column connected with one another
    as specified in the first (input) layer of the column"""
    lower = column[0, 1:]
    column = column[1:, 1:]
    n_nodes = len(column)
    network = np.zeros((n_regions * n_nodes, n_regions * n_nodes))
    for ii in range(n_regions):
        start = ii * n_nodes
        next_colum = range((ii + 1) * n_nodes, (ii + 2) * n_nodes)
        network[start:(start + n_nodes), start:(start + n_nodes)] = column
        # first layer of region feeds to the next region
        if ii < (n_regions - 1):
            network[start, next_colum] = lower
    # add first entry
    network = np.vstack((np.zeros((1, n_regions * n_nodes)), network))
    network = np.hstack((np.zeros((n_regions * n_nodes + 1, 1)), network))
    network[0, 1:(n_nodes + 1)] = lower
    return network


def make_horizontal_net(column, n_columns=4, n_regions=2, horizontal=list()):
    network = list()
    subnet = make_hierarchical_net(column, n_regions=n_regions)
    n_nodes = len(column) - 1
    n_hierch_nodes = n_nodes * n_regions + 1
    n_horiz_nodes = n_columns * n_hierch_nodes
    network = np.zeros((n_horiz_nodes, n_horiz_nodes))
    for ii in range(n_columns):
        start = ii * n_hierch_nodes
        network[start:(start + n_hierch_nodes),
                start:(start + n_hierch_nodes)] = subnet

    for this_region in range(n_regions):
        for this_column in range(n_columns):
            for link in horizontal:
                # add link from one column to opposite column
                sel_from = select_nodes(n_columns, n_regions,
                                        n_nodes=column.shape[0],
                                        column=this_column,
                                        region=this_region, node=link[0])
                opposite_column = (this_column + n_columns // 2) % n_columns
                sel_to = select_nodes(n_columns, n_regions,
                                      n_nodes=column.shape[0],
                                      column=opposite_column,
                                      region=this_region, node=link[1])
                if (len(sel_to) > 1) or (len(sel_from) > 1):
                    raise ValueError
                network[sel_from[0], sel_to[0]] = column[link[0], link[1]]
                # remove link within column
                sel_rm = select_nodes(n_columns, n_regions,
                                      n_nodes=column.shape[0],
                                      column=this_column,
                                      region=this_region, node=link[1])
                network[sel_from[0], sel_rm[0]] = 0
    return network


def select_nodes(n_columns=1, n_regions=1, n_nodes=4, column=None, region=None,
                 node=1):
    if not isinstance(column, list):
        column = [column] if column is not None else range(n_columns)
    if not isinstance(region, list):
        region = [region] if region is not None else range(n_regions)
    if not isinstance(node, list):
        node = [node] if node is not None else range(1, n_nodes)
    sel = np.zeros(n_columns * n_regions * (n_nodes - 1) + n_columns, int)
    kk = -1
    for this_column in range(n_columns):
        kk += 1
        # select input
        for this_region in range(n_regions):
            if ((0 in node) and (this_region == 0) and
                    (this_region in region) and
                    (this_column in column)):
                sel[kk] = 1
            for this_node in range(1, n_nodes):
                kk += 1
                if ((this_node in node) and
                    (this_region in region) and
                        (this_column in column)):
                    sel[kk] = 1
    return np.where(sel == 1)[0]


# ################################################################
# Simulation

def make_pulse(pulse, n_time=30, start=2, stop=20):
    n_time = 30 if n_time is None else n_time
    start = 2 if start is None else start
    stop = n_time // 2 if stop is None else stop
    activations = np.zeros(n_time)
    if pulse == 'starts':
        if not isinstance(start, (list, np.ndarray)):
            start = [start]
        for start_ in start:
            activations[start_] = 1
    if pulse == 'start_stop':
        activations[start] = 1
        activations[stop] = 1
    if pulse == 'discrete':
        activations[start] = 1
    if pulse == 'discrete2':
        activations[start] = 1
        activations[start+1] = 1
    if pulse == 'sustained':
        activations[start:stop] = 1
    if pulse == 'differential':
        activations[start] = 1
        activations[stop] = -1
    if pulse == 'differential2':
        activations[start:(start+2)] = 1
        activations[stop:(stop+2)] = -1
    return activations


def make_multidim_pulse(y, n_columns, n_regions, n_nodes, pulse=None,
                        start=None, stop=None, n_time=15,
                        encoder=encode_neurons, encoder_args=None):
    """generates the multidimensional input corresponding to y"""
    if encoder_args is None:
        encoder_args = dict()
    response = make_pulse(pulse, start=start, stop=stop, n_time=n_time)
    # y have to be transformed into 0 - 1 for the encode_angle function
    code = np.zeros((n_columns, n_time))
    for time, signal in enumerate(response):
        code[:, time] = signal * encoder(y, **encoder_args)
    # Put in the network
    sel = select_nodes(n_columns, n_regions, n_nodes, node=0, region=0)
    n_hierch_nodes = (n_nodes - 1) * n_regions + 1
    n_horiz_nodes = n_columns * n_hierch_nodes
    pulses = np.zeros((n_horiz_nodes, n_time))
    pulses[sel, :] = code
    return pulses, code


def compute_dynamics(connectivity, pulses, threshold=[0., 1., 0.]):
    if threshold is None:
        threshold = [-np.inf, np.inf]
    n_nodes = connectivity.shape[0]
    # if pulses is not the size of connectivity, assumes that it connects to
    # the first layer of the connectivity matrix.
    if pulses.ndim == 1:
        pulses = np.vstack((pulses, np.zeros((n_nodes - 1, len(pulses)))))
    n_nodes, n_times = pulses.shape

    def polarity_range(activation):
        activation[activation < threshold[0]] = threshold[0]
        activation[activation > threshold[1]] = threshold[1]
        return activation

    def spike_threshold(activation):
        activation[activation < threshold[2]] = 0.
        return activation

    activations = pulses[:, 0]
    activations = polarity_range(activations)
    dynamics = [np.array(activations, copy=True)]
    for pulse in pulses[:, 1:].T:
        activations = np.dot(connectivity.T, spike_threshold(activations))
        activations = polarity_range(activations)
        activations += pulse
        dynamics.append(np.array(activations, copy=True))
    return np.array(dynamics)

# ################################################################
# EEG


def make_cov(n=1, n_neuron=32, n_sensor=32):
    covs = []
    for ii in range(n):
        covs.append(np.random.randn(n_sensor, n_neuron))
    return np.array(covs)

# def networks(typ, duration=20):
#     duration = int(duration)
#     if typ == 'stable':
#         covs = np.array([make_cov(1)[0]] * duration)
#     if typ == 'sequence':
#         covs = make_cov(duration)
#     if typ == 'negative_loop':
#         sequence = networks('sequence', duration / 2)
#         covs = np.array([cov for cov in sequence] +
#                         [-1 * cov for cov in sequence])
#     return covs


def make_network_covs(n_columns, n_regions, n_nodes, n_sensor=32,
                      polarity=[1, 0, 0]):
    """make a covariance specific to each column, where neuron 1 (PE) is
    inverted in comparison to neuron 3 (Prior)
    assumes nodes = input layer + columnar nodes"""
    n_hierch_nodes = (n_nodes - 1) * n_regions + 1
    n_horiz_nodes = n_columns * n_hierch_nodes
    covs = np.zeros((n_sensor, n_horiz_nodes))
    for column in range(n_columns):
        for region in range(n_regions):
            cov = np.random.randn(n_sensor, 1)
            for node, pol in zip(range(1, n_nodes), polarity):
                sel = select_nodes(n_columns=n_columns, n_regions=n_regions,
                                   n_nodes=n_nodes, region=region,
                                   column=column, node=node)
                if len(sel):
                    covs[:, sel] = cov * pol
    return covs

def simulate_trials(network, n_columns, n_regions, n_nodes, n_sensor=32,
                    pulse='starts', snr=.5, start=None, stop=None,
                    y=None, n_trials=100, threshold=[0., 1., 0.],
                    polarity=[1, 0, 0], n_time=20, encoder_args=None):
    # random covariance
    covs = make_network_covs(n_columns, n_regions, n_nodes,
                             n_sensor=n_sensor, polarity=polarity)
    X = np.zeros((n_trials, n_sensor, n_time))
    if y is None:
        y = np.random.rand(n_trials) * 2 * np.pi
    for trial, this_y in enumerate(y):
        # generate the input corresponding to an this_y
        pulses, _ = make_multidim_pulse(this_y, n_columns, n_regions,
                                        n_nodes, pulse=pulse, start=start,
                                        stop=stop, n_time=n_time,
                                        encoder_args=encoder_args)
        dynamics = compute_dynamics(network, pulses, threshold=threshold)
        eeg = np.dot(covs, dynamics.T)
        eeg += np.random.randn(n_sensor, n_time) / snr
        X[trial, :, :] = eeg
    return X, y

# ################################################################
# Decoding
linear_clf = RidgeCV(alphas=[(2 * C) ** -1 for C in [1e-4, 1e-2, 1]])


def quick_score(X, y, clf=linear_clf, scorer=scorer_spearman):
    sel = np.where(~np.isnan(y))[0]
    X = X[sel, :, :]
    y = y[sel]
    epochs = mat2mne(X, sfreq=100)
    clf = make_pipeline(StandardScaler(), clf)
    cv = KFold(len(y), 5)
    gat = GeneralizationAcrossTime(clf=clf, n_jobs=-1, scorer=scorer, cv=cv)
    gat.fit(epochs, y)
    gat.score(epochs, y)
    return gat

# ################################################################
# Plots


def plot_connectivity(connectivity, ax=None, dual=False):
    if ax is None:
        ax = plt.gca()
    ax.matshow(connectivity, cmap='RdBu_r', vmin=-1, vmax=1,
               origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    if dual:
        n_nodes = connectivity.shape[0]
        ax.set_xticks([n_nodes / 4.-.5, n_nodes / 2.-.5,
                       .75 * n_nodes-.5, n_nodes-.5])
        ax.set_xticklabels(['feedforward', '', 'feedback', ''])
        ax.set_yticks([n_nodes / 4.-.5, n_nodes / 2.-.5,
                       .75 * n_nodes-.5, n_nodes-.5])
        ax.set_yticklabels(['feedforward', '', 'feedback', ''])
        ax.axhline(n_nodes/2.-.5, color='gray', linestyle=':')
        ax.axvline(n_nodes/2.-.5, color='gray', linestyle=':')
    ax.set_xlabel('neurons')
    # ax.set_ylabel('neurons')
    pretty_plot(ax)


def plot_dynamics(dynamics, ax=None, dual=False, start=2):
    if ax is None:
        ax = plt.gca()
    n_nodes = dynamics.shape[1]
    ax.matshow(dynamics.T, vmin=-1, vmax=1, cmap='RdBu_r', origin='lower')
    ax.axhline(n_nodes - .5, color='gray', linestyle=':')
    ax.set_yticks([n_nodes / 2., n_nodes - .5, 1.5 * n_nodes])
    ax.set_yticklabels(['feedforward', '', 'feedback'], rotation=90)
    ax.set_ylabel('neurons')
    ax.set_xlabel('Times')
    ax.set_aspect('auto')
    if not isinstance(start, (list, np.ndarray)):
        start = [start]
    for start_ in start:
        ax.set_xticks([start_ - .5])
        ax.set_xticklabels([start_ - start[0]])
    for start_ in start:
        ax.axvline(start_ - .5, color='k')
    pretty_plot(ax)
    if dual is False:
        ax.set_ylim(None, n_nodes - .5)
        ax.set_yticks([])


def plot_node(node, ax=None, linewidth=2):
    if ax is None:
        ax = plt.gca()
    connectivity = make_network([0, 0, 0], n_node=3)
    connectivity[2, 1] = node[1]
    init_pos = np.array([[0, .5, 1.], [0., 0., 0.]])
    options = dict(
        iterations=0, edge_curve=True, directional=True,
        node_color='k', node_alpha=1., negative_weights=True,
        init_pos=init_pos, ax=ax, final_pos=None, node_size=linewidth*100,
        edge_width=linewidth, self_edge=1000, clim=[-1, 1], arrowstyle='->')
    G, nodes, = plot_graph(connectivity,  edge_color='k', **options)
    nodes.set_linewidths(linewidth)
    nodes.set_zorder(-1)
    connectivity = make_network(node, n_node=3)
    connectivity[0, 1] = 0
    G, nodes, = plot_graph(connectivity,  edge_color=plt.get_cmap('bwr'),
                           **options)
    nodes.set_linewidths(linewidth)
    nodes.set_zorder(-1)
    ax.scatter(.5, 0, linewidth*100, color='w', zorder=3)
    ax.set_aspect('equal')
    ax.patch.set_visible(False)
    ax.set_ylim([-.2, .2])
    ax.set_xlim([-.1, 1.1])


def plot_network(network, n_columns=1, n_regions=1, radius=None, ax=None,
                 linewidth=2):
    """This plot the columnar network"""
    if ax is None:
        ax = plt.subplot()
    # network is made of n_columns + entry node
    n_nodes = (len(network) // n_columns - 1) // n_regions
    init_pos = np.zeros((network.shape[0], 2))
    x = np.linspace(-1, 1, n_regions)
    y = np.linspace(-1, 1, n_columns)
    if radius is None:
        radius = 1. / (2 * n_columns)
    if n_nodes > 2:
        z = np.linspace(-np.pi / 2, 3 * np.pi / 2, n_nodes + 1)[:-1]
        z = np.vstack(([-1., 0.], np.transpose([np.cos(z), np.sin(z) + 1.])))
    else:
        z = np.vstack((np.zeros(2), np.linspace(0, 1, n_nodes)))
        z = np.hstack(([-1, 0], z))
        z = np.vstack(([-1., 0.], z))  # XXX needs to be checked
    for column, region, node in itertools.product(
            range(n_columns), range(n_regions), range(0, n_nodes)):
        sel = select_nodes(n_columns, n_regions, n_nodes=(n_nodes + 1),
                           column=column, region=region, node=node)
        if node == 0:
            first_node = np.diff(x[:2]) if len(x) > 1 else 0.
            init_pos[sel, 0] = x[region] - first_node
        else:
            init_pos[sel, 0] = x[region] + z[node, 0] * radius
        init_pos[sel, 1] = y[column] + z[node, 1] * radius
    G, nodes, = plot_graph(
        network, iterations=0, edge_curve=True, directional=True,
        node_color='w', node_alpha=1., edge_color=plt.get_cmap('bwr'),
        negative_weights=True, init_pos=init_pos.T, ax=ax, final_pos=None,
        node_size=linewidth*100, edge_width=linewidth, self_edge=1000,
        clim=[-1, 1])
    nodes.set_linewidths(linewidth)
    ax.set_aspect('equal')
    ax.patch.set_visible(False)
    if n_columns > 1:
        ax.set_ylim([-.2, 1.2])
    if n_regions > 1:
        ax.set_xlim([-.2, 1.2])

    # if gif:
    #     anim = animate_graph(dynamics, G, nodes)
    #     fname = report.report.data_path + '/network_%s.gif' % network
    #     anim.save(fname, writer='imagemagick', dpi=75)
