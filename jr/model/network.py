import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from jr.plot import pretty_plot, nonlinear_cmap
from jr.gat import scorer_spearman, plot_graph, scorer_auc, force_predict
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

def make_columns(wait_init=.01, wait_acc=2.):
    """This generate a predictive coding column based on 3 neurons
        0 : lower region => removed at network level
        1-3 : column
        1 : entry layer
        2 : waiter
        3: predicter
    """
    within = np.zeros((2, 2))
    feedforward = np.zeros((2, 2))
    feedback = np.zeros((2, 2))
    feedforward[0, 0] = 1.   # from lower region to column
    feedforward[0, 1] = -1  # stop waiting
    within[0, 1] = wait_init  # initiate waiting
    within[1, 1] = wait_acc  # acceleration of increase
    within[1, 0] = -1
    return within, feedforward, feedback


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


def make_hierarchical_net(within, feedforward, feedback, n_regions=4):
    """this generates a serial network of column connected with one another
    as specified in the first (input) layer of the column"""
    # to_higher = column[0, 1:]
    # to_lower = column[1:, 0]
    # print(to_higher)
    # print(to_lower)
    # column = column[1:, 1:]
    n_nodes = len(within)
    network = np.zeros((1 + n_regions * n_nodes, 1 + n_regions * n_nodes))
    kwargs = dict(n_columns=1, n_regions=n_regions, n_nodes=n_nodes, column=0)
    for region, from_node, to_node in itertools.product(
            range(n_regions), range(n_nodes), range(n_nodes)):
        # within
        from_node_ = select_nodes(region=region, node=from_node, **kwargs)
        to_node_ = select_nodes(region=region, node=to_node, **kwargs)
        network[from_node_, to_node_] = 1. * within[from_node, to_node]
        # feedforward
        if region < (n_regions - 1):
            to_node_ = select_nodes(region=region + 1, node=to_node,
                                    **kwargs)
            network[from_node_, to_node_] = feedforward[from_node, to_node]
        # feedback
        if region > 0:
            to_node_ = select_nodes(region=region - 1, node=to_node, **kwargs)
            network[from_node_, to_node_] = feedback[from_node, to_node]
    # add first entry
    network[0, 1] = 1.
    return network


def make_horizontal_net(within, feedforward, feedback,
                        n_columns=4, n_regions=2, horizontal=None,
                        horizcolumns=None):
    horizcolumns = range(n_columns) if horizcolumns is None else horizcolumns
    network = list()
    subnet = make_hierarchical_net(within, feedforward, feedback,
                                   n_regions=n_regions)
    n_nodes = len(within)
    n_hierch_nodes = n_nodes * n_regions + 1
    n_horiz_nodes = n_columns * n_hierch_nodes
    network = np.zeros((n_horiz_nodes, n_horiz_nodes))
    for ii in range(n_columns):
        start = ii * n_hierch_nodes
        network[start:(start + n_hierch_nodes),
                start:(start + n_hierch_nodes)] = subnet

    if (horizontal is not None) and (np.sum(np.abs(horizontal))):
        for this_region, this_column, node_from, node_to in itertools.product(
                range(n_regions), horizcolumns,
                range(n_nodes), range(n_nodes)):
            # add link from one column to opposite column
            selfrom = select_nodes(n_columns, n_regions,
                                   n_nodes=n_nodes,
                                   column=this_column,
                                   region=this_region,
                                   node=node_from)
            opposite_column = (this_column + n_columns // 2) % n_columns
            selto = select_nodes(n_columns, n_regions,
                                 n_nodes=n_nodes,
                                 column=opposite_column,
                                 region=this_region, node=node_to)
            if (len(selto) > 1) or (len(selfrom) > 1):
                raise ValueError
            network[selfrom[0], selto[0]] = horizontal[node_from, node_to]
    return network


def select_nodes(n_columns=1, n_regions=1, n_nodes=1, column=None, region=None,
                 node=0):
    if not isinstance(column, list):
        column = [column] if column is not None else range(n_columns)
    if not isinstance(region, list):
        region = [region] if region is not None else range(n_regions)
    if not isinstance(node, list):
        node = [node] if node is not None else range(n_nodes)
    sel = np.zeros(n_columns * n_regions * n_nodes + n_columns, int)
    kk = -1
    for this_column in range(n_columns):
        kk += 1
        # select input
        for this_region in range(n_regions):
            if ((-1 in node) and (this_region == 0) and
                    (this_region in region) and
                    (this_column in column)):
                sel[kk] = 1
            for this_node in range(n_nodes):
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
                      polarity=None):
    """make a covariance specific to each column, where neuron 1 (PE) is
    inverted in comparison to neuron 3 (Prior)
    assumes nodes = input layer + columnar nodes"""
    n_hierch_nodes = n_nodes * n_regions + 1
    n_horiz_nodes = n_columns * n_hierch_nodes
    covs = np.zeros((n_sensor, n_horiz_nodes))
    if polarity is None:
        polarity = np.ones(n_nodes)
    for column in range(n_columns):
        for region in range(n_regions):
            cov = np.random.randn(n_sensor, 1)
            for node, pol in zip(range(n_nodes), polarity):
                sel = select_nodes(n_columns=n_columns, n_regions=n_regions,
                                   n_nodes=n_nodes, region=region,
                                   column=column, node=node)
                if len(sel):
                    covs[:, sel] = cov * pol
    return covs


def simulate_trials(network, n_columns=1, n_regions=1,
                    snr=.5,  y=None, n_trials=100, threshold=[0., 1., 0.],
                    n_time=50, pulse_params=None, covs=None, field=[0.]):
    # By default all trials are 1.
    if y is None:
        y = np.ones(n_trials)
    # Pulse can either be an array of n_nodes * n_times indicating when
    # each of them is activated
    if pulse_params is None:
        pulse_params = dict(n_time=n_time)
    # Random covariance
    n_nodes = (len(network) // n_columns - 1) // n_regions
    if covs is None:
        covs = dict()
    if isinstance(covs, dict):
        covs = make_network_covs(n_columns, n_regions, n_nodes, **covs)
    n_sensor = len(covs)
    X = np.zeros((n_trials, n_sensor, n_time))
    for trial, this_y in enumerate(y):
        # generate the input corresponding to an this_y
        if isinstance(pulse_params, dict):
            pulses, _ = make_multidim_pulse(this_y, n_columns, n_regions,
                                            n_nodes, **pulse_params)
        else:
            pulses = pulse_params
        dynamics = compute_dynamics(network, pulses, threshold=threshold)
        dynamics[dynamics < field] = 0.
        eeg = np.dot(covs, dynamics.T)
        eeg += np.random.randn(n_sensor, n_time) / snr
        X[trial, :, :] = eeg
    return X

# ################################################################
# Decoding


def quick_score(X, y, clf=None, scorer=None):
    from mne.decoding import GeneralizationAcrossTime
    from sklearn.cross_validation import KFold
    regression = (len(np.unique(y)) > 2) & isinstance(y[0], float)
    if scorer is None:
        scorer = scorer_spearman if regression else scorer_auc
    if clf is None:
        clf = RidgeCV(alphas=[(2 * C) ** -1 for C in [1e-4, 1e-2, 1]])\
            if regression else force_predict(LogisticRegression(), axis=1)
    sel = np.where(~np.isnan(y))[0]
    X = X[sel, :, :]
    y = y[sel]
    epochs = mat2mne(X, sfreq=100)
    clf = make_pipeline(StandardScaler(), clf)
    cv = KFold(len(y), 5) if regression else None
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


def plot_dynamics(dynamics, ax=None, start=2,
                  n_columns=None, n_regions=None, clim=[-1, 1],
                  region=None, node=None, column=None, cmap=None):
    if ax is None:
        ax = plt.gca()
    if (column is not None) or (region is not None) or (node is not None):
        n_nodes = (len(dynamics.T) // n_columns - 1) // n_regions
        sel = select_nodes(n_columns=n_columns, n_regions=n_regions,
                           n_nodes=n_nodes, node=node, region=region,
                           column=column)
        dynamics = dynamics[:, sel]
    if cmap is None:
        cmap = nonlinear_cmap('RdBu_r', 0, clim)
        clim = [0., 1.]
    ax.matshow(dynamics.T, vmin=clim[0], vmax=clim[1], cmap=cmap,
               origin='lower')
    if (isinstance(column, list) and len(column) == 1):
        for ii in range(n_columns):
            ax.axhline(ii * (len(dynamics.T) // n_columns) - .5,
                       color='gray', linestyle=':')
    ax.set_ylabel('Modules')
    ax.set_xlabel('Time')
    ax.set_aspect('auto')
    if not isinstance(start, (list, np.ndarray)):
        start = [start]
    for start_ in start:
        ax.set_xticks([start_ - .5])
        ax.set_xticklabels([start_ - start[0]])
    for start_ in start:
        ax.axvline(start_ - .5, color='k')
    pretty_plot(ax)


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
                 linewidth=2, cmap=None, clim=None):
    """This plot the columnar network"""
    from matplotlib.patches import ArrowStyle
    ax = plt.subplot() if ax is None else ax
    cmap = plt.get_cmap('bwr') if cmap is None else cmap
    clim = [-1, 1] if clim is None else clim
    # network is made of n_columns + entry node
    n_nodes = (len(network) // n_columns - 1) // n_regions
    init_pos = np.zeros((network.shape[0], 2))
    x = np.linspace(-1, 1, n_regions)
    y = np.linspace(-1, 1, n_columns)
    if radius is None:
        radius = 1. / n_columns
    z = np.linspace(-np.pi / 2, 3 * np.pi / 2, n_nodes + 1)[:-1]
    z = np.transpose([np.cos(z), np.sin(z) + 1.])
    for column, region, node in itertools.product(
            range(n_columns), range(n_regions), range(-1, n_nodes)):
        sel = select_nodes(n_columns, n_regions, n_nodes=n_nodes,
                           column=column, region=region, node=node)
        if node == -1:
            first_node = np.diff(x[:2]) if len(x) > 1 else 1.
            init_pos[sel, 0] = -1 - first_node
            init_pos[sel, 1] = y[column] - 1 * radius
        else:
            init_pos[sel, 0] = x[region] + z[node, 0] * radius
        init_pos[sel, 1] = y[column] + z[node, 1] * radius
    arrow_style = ArrowStyle.Fancy(head_length=1., head_width=1.25,
                                   tail_width=.25)
    G, nodes, = plot_graph(
        network, iterations=0, edge_curve=True, directional=True,
        node_color='w', node_alpha=1., edge_color=cmap,
        negative_weights=True, init_pos=init_pos.T, ax=ax, final_pos=None,
        node_size=linewidth*100, edge_width=linewidth, self_edge=1000,
        clim=clim, arrowstyle=arrow_style)
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


def plot_interactive_dynamic(pulses, n_nodes=1, n_time=50, n_regions=10,
                             n_columns=2, threshold=[0, 1, 0],
                             within=None, feedback=None, feedforward=None,
                             horizontal=None, slim=[-2., 2.]):
    from matplotlib.widgets import Slider, Button
    # initialize network values
    z = np.zeros((n_nodes, n_nodes))
    feedforward = np.copy(z) if feedforward is None else feedforward
    within = np.copy(z) if within is None else within
    feedback = np.copy(z) if feedback is None else feedback
    horizontal = np.copy(z) if horizontal is None else horizontal
    # initialize sliders
    fig = plt.figure()
    axes_all = list()
    sliders_all = list()
    for n_from, n_to in itertools.product(range(n_nodes), range(n_nodes)):
        axes_ = list()
        sliders_ = list()
        for ii, conn in enumerate([feedforward, within, feedback, horizontal]):
            x = 1. / n_nodes * n_to / 4. + ii / 4.
            y = 1. / n_nodes * n_from / 2.
            w = 1. / n_nodes / 4.
            h = 1. / n_nodes / 2.
            ax = fig.add_axes([x, y, .8 * w, .8 * h])
            ax.patch.set_visible(False)
            axes_.append(ax)
            slider = Slider(ax, [ii, n_from, n_to], slim[0], slim[1],
                            valinit=conn[n_from, n_to])
            sliders_.append(slider)
        axes_all.append(axes_)
        sliders_all.append(sliders_)

    # initialize dynamics
    axes_dyn = list()
    n_dyn = float(len(pulses))
    for onset in range(len(pulses)):
        ax = fig.add_axes([onset/n_dyn, .5, 1/n_dyn, .5])
        axes_dyn.append(ax)
    im_dyn = list()
    for ax in axes_dyn:
        im_dyn.append(ax.matshow(np.zeros((n_time, n_time)), vmin=-1, vmax=1,
                      cmap='RdBu_r', origin='lower'))

    # Compute and draw

    def update(val):
        # update values
        for n_from, n_to in itertools.product(
                range(n_nodes), range(n_nodes)):
            idx = n_from * n_nodes + n_to
            feedforward[n_from, n_to] = sliders_all[idx][0].val
            within[n_from, n_to] = sliders_all[idx][1].val
            feedback[n_from, n_to] = sliders_all[idx][2].val
            horizontal[n_from, n_to] = sliders_all[idx][3].val
        # build and simulate network
        network = make_horizontal_net(within, feedforward, feedback,
                                      n_regions=n_regions, n_columns=n_columns,
                                      horizontal=horizontal)
        # plot
        for pulse, im in zip(pulses, im_dyn):
            dynamics = compute_dynamics(network, pulse, threshold=threshold)
            im.set_data(dynamics.T)
        fig.canvas.draw()

    for ii, n_from, n_to in itertools.product(
            range(4), range(n_nodes), range(n_nodes)):
        idx = n_from * n_nodes + n_to
        sliders_all[idx][ii].on_changed(update)

    getax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(getax, 'get values')

    def get_values(event):
        print(feedforward)
        print(within)
        print(feedback)
        print(horizontal)
    button.on_clicked(get_values)
    update(None)
    plt.show()


def quick_gat(dyn1, dyn2=None, n_rep=3, snr=1e3):
    from jr.stats import repeated_corr
    dyn2 = dyn1 if dyn2 is None else dyn2
    n_time, n_chan = dyn1.shape
    gat = np.zeros((n_time, n_time, n_rep))
    for rep in range(n_rep):
        dyn1_ = dyn1 + (2 * np.random.rand(*dyn1.shape) - 1.) / snr
        dyn2_ = dyn2 + (2 * np.random.rand(*dyn2.shape) - 1.) / snr
        for ii in range(n_time):
            clf = dyn1_[ii, :]
            gat[ii, :, rep] = repeated_corr(dyn2_.T, clf)
    return np.mean(gat, axis=2)


def plot_interactive_dynamic_contrast(
    pulses, n_nodes=1, n_time=50, n_regions=10, n_columns=2,
    threshold=[0, 1, 0], within=None, feedback=None, feedforward=None,
        horizontal=None, horizcolumns=None, slim=[-2., 2.]):
    "XXX WIP XXX"
    from matplotlib.widgets import Slider, Button
    # initialize network values
    z = np.zeros((n_nodes, n_nodes))
    feedforward = np.copy(z) if feedforward is None else feedforward
    within = np.copy(z) if within is None else within
    feedback = np.copy(z) if feedback is None else feedback
    horizontal = np.copy(z) if horizontal is None else horizontal
    # initialize sliders
    fig = plt.figure()
    axes_all = list()
    sliders_all = list()
    for n_from, n_to in itertools.product(range(n_nodes), range(n_nodes)):
        axes_ = list()
        sliders_ = list()
        for ii, conn in enumerate([feedforward, within, feedback, horizontal]):
            x = 1. / n_nodes * n_to / 4. + ii / 4.
            y = 1. / n_nodes * n_from / 3.
            w = 1. / n_nodes / 4.
            h = 1. / n_nodes / 3.
            ax = fig.add_axes([x, y, .8 * w, .8 * h])
            ax.patch.set_visible(False)
            axes_.append(ax)
            slider = Slider(ax, [ii, n_from, n_to], slim[0], slim[1],
                            valinit=conn[n_from, n_to])
            sliders_.append(slider)
        axes_all.append(axes_)
        sliders_all.append(sliders_)

    # initialize dynamics
    axes_dyn = list()
    n_dyn = float(len(pulses))
    for onset in range(len(pulses)):
        ax = fig.add_axes([onset/n_dyn, .6, 1/n_dyn, .3])
        axes_dyn.append(ax)
    im_dyn = list()
    for ax in axes_dyn:
        im_dyn.append(ax.matshow(np.zeros((n_time, n_time)), vmin=-1, vmax=1,
                      cmap='RdBu_r', origin='lower'))
    # initialize gat
    axes_gat = list()
    for onset in range(len(pulses)):
        ax = fig.add_axes([onset/n_dyn, .3, 1/n_dyn, .3])
        axes_gat.append(ax)
    im_gat = list()
    for ax in axes_gat:
        im_gat.append(ax.matshow(np.zeros((n_time, n_time)), vmin=-1, vmax=1,
                      cmap='RdBu_r', origin='lower'))
        ax.plot(ax.get_xlim(), ax.get_ylim(), color='k')

    # Compute and draw

    def update(val):
        # update values
        for n_from, n_to in itertools.product(
                range(n_nodes), range(n_nodes)):
            idx = n_from * n_nodes + n_to
            feedforward[n_from, n_to] = sliders_all[idx][0].val
            within[n_from, n_to] = sliders_all[idx][1].val
            feedback[n_from, n_to] = sliders_all[idx][2].val
            horizontal[n_from, n_to] = sliders_all[idx][3].val
        # build network
        network = make_horizontal_net(within, feedforward, feedback,
                                      n_regions=n_regions, n_columns=n_columns,
                                      horizontal=horizontal,
                                      horizcolumns=horizcolumns)
        # simulate network
        dynamics_list = list()
        for pulse, im in zip(pulses, im_dyn):
            dynamics = compute_dynamics(network, pulse, threshold=threshold)
            im.set_data(dynamics.T)
            dynamics_list.append(dynamics)

        # gat
        # long
        gat_long = quick_gat(dynamics_list[2] - dynamics_list[0])
        im_gat[2].set_data(gat_long)
        # short
        gat_short = quick_gat(dynamics_list[2] - dynamics_list[0],
                              dynamics_list[1] - dynamics_list[0])
        soa_diff = (np.where(pulses[1][0] > 0)[0][0] -
                    np.where(pulses[2][0] > 0)[0][0])
        gat_short[:, :-soa_diff] = gat_short[:, soa_diff:]
        im_gat[1].set_data(gat_short)
        # diff
        gat_diff = gat_long - gat_short
        im_gat[0].set_data(gat_diff)

        fig.canvas.draw()

    for ii, n_from, n_to in itertools.product(
            range(4), range(n_nodes), range(n_nodes)):
        idx = n_from * n_nodes + n_to
        sliders_all[idx][ii].on_changed(update)

    getax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(getax, 'get values')

    def get_values(event):
        print(feedforward)
        print(within)
        print(feedback)
        print(horizontal)
    button.on_clicked(get_values)
    update(None)
    plt.show()
