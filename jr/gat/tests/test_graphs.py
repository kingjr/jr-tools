# LOAD  #######################################################################
import pickle
import numpy as np
source = 'ambiguity'
if source == 'construct_ambiguity':
    from sandbox.decoding.load_some_scores import load_scores
    from scripts.config import paths
    fname = paths('decod', subject='fsaverage', epoch='stim_lock',
                  analysis='average_ambiguity_score_tmp')
    gat_list, events_list, analysis = load_scores(ep_name='stim_lock')
    scores = np.array([gat.scores_ for gat in gat_list])

    with open(fname, 'wb') as f:
        pickle.dump([gat_list[0], scores], f)
elif source == 'ambiguity':
    from scripts.config import paths
    fname = paths('decod', subject='fsaverage', epoch='stim_lock',
                  analysis='average_ambiguity_score_tmp')
    gat_list = list([0])
    with open(fname, 'rb') as f:
        [gat_list[0], scores] = pickle.load(f)
    scores *= 100
elif source == 'lucie':
    # Lucie data
    import glob
    data_path = '/media/DATA/Pro/Projects/Paris/Gen_time/data/lucie_masking/'
    subjects = glob.glob(data_path + 'DATA*.pickle')
    scores = list()
    for fname in subjects:
        with open(fname, 'rb') as f:
            [gat, score] = pickle.load(f)
        scores.append(score)
    scores = np.array(scores) - .5
    gat_list = [gat]


# STATS #######################################################################
from mne.stats import spatio_temporal_cluster_1samp_test
start = np.where(gat_list[0].train_times_['times'] >= 0.)[0][0]
# start = 0
X = scores[:, start::1, start::1]
T_obs_, clusters, p_values_, _ = spatio_temporal_cluster_1samp_test(
    X,
    out_type='mask',
    n_permutations=128,
    threshold=dict(start=2, step=2.),
    n_jobs=4)

p_values = p_values_.reshape(X.shape[1:])
h = p_values < .05


# PLOT ########################################################################
import matplotlib.pyplot as plt
from sandbox.graphs.utils import plot_graph, annotate_graph, animate_graph

times = 1e3 * gat_list[0].train_times_['times'][start:]
mean_scores = np.mean(X, axis=0)

# Summary figure
fig, ax = plt.subplots(1)
nodes, keep, pos, G = plot_graph(mean_scores, prune=h, negative_weights=True,
                                 edge_curve=False, ax=ax)
annotate_graph(X, pos, keep, times, sel_times=np.arange(0, 700, 100), ax=ax)
# fig.savefig('all_times.png', dpi=300)
fig.show()

# Animation
fig, ax = plt.subplots(1)
nodes, keep, pos = plot_graph(mean_scores, h=h, negative=True,
                              curve=False, ax=ax, node_size=50.)
anim = animate_graph(mean_scores, nodes, keep, times, ax=ax)
anim.save('test_demoanimation.gif', writer='imagemagick', fps=24)
