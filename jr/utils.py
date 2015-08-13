import sys
import numpy as np


def tile_memory_free(y, shape):
    """
    Tile vector along multiple dimension without allocating new memory.

    Parameters
    ----------
     y : np.array, shape (n,)
        data
    shape : np.array, shape (m),
    Returns
    -------
    Y : np.array, shape (n, *shape)
    """
    y = np.lib.stride_tricks.as_strided(y,
                                        (np.prod(shape), y.size),
                                        (0, y.itemsize)).T
    return y.reshape(np.hstack((len(y), shape)))


class OnlineReport():
    def __init__(self, script='config.py', client=None, upload_on_save=True,
                 results_dir='results/'):
        import os
        import inspect
        from meeg_preprocessing.utils import setup_provenance
        # setup path according to highest script
        for item in inspect.stack():
            if (
                item and (item[1][0] != '<') and
                ('python2.7' not in item[1]) and
                ('utils.py' not in item[1]) and
                ('config.py' not in item[1])
            ):
                script = item[1]
                break
        print script
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        self.script = script
        # Setup display environment
        use_agg = os.getenv('use_agg')
        # Create report
        self.report, self.run_id, self.results_dir, self.logger = \
            setup_provenance(script, results_dir, use_agg=use_agg)
        self.upload_on_save = upload_on_save
        self.client = client

    def add_figs_to_section(self, fig, title, section):
        return self.report.add_figs_to_section(fig, title, section)

    def save(self, open_browser=False):
        self.report.save(open_browser=open_browser)
        if self.upload_on_save:
            self.client.upload(self.report.data_path, self.report.data_path)
