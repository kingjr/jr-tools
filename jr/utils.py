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
    y = np.lib.stride_tricks.as_strided(np.array(y),
                                        (np.prod(shape), y.size),
                                        (0, y.itemsize)).T
    return y.reshape(np.hstack((len(y), shape)))


class OnlineReport():
    def __init__(self, script='config.py', client=None, upload_on_save=True,
                 results_dir='results/', use_agg=None):
        """WIP"""
        import inspect
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
        self.results_dir = results_dir
        self.script = script
        self.use_agg = use_agg
        self.upload_on_save = upload_on_save
        self.client = client
        if script != 'ipython':
            self._setup_provenance()

    def _setup_provenance(self):
        import os
        from meeg_preprocessing.utils import setup_provenance
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        # Setup display environment
        if self.use_agg is None:
            self.use_agg = os.getenv('use_agg')
            if self.use_agg:
                self.use_agg = True
        # Create report
        self.report, self.run_id, self.results_dir, self.logger = \
            setup_provenance(self.script, self.results_dir,
                             use_agg=self.use_agg)

    def add_figs_to_section(self, fig, title, section):
        if not hasattr(self, 'report'):
            self._setup_provenance()
        return self.report.add_figs_to_section(fig, title, section)

    def save(self, open_browser=False):
        if not hasattr(self, 'report'):
            self._setup_provenance()
        self.report.save(open_browser=open_browser)
        if self.upload_on_save:
            self.client.upload(self.report.data_path, self.report.data_path)
