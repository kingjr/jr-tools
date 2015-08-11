import sys
import numpy as np


class ProgressBar(object):
    # Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
    #
    # License: BSD (3-clause)
    """Class for generating a command-line progressbar

    Parameters
    ----------
    max_value : int
        Maximum value of process (e.g. number of samples to process, bytes to
        download, etc.).
    initial_value : int
        Initial value of process, useful when resuming process from a specific
        value, defaults to 0.
    mesg : str
        Message to include at end of progress bar.
    max_chars : int
        Number of characters to use for progress bar (be sure to save some room
        for the message and % complete as well).
    progress_character : char
        Character in the progress bar that indicates the portion completed.
    spinner : bool
        Show a spinner.  Useful for long-running processes that may not
        increment the progress bar very often.  This provides the user with
        feedback that the progress has not stalled.

    Example
    -------
    >>> progress = ProgressBar(13000)
    >>> progress.update(3000) # doctest: +SKIP
    [.........                               ] 23.07692 |
    >>> progress.update(6000) # doctest: +SKIP
    [..................                      ] 46.15385 |

    >>> progress = ProgressBar(13000, spinner=True)
    >>> progress.update(3000) # doctest: +SKIP
    [.........                               ] 23.07692 |
    >>> progress.update(6000) # doctest: +SKIP
    [..................                      ] 46.15385 /

    """

    spinner_symbols = ['|', '/', '-', '\\']
    template = '\r[{0}{1}] {2:.05f} {3} {4}   '

    def __init__(self, max_value, initial_value=0, mesg='', max_chars=40,
                 progress_character='.', spinner=False, verbose_bool=True):
        self.cur_value = initial_value
        self.max_value = float(max_value)
        self.mesg = mesg
        self.max_chars = max_chars
        self.progress_character = progress_character
        self.spinner = spinner
        self.spinner_index = 0
        self.n_spinner = len(self.spinner_symbols)
        self._do_print = verbose_bool

    def update(self, cur_value, mesg=None):
        """Update progressbar with current value of process

        Parameters
        ----------
        cur_value : number
            Current value of process.  Should be <= max_value (but this is not
            enforced).  The percent of the progressbar will be computed as
            (cur_value / max_value) * 100
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        # Ensure floating-point division so we can get fractions of a percent
        # for the progressbar.
        self.cur_value = cur_value
        progress = min(float(self.cur_value) / self.max_value, 1.)
        num_chars = int(progress * self.max_chars)
        num_left = self.max_chars - num_chars

        # Update the message
        if mesg is not None:
            self.mesg = mesg

        # The \r tells the cursor to return to the beginning of the line rather
        # than starting a new line.  This allows us to have a progressbar-style
        # display in the console window.
        bar = self.template.format(self.progress_character * num_chars,
                                   ' ' * num_left,
                                   progress * 100,
                                   self.spinner_symbols[self.spinner_index],
                                   self.mesg)
        # Force a flush because sometimes when using bash scripts and pipes,
        # the output is not printed until after the program exits.
        if self._do_print:
            sys.stdout.write(bar)
            sys.stdout.flush()
        # Increament the spinner
        if self.spinner:
            self.spinner_index = (self.spinner_index + 1) % self.n_spinner

    def update_with_increment_value(self, increment_value, mesg=None):
        """Update progressbar with the value of the increment instead of the
        current value of process as in update()

        Parameters
        ----------
        increment_value : int
            Value of the increment of process.  The percent of the progressbar
            will be computed as
            (self.cur_value + increment_value / max_value) * 100
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        self.cur_value += increment_value
        self.update(self.cur_value, mesg)


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
