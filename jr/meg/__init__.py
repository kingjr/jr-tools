from .mne_anatomy import mne_anatomy, check_freesurfer
from .base import (make_meta_epochs, mat2mne, resample_epochs, decimate,
                   detect_bad_channels, forward_pipeline, add_channels,
                   anonymize)
from .kit import least_square_reference
from .artefact import remove_linenoise, find_reference
