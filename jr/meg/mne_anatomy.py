import os
import os.path as op
from mne.bem import make_watershed_bem
from mne.commands.mne_make_scalp_surfaces import _run as make_scalp_surfaces
from mne.surface import read_morph_map

"""
# need to export the following vars
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export MNE_ROOT=/home/jrking/MNE-2.7.4-3452-Linux-x86_64
source $MNE_ROOT/bin/mne_setup_sh
export LD_LIBRARY_PATH=/home/jrking/anaconda/lib/
"""


def check_freesurfer(subjects_dir, subject):
    # Check freesurfer finished without any errors
    fname = op.join(subjects_dir, subject, 'scripts', 'recon-all.log')
    if op.isfile(fname):
        with open(fname, 'rb') as fh:
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
        print last
        print('{}: ok'.format(subject))
        return True
    else:
        print('{}: missing'.format(subject))
        return False


def mne_anatomy(subject, subjects_dir, overwrite=False):
    import warnings

    # Checks that watershed hasn't already been run
    for fname in ['fiducials.fif', 'head.fif', 'head-dense.fif',
                  'head-medium.fif', 'head-sparse.fif', 'inner_skull.surf',
                  'oct-6-src.fif']:
        fname = op.join(subjects_dir, subject, 'bem', subject + '-' + fname)
        if (not overwrite) and op.exists(fname):
            raise IOError('%s already exists. Set overwrite=True.' % fname)
            return

    # Create BEM surfaces
    make_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                       overwrite=True, volume='T1', atlas=False,
                       gcaatlas=False, preflood=None)

    # Copy files outside watershed folder in case of bad manipulation
    for surface in ['inner_skull', 'outer_skull', 'outer_skin']:
        from_file = op.join(subjects_dir, subject, 'bem',
                            'watershed/%s_%s_surface' % (subject, surface))
        to_file = op.join(subjects_dir, subject, 'bem', '%s.surf' % surface)
        if op.exists(to_file):
            if overwrite:
                os.remove(to_file)
            else:
                continue
        # Update file
        try:
            os.symlink(from_file, to_file)
        except OSError as e:
            # if disk is not NTFS, symoblic link isn't possible
            if e.strerror == 'Operation not permitted':
                from shutil import copyfile
                copyfile(from_file, to_file)

    # Make scalp surfaces
    make_scalp_surfaces(subjects_dir, subject, force='store_true',
                        overwrite='store_true', verbose=None)

    # Setup source space
    src_fname = op.join(subjects_dir, subject, 'bem',
                        subject + 'oct-6-src.fif')
    if not op.isfile(src_fname):
        from mne import setup_source_space
        setup_source_space(subject, subjects_dir=subjects_dir, fname=src_fname,
                           spacing='oct6', surface='white', overwrite=True,
                           add_dist=True, n_jobs=-1, verbose=None)

    # Prepare BEM model
    bem_fname = op.join(subjects_dir, subject, 'bem',
                        subject + '-5120-bem.fif')
    bem_sol_fname = op.join(subjects_dir, subject, 'bem',
                            subject + '-5120-bem-sol.fif')
    if not op.exists(bem_sol_fname):
        from mne.bem import (make_bem_model, write_bem_surfaces,
                             make_bem_solution, write_bem_solution)
        surfs = make_bem_model(subject, subjects_dir=subjects_dir)
        write_bem_surfaces(fname=bem_fname, surfs=surfs)
        bem = make_bem_solution(surfs)
        write_bem_solution(fname=bem_sol_fname, bem=bem)

    # Make morphs to fsaverage if has it
    try:
        read_morph_map(subject, 'fsaverage', subjects_dir=subjects_dir)
    except IOError as e:
        if 'No such file or directory' in e.strerror:
            warnings.warn(e.strerror)
