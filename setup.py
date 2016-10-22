#! /usr/bin/env python
#

# Copyright (C) 2015 Jean-Remi King
# <jeanremi.king@gmail.com>
#
# Adapted from MNE-Python

import os
import setuptools  # noqa
from numpy.distutils.core import setup

# Get version
version = None
with open(os.path.join('jr', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = """Tools to facilitate MEG analyses"""

DISTNAME = 'jr'
DESCRIPTION = descr
MAINTAINER = 'Jean-Remi King'
MAINTAINER_EMAIL = 'jeanremi.king@gmail.com'
URL = 'https://github.com/kingjr/jr-tools'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/kingjr/jr-tools'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=False,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Linux'],
          platforms='any',
          packages=['jr', 'jr.tests',
                    'jr.cloud', 'jr.cloud.tests',
                    'jr.gat', 'jr.gat.tests',
                    'jr.gif', 'jr.gif.tests',
                    'jr.meg', 'jr.meg.tests',
                    'jr.model',
                    'jr.plot',
                    'jr.stats', 'jr.stats.tests'],
          package_data={},
          scripts=[])
