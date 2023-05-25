
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

DISTNAME = 'rootcp'
DESCRIPTION = '...'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = '...'
MAINTAINER_EMAIL = '...'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = '...'
URL = '...'
VERSION = None

setup(name='rootcp',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['rootcp'],
      # ext_modules=cythonize("bisection/bisect.pyx"),
      include_dirs=[np.get_include()]
      )
