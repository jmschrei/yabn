from distutils.core import setup
from distutils.extension import Extension
import numpy as np

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }

if use_cython:
    ext_modules = [
        Extension("yabn.yabn", [ "yabn/yabn.pyx" ], include_dirs=[np.get_include()]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules = [
        Extension("yabn.yabn", [ "yabn/yabn.c" ], include_dirs=[np.get_include()]),
    ]


setup(
    name='yabn',
    version='0.1.0',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['yahbn'],
    url='http://pypi.python.org/pypi/yabn/',
    license='LICENSE.txt',
    description='YABN is a Bayesian Network package for Python, implemented in Cython for speed.',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "scipy >= 0.13.3",
        "networkx >= 1.8.1"
    ],
)
