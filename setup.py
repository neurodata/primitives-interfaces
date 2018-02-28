import os
import sys
from setuptools import setup
from setuptools.command.install import install
from subprocess import check_output, call
from sys import platform

PACKAGE_NAME = 'jhu_primitives'
MINIMUM_PYTHON_VERSION = 3, 6
VERSION = '0.0.4'

class InstallR(install):
    #https://stackoverflow.com/questions/15440115/how-would-i-run-a-script-file-as-part-of-the-python-setup-py-install#16609054
    def run(self):
        install.run(self)
        print("\n\n\nInstalling R:\n\n\n")
        subprocess.call("install_r.sh", shell=True)


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))

#def install_r():
#    print("Trying to install R")
#    sys.stdout.flush()
#    """ Install r-base using apt-get if on UBUNTU"""
#    if platform == "linux" or platform == "linux2":
#        ## https://cran.rstudio.com/bin/linux/ubuntu/
#        print("Adding rstudio repo for Artful")
#        sys.stdout.flush()
#        os.system("sh -c '''echo 'deb https://cran.rstudio.com/bin/linux/ubuntu artful/ >> /etc/apt/sources.list'''")
#        os.system("gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9")
#        os.system("gpg -a --export E084DAB9 | apt-key add -")
#        os.system("apt-get update")
#        os.system("apt-get -y install r-base")
#        #os.system("apt-get -y install r-base-dev")
#        #os.system("apt-get -y install r-recommended")


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)

check_python_version()

#install_r()

setup(
    include_package_data = True,
    scripts=['install_r.sh'],
    name=PACKAGE_NAME,
    version=VERSION,
    description='Python interfaces for TA1 primitives',
    long_description='A library wrapping JHU\'s Python interfaces for the D3M program\'s TA1 primitives.',
    author='Disa Mhembere, Eric Bridgeford, Youngser Park, Heather G. Patsolic, Tyler M. Tomita, Jesse L. Patsolic',
    author_email="disa@jhu.edu",
    packages=[
              PACKAGE_NAME,
              'jhu_primitives.ase',
              'jhu_primitives.lse',
              'jhu_primitives.dimselect',
              'jhu_primitives.gclust',
              'jhu_primitives.nonpar',
              'jhu_primitives.numclust',
              'jhu_primitives.ptr',
              'jhu_primitives.oocase',
              'jhu_primitives.sgc',
              'jhu_primitives.sgm',
              'jhu_primitives.vnsgm'
    ],
    entry_points = {
        'd3m.primitives': [
            'jhu_primitives.AdjacencySpectralEmbedding=jhu_primitives.ase:AdjacencySpectralEmbedding',
            'jhu_primitives.LaplacianSpectralEmbedding=jhu_primitives.lse:LaplacianSpectralEmbedding',
            'jhu_primitives.DimensionSelection=jhu_primitives.dimselect:DimensionSelection',
            'jhu_primitives.GaussianClustering=jhu_primitives.gclust:GaussianClustering',
            'jhu_primitives.NonParametricClustering=jhu_primitives.nonpar:NonParametricClustering',
            'jhu_primitives.NumberOfClusters=jhu_primitives.numclust:NumberOfClusters',
            'jhu_primitives.OutOfCoreAdjacencySpectralEmbedding=jhu_primitives.oocase:OutOfCoreAdjacencySpectralEmbedding',
            'jhu_primitives.PassToRanks=jhu_primitives.ptr:PassToRanks',
            'jhu_primitives.SpectralGraphClustering=jhu_primitives.sgc:SpectralGraphClustering',
            'jhu_primitives.SeededGraphMatching=jhu_primitives.sgm:SeededGraphMatching',
            'jhu_primitives.VertexNominationSeededGraphMatching=jhu_primitives.vnsgm:VertexNominationSeededGraphMatching'
            ]
    },
    package_data = {'': ['*.r', '*.R']},
    cmdclass={'install': InstallR},
    install_requires=['typing', 'numpy', 'scipy',
        'python-igraph', 'rpy2', 'sklearn', 'jinja2', 'primitive_interfaces'],
    url='https://github.com/neurodata/primitives-interfaces',
)

"""
    packages=[
              PACKAGE_NAME,
              'jhu_primitives.ase',
              'jhu_primitives.lse',
              'jhu_primitives.dimselect',
              'jhu_primitives.gclust',
              'jhu_primitives.nonpar',
              'jhu_primitives.numclust',
              'jhu_primitives.oocase',
              'jhu_primitives.ptr',
              'jhu_primitives.sgc',
              'jhu_primitives.sgm',
              'jhu_primitives.vnsgm',
              'jhu_primitives.utils',
              'jhu_primitives.wrapper',
              'jhu_primitives.core',
              'jhu_primitives.monomial'
    ]"""
