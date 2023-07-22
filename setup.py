from setuptools import setup, find_packages

VERSION = '0.0.21'
DESCRIPTION = 'Identification and Classification of the Most Influential Nodes'
LONG_DESCRIPTION = """

Contains functions for the classification and ranking of top candidate features, reconstruction of networks from
adjacency matrices and data frames, analysis of the topology of the network 
and calculation of centrality measures, and identification of the most
influential nodes. Also, a function is provided for running SIRIR model, which 
is the combination of leave-one-out cross validation technique and the conventional SIR model, 
on a network to unsupervisedly rank the true influence of vertices. 
Additionally, some functions have been provided for the assessment of dependence and 
correlation of two network centrality measures as well as the conditional probability of 
deviation from their corresponding means in opposite direction.
Fred Viole and David Nawrocki (2013, ISBN:1490523995).
Csardi G, Nepusz T (2006). 'The igraph software package for complex network research.' InterJournal, Complex Systems, 1695.
Adopted algorithms and sources are referenced in function document.

"""

setup(
    name="influential",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Adrian Salavaty",
    author_email="abbas.salavaty@gmail.com",
    license='GPL-3',
    packages=find_packages(),
    url='http://pypi.python.org/pypi/influential/',
    install_requires=["igraph", "pandas", "numpy", "statistics", "statsmodels", "scipy", "plotnine", "tqdm", "scikit-learn"],
    keywords='influential',
    classifiers= [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Programming Language :: Python :: 3",
    ]
)
