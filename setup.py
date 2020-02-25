import setuptools

with open('README.md', 'r') as fh:
    README = fh.read()

import lingam
VERSION = lingam.__version__

setuptools.setup(
    name='lingam',
    version=VERSION,
    author='T.Ikeuchi, G.Haraoka, S.Shimizu',
    description='LiNGAM Python Package',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'graphviz'
    ],
    url='https://github.com/cdt15/lingam',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)