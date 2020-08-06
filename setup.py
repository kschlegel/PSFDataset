from setuptools import setup, find_packages

setup(
    name="psfdataset",
    version="0.0.1",
    author="Kevin Schlegel",
    author_email="kevinschlegel@cantab.net",
    description=
    "Implementation of the Path-signature-featuremethodology for creating datasets for human action recognition from landmark data.",
    url="https://github.com/kschlegel/PSFDataset",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.5',
        'tqdm>=4.46.1',
        'esig==0.7.1',
    ],
    extras_require={
        ':python_version < "3.8"': [
            'typing_extensions>=3.7.4.2',
        ],
    },
)
