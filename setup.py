from setuptools import setup, find_packages

setup(
    name="psfdataset",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    url="https://github.com/pypa/sampleproject",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.5',
        'tqdm>=4.46.1',
        'esig>=0.7.1',
    ],
)
