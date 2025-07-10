from setuptools import setup, find_packages

setup(
    name="loan_algo_knuth",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.2',
        'tqdm>=4.65.0',
    ],
) 