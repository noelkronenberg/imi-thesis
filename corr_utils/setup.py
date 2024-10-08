from setuptools import setup

setup(name='corr_utils',
    version='0.1',
    description='Utilities for working with the CORR.',
    url='TBA',
    author='Noel Kronenberg',
    author_email='noel.kronenberg@charite.de',
    license='TBA',
    packages=['corr_utils'],
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'dcurves',
        'impyla',
        'pexpect',
        'torchvision',
        'imbalanced-learn',
        'torch',
        # 'sqlite3',
        'pycaret',
        'onnx'
    ],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'])