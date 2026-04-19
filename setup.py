from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='scikit-ptm-fs',
    version='0.12',
    author='Omaima Al Hosni',
    author_email='alhosni_o@hotmail.com',
    description='A Python package for integrating scikit-learn feature selection with multi-label problem transformation methods.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Omaimah-AlHosni/scikit-ptm-fs',
    project_urls={
        'Source': 'https://github.com/Omaimah-AlHosni/scikit-ptm-fs',
        'Tracker': 'https://github.com/Omaimah-AlHosni/scikit-ptm-fs/issues',
    },
    packages=find_packages(include=['scikit_ptm_fs', 'scikit_ptm_fs.*']),
    install_requires=[
        'numpy>=1.20',
        'scikit-learn>=1.0',
        'scipy>=1.5',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)