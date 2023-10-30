from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='timelapsed_remodelling',
    version='1.0.7',
    author='Matthias Walle',
    author_email='matthias.walle@hest.ethz.ch',
    description='Timelapsed Remodelling  for HR-pQCT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/OpenMSKImaging/remodelling',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'remodelling=example.main:main',
        ],
    },
)
