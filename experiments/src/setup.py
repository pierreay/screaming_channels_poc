from setuptools import setup, find_packages

setup(
    name="ScreamingChannels",
    version="2.0",
    packages=find_packages(),
    python_requires=">=3.0,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*",
    entry_points={
        "console_scripts": [
            "sc-experiment = screamingchannels.reproduce:cli",
            "sc-attack = screamingchannels.attack:cli",
            "sc-triage = screamingchannels.triage:cli",
            "sc-waterfall = screamingchannels.waterfall:main"
        ]
    },
    install_requires=[
        "click",
        "numpy",
        "scipy",
        "pyserial",
        "matplotlib",
        "statsmodels",
        "pycryptodome",
        "scikit-learn"
# to use system packages
#        ln -s /usr/lib/python2.7/site-packages/gnuradio ../../../../screaming-channel/nRF52832/experiments/VENV_sc/lib/python2.7/site-packages
#        "gnuradio",
#        "osmosdr",
    ],

    author="S3@EURECOM",
    author_email="camurati@eurecom.fr, poeplau@eurecom.fr, muench@eurecom.fr",
    description="Code for our screaming channel attacks",
    license="GNU General Public License v3.0"
    # TODO URLs
)
