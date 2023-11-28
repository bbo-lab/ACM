import pathlib
from setuptools import find_packages, setup
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

VERSION = "0.2.4"

# This call to setup() does all the work
setup(
    name="bbo-acm",
    version=VERSION,
    description="Anatomically constraint pose reconstruction from video data",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bbo-lab/ACM",
    author="Arne Monsees, BBO lab",
    author_email="bbo-admin@caesar.de",
    license="LGPLv2+",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=['ACM','ACM.gui'],
    include_package_data=True,
    install_requires=["numpy", "torch", 'scipy', 'argparse'],
)
