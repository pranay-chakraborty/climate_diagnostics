from setuptools import setup, find_packages

# Read the README.md for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="climate_diagnostics",  # PyPI name (uses hyphens)
    version="0.1.0",
    author="Pranay Chakraborty",
    author_email="pranay.chakraborty.personal@gmail.com",
    description="Climate diagnostics tools for analyzing and visualizing climate data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "xarray",
        "dask",
        "netCDF4",
        "bottleneck",
        "matplotlib",
        "numpy",
        "scipy",
        "cartopy",  # Note: May require conda-forge for installation
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mock",
            "pytest-cov",
            "flake8",
            "mypy",
            "black",
            "isort",
            "pre-commit",
            "sphinx",
            "sphinx-rtd-theme",
            "nbsphinx",
            "tox",
            "build",
            "twine",
            "jupyter",
            "ipykernel",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
