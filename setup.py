from setuptools import setup, find_packages
from rainreporter import __version__ as version

setup(
    name="RainReporter",
    version=version,
    description="Downloader Package for rain obtained from MERGE/GPM model processed by INPE",
    author="Mauricio Cordeiro",
    author_email="cordmaur@gmail.com",
    packages=find_packages(),
    # install_requires=[
    #     "geopandas",
    #     "xarray",
    #     "rasterio",
    #     "rioxarray",
    #     "cfgrib",
    #     "contextily",
    #     "ecCodes",
    #     "ecmwflibs",
    # ],
)
