"""
This module implements the monthly report class. 
"""

from raindownloader.inpeparser import INPEParsers, INPE
from .reporter import AbstractReport


class MonthlyReport(AbstractReport):
    """Docstring"""

    parsers = {INPEParsers.monthly_accum, INPEParsers.month_accum_manual}
    post_processors = {".grib2": INPE.grib2_post_proc, ".nc": INPE.nc_post_proc}

    def __init__(self):
        pass

    def generate_report(self, *args, **kwargs):
        return super().generate_report(*args, **kwargs)
