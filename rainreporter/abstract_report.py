"""
Defines the AbstractReport class
"""
from pathlib import Path
import abc
from typing import List, Dict, Optional, Union
from raindownloader.downloader import Downloader
from raindownloader.parser import BaseParser

from .mapper import Mapper


class AbstractReport(abc.ABC):
    """Abstract class to serve as base for each report"""

    parsers: List[BaseParser] = []

    def __init__(self, downloader: Downloader, mapper: Mapper):
        self.downloader = downloader
        self.mapper = mapper

    @abc.abstractmethod
    def generate_report(self, *args, **kwargs):
        """Abstract method that needs to be implemented"""

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls,
        downloader: Downloader,
        mapper: Mapper,
        config: Dict,
        bases_folder: Optional[Union[str, Path]] = None,
    ):
        """Create the report class based on a json specification"""
