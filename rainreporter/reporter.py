"""
Main module for the reporter class
"""

import io
import logging
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, Dict, Any, Tuple

from PIL import Image
import matplotlib.pyplot as plt

from shapely import Geometry
from pyproj import Geod
import xarray as xr

from pypdf import PdfWriter, PdfReader

from mergedownloader.downloader import Downloader
from mergedownloader.utils import DateProcessor

from .utils import open_json_file

from .abstract_report import AbstractReport
from .monthly_report import MonthlyReport

# from .daily_report import DailyReport
from .mapper import Mapper


class Reporter:
    """Docstring"""

    templates: Dict[str, type[AbstractReport]] = {
        "Mensal": MonthlyReport,
        # "Diario": DailyReport,
    }

    def __init__(
        self,
        downloader: Downloader,
        config_file: Optional[Union[str, Path]] = None,
        bases_folder: Optional[Union[Path, str]] = None,
        log_level: int = logging.DEBUG,
    ):
        """
        :param server: FTP server to connect the downloader to.
        :param download_folder: Folder to store downloaded images.
        :param avoid_update: Avoid updates when file already downlaoded, defaults to True
        :param config_file: Path to the config file. If not passed,
        it will try to load `reporter.json5` from the current folder.
        :param bases_folder: Folder to store the geographic bases.
        :param log_level: Logging level, defaults to logging.DEBUG
        """

        # create a downloader instance
        self.downloader = downloader

        # create the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()
        self.logger.addHandler(self.downloader._logger.handlers[0])

        # load the config file
        self.logger.info("Configuring Reporter class")
        self.config = Reporter.open_config_file(config_file)

        self.logger.info(self.config)

        # create an instance of MapReporter and pass to it the shapes used as backgound
        self.logger.info("Creating a MapReporter instance")
        self.mapper = Mapper(
            config=self.config["shape_style"], shapes=self.config["context_shapes"]
        )

        self.bases_folder = bases_folder

    @staticmethod
    def open_config_file(config_file: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """
        Open the configuration file. If not provided, it will try to locate
        `reporter.json5` in the current folder.
        :param config_file: Path to the configuration file
        :return: Config file as a Dict.
        """

        # transform the config file to Path
        config_file = (
            Path(config_file) if config_file is not None else Path("./reporter.json5")
        )

        return open_json_file(config_file)

    @staticmethod
    def calc_geodesic_area(geom: Geometry) -> float:
        """Calculate the geodesic area given a shapely Geometry"""
        # specify a named ellipsoid
        geod = Geod(ellps="WGS84")
        return abs(geod.geometry_area_perimeter(geom)[0]) / 1e6

    def generate_report(self, date_str: str, attrs: Dict) -> Tuple:
        """
        Generate a single report. It can be a monthly or daily report.
        The simple report must not be confused with the ReportCollection (PDF) that
        aggregates several reports in one document.

        :param template: The template to use. It can be a string with the name of the template
            or a class that inherits from AbstractReport
        :param date_str: The date to generate the report for
        :param attrs: The attributes to use for the report as a dictionary.
        :return: The report as a Tuple [Fig, Axes, DataFrame, GeoDataFrame]
        """

        # Get the correct report template
        template = Reporter.templates[attrs["tipo"]]

        # create the reporter
        reporter = template.from_dict(
            downloader=self.downloader,
            mapper=self.mapper,
            config=attrs,
            bases_folder=self.bases_folder,
        )

        # return the report for the specific date
        return reporter.generate_report(date_str)

    def export_report_data(
        self,
        date_str: str,
        attrs: Dict,
        output_db: Union[Path, str],
        assets_folder: Union[Path, str],
    ):
        """
        Export data for a single report. It can be a monthly or daily report.
        The simple report must not be confused with the ReportCollection (PDF) that
        aggregates several reports in one document.

        :param template: The template to use. It can be a string with the name of the template
            or a class that inherits from AbstractReport
        :param date_str: The date to generate the report for
        :param attrs: The attributes to use for the report as a dictionary.
        :param output_db: The output database to append data to
        """

        # Get the correct report template
        template = Reporter.templates[attrs["tipo"]]

        # create the reporter
        reporter = template.from_dict(
            downloader=self.downloader,
            mapper=self.mapper,
            config=attrs,
            bases_folder=self.bases_folder,
        )

        # call the export_report_data for the specific date
        return reporter.export_report_data(
            date_str, file=Path(output_db), assets_folder=Path(assets_folder)
        )

    def generate_pdf(
        self, json_file: Union[Path, str, Dict], output_folder: Union[Path, str]
    ):
        """
        Generate a ReportColleciton (PDF) on a json file specification.

        :param json_file: The json file specification
        :param output_folder: The output folder
        """

        # open the json file with the pdf specification
        pdf_config = (
            json_file if isinstance(json_file, Dict) else open_json_file(json_file)
        )

        # get the date and use TODAY if it is absent
        date = pdf_config.get("data")
        date = DateProcessor.today() if not date else date
        date_str = DateProcessor.pretty_date(date, "%Y-%m-%d")
        filename = f"{pdf_config['arquivo']}_{date_str}.pdf"

        self.logger.info("Preparing to generate file %s", filename)

        pdf_doc = PdfWriter()
        for report_config in pdf_config["relatorios"]:
            try:
                report = self.generate_report(date_str=date_str, attrs=report_config)

                # get the figure
                # plt_axs = result[0]
                fig = report[0]  # .figure

                # save the PDF to a memory file
                file = io.BytesIO()
                fig.savefig(file, bbox_inches="tight", pad_inches=0.6, format="pdf")

                # append the page to the file
                pdf_doc.append(PdfReader(file))

            except Exception as error:  # pylint: disable=W0703
                self.logger.error(error)

        # save the pdf report to disk
        pdf_doc.write(Path(output_folder) / filename)

    def process_folder(
        self,
        input_folder: Union[str, Path],
        output_folder: Union[str, Path],
        hot: bool = False,
    ):
        """
        Process an entire folder and save the outputs to the output folder.
        The input_folder must contain .json5 files with the PDF definitions.
        The hot flag indicate if the input_folder should be treated as a hot_folder.
        In hot mode, the .json5 files will be moved to the processed folder just after processing.
        """

        self.logger.info("Starting to process folder %s", input_folder)

        # convert the input folder to Path
        input_folder = Path(input_folder)

        # get the files to be processed
        # all .json file in the config folder will be used
        files = list(input_folder.glob("*.json5"))

        if len(files) == 0:
            print(f"No files found to process in {str(input_folder)}")
        else:
            self.logger.info("Output folder: %s", output_folder)
            self.logger.info("Execution mode: %s", "HOT" if hot else "NORMAL")
            self.logger.info("%s files found for processing", len(files))
            self.logger.info(files)

        # if it is a hot folder, create the processed subdirectory
        # load the files in memory and move them to the processed folder
        if hot:
            (input_folder / "hot_processed").mkdir(exist_ok=True)

            parsed_files = []
            for file in files:
                parsed_files.append(open_json_file(file))
                new_name = file.name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                target = file.parent / "hot_processed" / new_name
                file.rename(target)

                self.logger.debug("Renaming file %s to %s", file.name, target.name)

            files = parsed_files

        for file in files:
            try:
                self.generate_pdf(json_file=file, output_folder=output_folder)
            except Exception as error:  # pylint: disable=W0718
                self.logger.error(error)

    @staticmethod
    def animate_cube(
        cube: xr.DataArray,
        filename: Union[Path, str],
        dim: str = "time",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """Animate a given cube along the desired dimension"""

        # Set the limits for the colorscale
        vmin = 0 if vmin is None else vmin
        vmax = float(cube.max()) * 0.8 if vmax is None else vmax

        # create a figure to be used as a canvas
        fig = plt.figure()

        # create a list to store the temporary in-memory files
        files = []
        for pos in cube[dim].data:
            plt_ax = fig.add_subplot()  # type: ignore

            # get a slice of the cube in the specific position (pos)
            cube.sel({dim: pos}).plot(ax=plt_ax, vmin=vmin, vmax=vmax)  # type: ignore

            # Create a temporary file
            file_like = io.BytesIO()

            fig.savefig(file_like)
            files.append(file_like)
            fig.clear()

        # Now, with the files created in memory, let's use PIL to save the GIF
        images = []
        for file in files:
            img = Image.open(file)
            images.append(img)

        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=300,
            loop=1,
        )
