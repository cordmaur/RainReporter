"""
Main module for the reporter class
"""
import io
import logging
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, Dict, Type, Any

from PIL import Image
import matplotlib.pyplot as plt

from shapely import Geometry
from pyproj import Geod
import xarray as xr

from pypdf import PdfMerger, PdfReader

from raindownloader.downloader import Downloader
from raindownloader.utils import DateProcessor
from raindownloader.inpeparser import INPEParsers

from .utils import open_json_file

from .abstract_report import AbstractReport
from .monthly_report import MonthlyReport
from .mapper import Mapper


class Reporter:
    """Docstring"""

    templates: Dict[str, type[AbstractReport]] = {
        "Mensal": MonthlyReport,
    }

    def __init__(
        self,
        server: str,
        download_folder: Union[Path, str],
        avoid_update: bool = True,
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
        # get the parsers necessary for the reports
        parsers = set()
        for template in Reporter.templates.values():
            parsers = parsers.union(set(template.parsers))

        # create a downloader instance
        self.downloader = Downloader(
            server=server,
            parsers=parsers,  # type: ignore
            local_folder=download_folder,
            avoid_update=avoid_update,
            post_processors=INPEParsers.post_processors,
            log_level=log_level,
        )

        # create the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()
        self.logger.addHandler(self.downloader.logger.handlers[0])

        # load the config file
        self.logger.info("Configuring Reporter class")
        self.config = Reporter.open_config_file(config_file)

        self.logger.info(self.config)

        # create an instance of MapReporter and pass to it the shapes used as backgound
        self.logger.info("Creating a MapReporter instance")
        self.mapper = Mapper(
            config=self.config["shape_style"], shapes=self.config["context_shapes"]
        )

        self.download_folder = Path(download_folder)
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

    def generate_report(
        self, rep_type: Union[str, Type[AbstractReport]], date_str: str, **kwargs
    ):
        """Docstring"""

        if isinstance(rep_type, str):
            template = Reporter.templates[rep_type]
        else:
            template = rep_type

        self.logger.info("Generating report for date %s", date_str)
        self.logger.info("Using report template %s", template)
        self.logger.info("Configurations: %s", kwargs)
        report = template(downloader=self.downloader, mapper=self.mapper, **kwargs)

        return report.generate_report(date_str=date_str)

    def generate_pdf(
        self, json_file: Union[Path, str, Dict], output_folder: Union[Path, str]
    ):
        """Generate a PDF report based on a json file specification"""

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

        pdf_doc = PdfMerger()
        for report_config in pdf_config["relatorios"]:
            try:
                # get the appropriate template
                template = Reporter.templates[report_config["tipo"]]

                # create the report instance using the configuration
                report = template.from_dict(
                    downloader=self.downloader,
                    mapper=self.mapper,
                    config=report_config,
                    bases_folder=self.bases_folder,
                )

                result = report.generate_report(date_str=date_str)  # type: ignore

                # get axes and figure
                plt_axs = result[0]
                fig = plt_axs[0].figure

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
            self.generate_pdf(json_file=file, output_folder=output_folder)

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

    # @staticmethod
    # def rain_in_geoms(rain: xr.DataArray, geometries: Iterable[Geometry]):
    #     """
    #     Calculate the rain inside the given geometries and returns a dictionary.
    #     The geometries are shapely.Geometry type and it can be a GeoSeries from Pandas
    #     """

    #     # Let's use clip to ignore data outide the geometry
    #     clipped = rain.rio.clip(geometries)

    #     # calculate the mean height in mm
    #     height = float(clipped.mean())

    #     # calculate the area in km^2
    #     areas = pd.Series(map(Reporter.calc_geodesic_area, geometries))
    #     area = areas.sum()

    #     # multiply by area of geometries to get total volume em km^3
    #     volume = area * (height / 1e6)

    #     results = {"volume (kmˆ3)": volume, "area (kmˆ2)": area, "height (mm)": height}
    #     return results

    # @staticmethod
    # def write_tabular_info(plt_ax: plt.Axes, stats: dict):
    #     """Prepare the tabular section of the report"""

    #     plt_ax.axis("off")

    #     start_date = DateProcessor.pretty_date(stats["start_date"])
    #     end_date = DateProcessor.pretty_date(stats["end_date"])

    #     title_str = f"Período: de {start_date} até {end_date}"
    #     plt_ax.text(0.04, 0.97, title_str, fontsize=12)

    #     area_str = f"Área  total: {stats['area (kmˆ2)']:.2f} km²"
    #     plt_ax.text(0, 0.90, area_str)

    #     rain_str = f"Chuva acumulada na bacia: {stats['height (mm)']:.0f} mm"
    #     plt_ax.text(0, 0.85, rain_str)

    #     accum_str = f"Chuva esperada na bacia: {stats['mean height (mm)']:.0f} mm"
    #     plt_ax.text(0, 0.80, accum_str)

    #     volume_str = f"Volume de chuva na bacia: {stats['volume (kmˆ3)']:.0f} km³"
    #     plt_ax.text(0, 0.75, volume_str)

    #     volume_str = (
    #         f"Volume esperado de chuva na bacia: {stats['mean volume (kmˆ3)']:.0f} km³"
    #     )
    #     plt_ax.text(0, 0.7, volume_str)

    # def daily_rain_report(
    #     self,
    #     start_date: str,
    #     end_date: str,
    #     shapefile: Union[str, Path],
    # ):
    #     """
    #     Create a rain report for the given period and shapefile (can have multiple features)
    #     """

    #     # first, let's grab the daily rain in the period
    #     cube = self.downloader.create_cube(
    #         start_date=start_date, end_date=end_date, datatype=INPETypes.DAILY_RAIN
    #     )

    #     # accumulate the rain in the time axis
    #     rain = cube.sum(dim="time")

    #     # then, open the shapefile
    #     shp = gpd.read_file(shapefile)

    #     # check if there is something in the shapefile
    #     if len(shp) == 0:
    #         raise ValueError("No elements in the input shapefile")

    #     if len(shp) > 1:
    #         print(f"{len(shp)} featuers found in shapefile, selecting all of them.")

    #     # convert the shapefile to the raster CRS (more cost effective)
    #     shp.to_crs(rain.rio.crs, inplace=True)

    #     ### Create the layout for the report using Matplotlib Gridspec
    #     fig, rep_axs = RainReporter.create_report_layout()
    #     fig.suptitle(Path(shapefile).stem, fontsize=16)

    #     ### Plot the map with the accumulated rain
    #     self.plot_raster_shape(raster=rain, shp=shp, plt_ax=rep_axs[1])

    #     ### Add cities and state boundaries
    #     self.map_reporter.plot_states(plt_ax=rep_axs[1])
    #     self.map_reporter.plot_cities(plt_ax=rep_axs[1])

    #     ### Plot the daily rain graph
    #     daily_rain = Downloader.get_time_series(
    #         cube=cube, shp=shp, reducer=xr.DataArray.mean
    #     )

    #     # plot the bars
    #     RainReporter.plot_daily_rain(plt_ax=rep_axs[2], time_series=daily_rain)

    #     ### Plot the daily average rain
    #     # get the daily average cube
    #     avg_cube = self.downloader.create_cube(
    #         start_date=start_date, end_date=end_date, datatype=INPETypes.DAILY_AVERAGE
    #     )

    #     # get the time series of the daily average within the basin
    #     daily_average = Downloader.get_time_series(
    #         cube=avg_cube, shp=shp, reducer=xr.DataArray.mean
    #     )

    #     # at the end, make sure the indices are equivalent
    #     daily_average.index = cube["time"].data

    #     # Plot the line
    #     RainReporter.plot_daily_average(plt_ax=rep_axs[2], time_series=daily_average)

    #     # turn on the legend
    #     rep_axs[2].legend()

    #     ### write the tabular text of the report
    #     rain_stats = self.rain_in_geoms(rain, shp.geometry)
    #     mean_height = daily_average.sum()
    #     mean_volume = mean_height * rain_stats["area (kmˆ2)"] / 1e6
    #     rain_stats.update(
    #         {
    #             "start_date": start_date,
    #             "end_date": end_date,
    #             "mean height (mm)": mean_height,
    #             "mean volume (kmˆ3)": mean_volume,
    #         }
    #     )
    #     RainReporter.write_tabular_info(plt_ax=rep_axs[0], stats=rain_stats)

    #     return rep_axs, rain, shp, cube
    # @staticmethod
    # def plot_daily_rain(plt_ax: plt.Axes, time_series: pd.Series):
    #     """Create the plot with the daily rain in the period"""

    #     plt_ax.bar(x=time_series.index, height=time_series.to_list(), label="Chuva")

    #     # format the x-axis labels
    #     date_format = mdates.DateFormatter("%d/%m")
    #     plt_ax.xaxis.set_major_formatter(date_format)
    #     plt.xticks(rotation=60, ha="right")

    #     # get the years
    #     dates = pd.Series(time_series.index)
    #     plt_ax.set_xlabel(f"Chuva média na bacia - Ano: {list(dates.dt.year.unique())}")

    #     plt_ax.set_ylabel("Chuva média na bacia (mm)")
    #     plt_ax.set_title("Chuva Diária Média na Bacia")

    # @staticmethod
    # def plot_daily_average(plt_ax: plt.Axes, time_series: pd.Series):
    #     """Plot the daily average as line, for reference"""

    #     plt_ax.plot(
    #         time_series.index,
    #         time_series.values,
    #         label="Média diária",
    #         color="orange",
    #     )

    # @staticmethod
    # def plot_monthly_rain_gutto(
    #     plt_ax: plt.Axes,
    #     cube: xr.DataArray,
    #     shp: gpd.GeoDataFrame,
    #     plot_mlt: bool = True,
    #     **kw_formatting,
    # ):
    #     """Plot the monthly rain in a given axes"""

    #     # get the time series
    #     time_series = Downloader.get_time_series(
    #         cube=cube, shp=shp, reducer=xr.DataArray.mean, keep_dim="time"
    #     )

    #     time_series.plot(ax=plt_ax, legend=True, **kw_formatting)

    #     if plot_mlt:
    #         # create a dataframe
    #         dframe = time_series.to_frame().reset_index()

    #         mlt = (
    #             dframe.groupby(by=dframe["time"].dt.month)
    #             .mean()
    #             .rename(columns={time_series.name: "MLT"})
    #         )

    #         # reindex DF with the month numbers
    #         dframe.set_index(dframe["time"].dt.month.values)
    #         dframe = dframe.join(mlt)
    #         dframe.set_index("time", inplace=True)

    #         dframe.plot(
    #             y="MLT",
    #             color="orange",
    #             linestyle="--",
    #             marker="o",
    #             ax=plt_ax,
    #             linewidth=0.9,
    #         )

    #         return mlt
