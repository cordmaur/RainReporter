"""
Main module for the reporter class
"""
import io
from pathlib import Path
from typing import Union, Iterable, Optional, List
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import abc

# import matplotlib.dates as mdates

import pyjson5

import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely import Geometry
from pyproj import Geod

from raindownloader.downloader import Downloader
from raindownloader.inpeparser import INPETypes, INPEParsers
from raindownloader.parser import BaseParser
from raindownloader.utils import DateProcessor

from .reporter_map import MapReporter


class Report(abc.ABC):
    """Abstract class to serve as base for each report"""

    @abc.abstractmethod
    def generate_report(self, *args, **kwargs):
        """Abstract method that needs to be implemented"""
        pass


class Reporter:
    """Docstring"""

    def __init__(
        self,
        server: str,
        download_folder: Union[Path, str],
        parsers: List[BaseParser] = INPEParsers.parsers,
        avoid_update: bool = True,
        post_processors: Optional[dict] = INPEParsers.post_processors,
        config_file: Optional[Union[str, Path]] = None,
    ):  # pylint: disable=dangerous-default-value
        # create a downloader instance
        self.downloader = Downloader(
            server=server,
            parsers=parsers,
            local_folder=download_folder,
            avoid_update=avoid_update,
            post_processors=post_processors,
        )

        # load the config file
        self.config = Reporter.open_config_file(config_file)

        # create an instance of MapReporter and pass to it the shapes used as backgound
        self.map_reporter = MapReporter(shapes=self.config["context_shapes"])

        self.download_folder = Path(download_folder)

    @staticmethod
    def open_config_file(config_file: Optional[Union[str, Path]] = None):
        """
        Try to locate the configuration file and return it as a dictionary
        If a file is not passed, try to locate in current directory
        """

        # transform the config file to Path
        config_file = (
            Path(config_file) if config_file is not None else Path("./reporter.json5")
        )

        # check if it exists
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found {str(config_file.absolute())}"
            )

        # open the config file
        with open(config_file, "r", encoding="utf-8") as file:
            report_config = pyjson5.decode_io(file)  # pylint: disable=no-member

        return report_config

    @staticmethod
    def create_report_layout() -> tuple:
        """Create the layout and return figure and axes as a list"""
        fig = plt.figure(constrained_layout=True, figsize=(10, 10))

        gridspec = fig.add_gridspec(
            2, 3, width_ratios=[0.3, 0.3, 0.3], height_ratios=[1.2, 1]
        )
        text_ax = fig.add_subplot(gridspec[0, 0])  # type: ignore
        raster_ax = fig.add_subplot(gridspec[0, 1:])  # type: ignore
        chart_ax = fig.add_subplot(gridspec[1, :])  # type: ignore

        rect = patches.Rectangle(
            (-0.025, -0.025),
            1.05,
            1.15,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
        fig.add_artist(rect)

        return fig, [text_ax, raster_ax, chart_ax]

    @staticmethod
    def calc_geodesic_area(geom: Geometry) -> float:
        """Calculate the geodesic area given a shapely Geometry"""
        # specify a named ellipsoid
        geod = Geod(ellps="WGS84")
        return abs(geod.geometry_area_perimeter(geom)[0]) / 1e6

    @staticmethod
    def rain_in_geoms(rain: xr.DataArray, geometries: Iterable[Geometry]):
        """
        Calculate the rain inside the given geometries and returns a dictionary.
        The geometries are shapely.Geometry type and it can be a GeoSeries from Pandas
        """

        # Let's use clip to ignore data outide the geometry
        clipped = rain.rio.clip(geometries)

        # calculate the mean height in mm
        height = float(clipped.mean())

        # calculate the area in km^2
        areas = pd.Series(map(Reporter.calc_geodesic_area, geometries))
        area = areas.sum()

        # multiply by area of geometries to get total volume em km^3
        volume = area * (height / 1e6)

        results = {"volume (kmˆ3)": volume, "area (kmˆ2)": area, "height (mm)": height}
        return results

    def plot_anomaly_map(
        self, date: Union[str, datetime], shp: gpd.GeoDataFrame, plt_ax: plt.Axes
    ) -> None:
        """Plot the anomaly map of a given month"""

        # convert date to str
        date = DateProcessor.pretty_date(date)

        # get the raster for the accumulated rain in the month
        rain = self.downloader.create_cube(
            start_date=date, end_date=date, datatype=INPETypes.MONTHLY_ACCUM_MANUAL
        ).squeeze()

        # get the rater for the long term average in the same month
        lta = self.downloader.create_cube(
            start_date=date, end_date=date, datatype=INPETypes.MONTHLY_ACCUM
        ).squeeze()

        # make sure we are comparing the same months
        assert (
            pd.to_datetime(rain.time.values).month
            == pd.to_datetime(lta.time.values).month
        )

        # get the extents of the viewport
        x_lims, y_lims = MapReporter.calc_aspects_lims(
            shp, aspect=1.0, percent_buffer=0.05
        )

        # fix the limits of the viewport
        plt_ax.set_xlim(*x_lims)
        plt_ax.set_ylim(*y_lims)

        ### Before plotting the ROI, let's plot the context shapes with z_order < 0
        self.map_reporter.plot_context_layers(plt_ax=plt_ax, z_max=0, crs=rain.rio.crs)

        # create the anomaly raster
        anomaly = rain.copy()
        anomaly.data = rain.data - lta.data

        MapReporter.plot_raster_shape(
            raster=anomaly,
            shp=shp,
            plt_ax=plt_ax,
            cmap="bwr_r",
            diverging=True,
            colorbar_label="Anomalia de chuva (mm)",
            style_kwds=self.config["shape_style"],
        )

        self.map_reporter.plot_context_layers(plt_ax=plt_ax, z_min=0, crs=rain.rio.crs)

        # write the title of the map
        month_str = DateProcessor.pretty_date(date, "%m-%Y")
        plt_ax.set_title(f"Anomalia do mês {month_str}")

    def write_tabular_monthly(
        self,
        plt_ax: plt.Axes,
        rain_ts: pd.Series,
        lta_ts: pd.Series,
        last_date: Optional[datetime] = None,
    ) -> None:
        """Write the tabular information for the monthly report"""

        # create a dataframe to plot as a table
        rain_df = rain_ts.rename("Prec_f").to_frame()
        rain_df["MLT_f"] = lta_ts.values
        rain_df["MLT"] = rain_df["MLT_f"].apply(lambda x: f"{x:.1f}")
        rain_df["Prec"] = rain_df["Prec_f"].apply(lambda x: f"{x:.1f}")
        rain_df["Mês"] = rain_df.index.astype("str")
        rain_df["Mês"] = rain_df["Mês"].str[:7]
        rain_df = rain_df.sort_index(ascending=False)
        rain_df.index = pd.DatetimeIndex(rain_df.index)

        ### write the accumulated rain since the wet period starts
        # get the months where the wet period begins
        # and extract the last period
        begin_wet_period = self.config["begin_wet_period"]
        last_wet_period = rain_df[rain_df.index.month == begin_wet_period].index[0]

        # explicitly cast last_wet_period as datetime to avoid PYLINT warnings
        last_wet_period = pd.to_datetime(last_wet_period)  # type: ignore

        # Now, with the last wet period selected, let's get all the rows up to present
        # that will be called rain of the wet period
        rain_wet = rain_df[rain_df.index >= last_wet_period]

        ### Prepare the text
        if last_date is None:
            last_date = rain_df.index[0]
            last_date_str = DateProcessor.pretty_date(last_date, "%m-%Y")
        else:
            last_date_str = DateProcessor.pretty_date(last_date, "%d-%m-%Y")

        accum_rain = round(rain_wet["Prec_f"].sum())
        accum_mlt = round(rain_wet["MLT_f"].sum())

        last_wet_period_str = DateProcessor.pretty_date(last_wet_period, "%m-%Y")
        accum_text = f"Prec. acum de {last_wet_period_str}"
        accum_text += f" até {last_date_str}: {accum_rain} mm"
        mlt_text = f"MLT de {last_wet_period_str} até "
        mlt_text += f"{DateProcessor.month_abrev(last_date)}: {accum_mlt} mm"

        plt_ax.text(0, 1, accum_text)
        plt_ax.text(0, 0.96, mlt_text)
        plt_ax.text(0, 0.92, "Prec. últimos 12 meses (mm)", fontsize=10)
        plt_ax.axis("off")

        rain_df = rain_df[["Mês", "MLT", "Prec"]]

        table = plt_ax.table(
            cellText=rain_df.iloc[:12].values,
            colLabels=rain_df.columns.to_list(),
            loc="top",
            bbox=[0.01, 0.075, 1.5, 0.8],  # type: ignore
        )
        # set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # set table headings
        for i in range(3):
            table[0, i].set_height(0.05)
            table[0, i].set_text_props(weight="bold")

    def monthly_anomaly_report(
        self,
        date_str: str,
        shapefile: Union[str, Path],
        month_lbk: int = 23,
    ):
        """Docstring"""

        ### Before doing anything, check if the given month is the actual month
        # and if we have at least 1 day for it, otherwise, generate for the previous month
        date = DateProcessor.parse_date(date_str)
        today = DateProcessor.today()

        # if we are in the current month, check if there is at least the first
        # date available, otherwise, perform the previous report
        if date.month >= today.month:
            valid_month = self.downloader.remote_file_exists(
                date + relativedelta(day=1), datatype=INPETypes.DAILY_RAIN
            )

            if not valid_month:
                # if month is does not have at least the 1st day available
                # generate the report for the previous month
                date = date + relativedelta(months=-1)
                date_str = DateProcessor.pretty_date(date)

        ### Create the layout for the report using Matplotlib Gridspec
        fig, rep_axs = Reporter.create_report_layout()
        fig.suptitle(Path(shapefile).stem, fontsize=16)

        ### Open the cubes
        # get the period to be considered
        start_month, end_month = DateProcessor.last_n_months(date_str, month_lbk)

        # get the rain
        rain = self.downloader.create_cube(
            start_month, end_month, datatype=INPETypes.MONTHLY_ACCUM_MANUAL
        )
        # get the long term average
        lta = self.downloader.create_cube(
            start_month, end_month, datatype=INPETypes.MONTHLY_ACCUM
        )

        # after loading the cubes, if we are plotting the current month
        #  we need to include an observation stating the last day considered
        # for the accumulation. THe last considered date will be called last_date
        if date.month == today.month:
            # we will check the grib files downloaded in the daily rain
            last_date = today + relativedelta(day=1)

            while self.downloader.local_file_exists(
                date=last_date, datatype=INPETypes.DAILY_RAIN
            ):
                last_date += relativedelta(days=1)

            last_date += relativedelta(days=-1)
            counter_date_str = DateProcessor.pretty_date(last_date)
            rep_axs[0].text(
                0.01,
                0.05,
                f"* Prec. acumulada até {counter_date_str}",
                ha="left",
                va="top",
            )
        else:
            last_date = None

        ### open the shapefile
        shp = gpd.read_file(shapefile)
        shp = shp.to_crs(rain.rio.crs)

        ### plot the anomaly raster
        if date.month == today.month:
            anomaly_date = date + relativedelta(months=-1)
        else:
            anomaly_date = date

        self.plot_anomaly_map(date=anomaly_date, shp=shp, plt_ax=rep_axs[1])

        ### Plot chart
        # get the time series of the monthly rain
        rain_ts = Downloader.get_time_series(
            cube=rain, shp=shp, reducer=xr.DataArray.mean, keep_dim="time"
        )

        lta_ts = Downloader.get_time_series(
            cube=lta, shp=shp, reducer=xr.DataArray.mean, keep_dim="time"
        )

        # put everything into a dataframe
        dframe = pd.DataFrame(rain_ts)
        dframe["lta"] = lta_ts.values

        # reset the index just to convert it to string
        dframe.reset_index(inplace=True)
        dframe["time"] = dframe["time"].astype("str")
        dframe.index = pd.Index(dframe["time"].str[:7])

        rep_axs[-1].bar(dframe.index, dframe["monthacum"])
        rep_axs[-1].plot(dframe.index, dframe["lta"], color="orange", marker="x")
        rep_axs[-1].tick_params(axis="x", labelrotation=90)

        ### Write the tabular information
        self.write_tabular_monthly(
            plt_ax=rep_axs[0], rain_ts=rain_ts, lta_ts=lta_ts, last_date=last_date
        )

        return rep_axs, rain_ts, lta_ts, shp

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
