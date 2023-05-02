"""
This module implements the monthly report class. 
"""
from pathlib import Path
from typing import Union, Optional, Dict
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyjson5

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
import geopandas as gpd
import xarray as xr

from raindownloader.downloader import Downloader
from raindownloader.utils import DateProcessor
from raindownloader.inpeparser import INPEParsers, INPETypes
from .mapper import Mapper
from .reporter import AbstractReport


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
        raise FileNotFoundError(f"Config file not found {str(config_file.absolute())}")

    # open the config file
    with open(config_file, "r", encoding="utf-8") as file:
        report_config = pyjson5.decode_io(file)  # pylint: disable=no-member

    return report_config


class MonthlyReport(AbstractReport):
    """Docstring"""

    parsers = [
        INPEParsers.monthly_accum,
        INPEParsers.month_accum_manual,
        INPEParsers.daily_rain_parser,
    ]

    def __init__(
        self,
        downloader: Downloader,
        mapper: Mapper,
        shp_file: str,
        name: str = "",
        month_lbk: Optional[int] = 23,
        wet_month: int = 10,
    ):
        super().__init__(downloader=downloader, mapper=mapper)

        # store the variables
        self.name = name if name != "" else Path(shp_file).stem
        self.shp = gpd.read_file(shp_file)
        self.month_lbk = month_lbk if month_lbk is not None else 23
        self.wet_month = wet_month

    @classmethod
    def from_json_file(
        cls, downloader: Downloader, mapper: Mapper, json_file: Union[str, Path]
    ):
        """Create a MonthlyReport instance based on the json file"""
        config = open_config_file(json_file)

        return cls.from_dict(downloader=downloader, mapper=mapper, config=config)

    @classmethod
    def from_dict(
        cls, downloader: Downloader, mapper: Mapper, config: Dict
    ) -> AbstractReport:
        """Create a MonthlyReport instance based on a configuration dict"""

        return cls(
            downloader=downloader,
            mapper=mapper,
            name=config["nome"],
            shp_file=config["shp"],
            month_lbk=config.get("total_meses"),
            wet_month=config["inicio_periodo_chuvoso"],
        )

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

    def plot_anomaly_map(
        self, date: Union[str, datetime], shp: gpd.GeoDataFrame, plt_ax: plt.Axes
    ) -> None:
        """Plot the anomaly map of a given month"""

        # convert date to str
        date = DateProcessor.pretty_date(date, "%Y-%m")

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
        x_lims, y_lims = Mapper.calc_aspects_lims(shp, aspect=1.0, percent_buffer=0.05)

        # fix the limits of the viewport
        plt_ax.set_xlim(*x_lims)
        plt_ax.set_ylim(*y_lims)

        ### Before plotting the ROI, let's plot the context shapes with z_order < 0
        self.mapper.plot_context_layers(plt_ax=plt_ax, z_max=0, crs=rain.rio.crs)

        # create the anomaly raster
        anomaly = rain.copy()
        anomaly.data = rain.data - lta.data

        Mapper.plot_raster_shape(
            raster=anomaly,
            shp=shp,
            plt_ax=plt_ax,
            cmap="bwr_r",
            diverging=True,
            colorbar_label="Anomalia de chuva (mm)",
            style_kwds=self.mapper.config,
        )

        self.mapper.plot_context_layers(plt_ax=plt_ax, z_min=0, crs=rain.rio.crs)

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
        last_wet_period = rain_df[rain_df.index.month == self.wet_month].index[0]

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

    def generate_report(
        self,
        date_str: str,
    ):  ## pylint: disable=arguments-differ
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
                date = date - relativedelta(months=1)

        # adjust the date_str
        date_str = DateProcessor.pretty_date(date, "%m-%Y")

        ### Create the layout for the report using Matplotlib Gridspec
        fig, rep_axs = MonthlyReport.create_report_layout()

        title = f"Bacia: {self.name} / Mês: {date_str}"
        fig.suptitle(title, y=1.1, fontsize=14)

        subtitle = f"Relatório gerado em: {DateProcessor.pretty_date(today)}\n"

        # check if we are in the current month
        if today.month == date.month:
            subtitle += "* Chuva acumulada no mês atual até último dia disponível."

        fig.text(0.01, 1.06, subtitle, ha="left", va="top", fontsize=12)

        ### Open the cubes
        # get the period to be considered
        start_month, end_month = DateProcessor.last_n_months(date_str, self.month_lbk)

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

        ### Project the shapefile
        self.shp = self.shp.to_crs(rain.rio.crs)

        ### plot the anomaly raster
        if date.month == today.month:
            anomaly_date = date + relativedelta(months=-1)
        else:
            anomaly_date = date

        self.plot_anomaly_map(date=anomaly_date, shp=self.shp, plt_ax=rep_axs[1])

        ### Plot chart
        # get the time series of the monthly rain
        rain_ts = Downloader.get_time_series(
            cube=rain, shp=self.shp, reducer=xr.DataArray.mean, keep_dim="time"
        )

        lta_ts = Downloader.get_time_series(
            cube=lta, shp=self.shp, reducer=xr.DataArray.mean, keep_dim="time"
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

        return rep_axs, rain_ts, lta_ts, self.shp
