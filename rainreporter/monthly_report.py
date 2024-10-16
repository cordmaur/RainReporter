"""
This module implements the monthly report class.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

from unidecode import unidecode

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
import geopandas as gpd
import xarray as xr

from mergedownloader.downloader import Downloader
from mergedownloader.utils import DateProcessor, GISUtil
from mergedownloader.inpeparser import InpeTypes

from .mapper import Mapper
from .abstract_report import AbstractReport


class MonthlyReport(AbstractReport):
    """Docstring"""

    def __init__(
        self,
        downloader: Downloader,
        mapper: Mapper,
        shp_file: Union[str, Path],
        name: str = "",
        month_lbk: Optional[int] = 24,
        wet_month: int = 10,
    ):
        super().__init__(downloader=downloader, mapper=mapper)

        # store the variables
        self.name = name if name != "" else Path(shp_file).stem
        self.shp = gpd.read_file(shp_file)
        self.month_lbk = month_lbk if month_lbk is not None else 23
        self.wet_month = wet_month

    @classmethod
    def from_dict(
        cls,
        downloader: Downloader,
        mapper: Mapper,
        config: Dict,
        bases_folder: Optional[Union[str, Path]] = None,
    ) -> AbstractReport:
        """Create a MonthlyReport instance based on a configuration dict"""

        shp_file = Path(config["shp"])

        if bases_folder is not None and not shp_file.is_absolute():
            shp_file = Path(bases_folder) / shp_file

        return cls(
            downloader=downloader,
            mapper=mapper,
            name=config["nome"],
            shp_file=shp_file,
            month_lbk=config.get("total_meses"),
            wet_month=config["inicio_periodo_chuvoso"],
        )

    @staticmethod
    def create_report_layout() -> Tuple:
        """Create the layout and return figure and axes as a list"""
        fig = plt.figure(num=1, constrained_layout=True, figsize=(10, 10))
        fig.clear()

        gridspec = fig.add_gridspec(
            2, 3, width_ratios=[0.4, 0.3, 0.3], height_ratios=[1.2, 1]
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
            start_date=date, end_date=date, datatype=InpeTypes.MONTHLY_ACCUM_MANUAL
        ).squeeze()

        # get the raster for the long term average in the same month
        lta = self.downloader.create_cube(
            start_date=date, end_date=date, datatype=InpeTypes.MONTHLY_ACCUM
        ).squeeze()

        # make sure they have the same shape
        rain = rain.rio.reproject_match(lta)

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
        anomaly = lta.copy()
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

        plt_ax.legend(loc="lower right")

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
        last_wet_period = rain_df[rain_df.index.month == self.wet_month].index[0]  # type: ignore

        # explicitly cast last_wet_period as datetime to avoid PYLINT warnings
        last_wet_period = pd.to_datetime(last_wet_period)  # type: ignore

        # Now, with the last wet period selected, let's get all the rows up to present
        # that will be called rain of the wet period
        rain_wet = rain_df[rain_df.index >= last_wet_period]

        ### Prepare the text
        if last_date is None:
            last_date = rain_df.index[0]  # type: ignore
            last_date_str = DateProcessor.pretty_date(last_date, "%m-%Y")  # type: ignore
        else:
            last_date_str = DateProcessor.pretty_date(last_date, "%d-%m-%Y")

        accum_rain = round(rain_wet["Prec_f"].sum())
        accum_mlt = round(rain_wet["MLT_f"].sum())

        last_wet_period_str = DateProcessor.pretty_date(last_wet_period, "%m-%Y")
        accum_text = f"Prec. acum de {last_wet_period_str}"
        accum_text += f" até {last_date_str}: {accum_rain} mm"
        mlt_text = f"MLT de {last_wet_period_str} até "
        mlt_text += f"{DateProcessor.month_abrev(last_date)}: {accum_mlt} mm"  # type: ignore

        plt_ax.text(-0.50, 1, accum_text)
        plt_ax.text(-0.50, 0.96, mlt_text)
        plt_ax.text(-0.50, 0.92, "Prec. últimos 12 meses (mm)", fontsize=10)
        plt_ax.axis("off")

        rain_df = rain_df[["Mês", "MLT", "Prec"]]

        table = plt_ax.table(
            cellText=rain_df.iloc[:12].values,
            colLabels=rain_df.columns.to_list(),
            # loc="top",
            colWidths=[0.5, 0.25, 0.25],
            bbox=[-0.5, 0.075, 1.4, 0.8],  # type: ignore
        )
        # set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # set table headings
        for i in range(3):
            table[0, i].set_height(0.05)
            table[0, i].set_text_props(weight="bold")

    def _create_rain_lta_df(self, date: Union[datetime, str]) -> pd.DataFrame:
        """
        Create a dataframe with rain and lta for the monthly report
        """
        # Parse date
        date = DateProcessor.parse_date(date)

        ### Open the cubes
        # get the period to be considered
        start_month, end_month = DateProcessor.last_n_months(date, self.month_lbk)

        # get the rain
        rain = self.downloader.create_cube(
            start_month, end_month, datatype=InpeTypes.MONTHLY_ACCUM_MANUAL
        )
        # get the long term average
        lta = self.downloader.create_cube(
            start_month, end_month, datatype=InpeTypes.MONTHLY_ACCUM
        )

        ### Project the shapefile
        self.shp = self.shp.to_crs(rain.rio.crs)

        ### Get the time series of the monthly rain
        rain_ts = GISUtil.get_time_series(
            cube=rain, shp=self.shp, reducer=xr.DataArray.mean, keep_dim="time"
        )

        lta_ts = GISUtil.get_time_series(
            cube=lta, shp=self.shp, reducer=xr.DataArray.mean, keep_dim="time"
        )

        # put everything into a dataframe
        dframe = pd.DataFrame(rain_ts)
        dframe["lta"] = lta_ts.values
        dframe["basin"] = self.name

        # reset the index just to convert it to string
        dframe.reset_index(inplace=True)
        dframe.index = pd.Index(dframe["time"].astype("str").str[:7])
        dframe.index.name = "month"

        # after loading the cubes, if we are retrieving the current month
        # we need to include an observation stating the last day considered
        # for the accumulation. THe last considered date will be called last_date
        today = DateProcessor.today()
        dframe["last_date"] = None
        if (date.year == today.year) and (date.month == today.month):
            file = self.downloader.get_file(date, InpeTypes.MONTHLY_ACCUM_MANUAL)
            dset = xr.open_dataset(file)
            last_date = DateProcessor.parse_date(dset.attrs["last_day"])
            dset.close()

            dframe.iloc[-1, -1] = last_date

        dframe['last_date'] = pd.to_datetime(dframe['last_date'])
        return dframe

    def generate_report(
        self,
        date_str: str,
    ):  ## pylint: disable=arguments-differ
        """Docstring"""

        backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        ### Before doing anything, check if the given month is the actual month
        # and if we have at least 1 day for it, otherwise, generate for the previous month
        date = DateProcessor.parse_date(date_str)
        today = DateProcessor.today()

        # if we are in the current month, check if there is at least the first
        # date available, otherwise, perform the previous report
        if date.month >= today.month:
            file = self.downloader.get_file(
                date + relativedelta(day=1), datatype=InpeTypes.DAILY_RAIN
            )

            if file is None:
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
        subtitle += "MLT (INPE) calculada de 2000-2022 (23 anos)\n"

        # check if we are in the current month
        if today.month == date.month:
            subtitle += "* Chuva acumulada no mês atual até último dia disponível."

        fig.text(0.01, 1.06, subtitle, ha="left", va="top", fontsize=10)

        dframe = self._create_rain_lta_df(date)

        # if there is "last_date" info in the dataframe, it means
        # we are not accumulating the whole month and should inform that
        if (~dframe["last_date"].isna()).any():
            last_date = dframe["last_date"].values[-1]

            # convert last date to python datetime if it is in pandas format
            if not isinstance(last_date, datetime):
                last_date = pd.Timestamp(last_date).to_pydatetime()

            rep_axs[0].text(
                -0.5,
                0.05,
                f"* Prec. acumulada até {DateProcessor.pretty_date(last_date)}",
                ha="left",
                va="top",
            )
        else:
            last_date = None

        ### plot the anomaly raster
        if date.month == today.month:
            anomaly_date = date + relativedelta(months=-1)
        else:
            anomaly_date = date

        self.plot_anomaly_map(date=anomaly_date, shp=self.shp, plt_ax=rep_axs[1])

        rep_axs[-1].bar(dframe.index, dframe["pacum"])
        rep_axs[-1].plot(dframe.index, dframe["lta"], color="orange", marker="x")
        rep_axs[-1].tick_params(axis="x", labelrotation=90)
        rep_axs[-1].set_ylabel("Precipitação mensal (mm)")

        ### Write the tabular information
        self.write_tabular_monthly(
            plt_ax=rep_axs[0],
            rain_ts=dframe["pacum"],
            lta_ts=dframe["lta"],
            last_date=last_date,
        )

        matplotlib.use(backend)

        return fig, rep_axs, dframe, self.shp

    def export_report_data(  # pylint: disable=arguments-differ
        self, date: Union[str, datetime], file: Path, assets_folder: Path
    ):
        """
        The export data for the report will save the rain and long term average that
        will be used in the powerBI report.
        In addition, we need to save the anomaly map that will also be used in the powerBI.

        Args:
            date (str): The date of the report

        """

        # Load the dataframe for this report
        dframe = self._create_rain_lta_df(date)

        # Create a multi-index with month and basin
        dframe = dframe.reset_index()
        dframe.index = pd.MultiIndex.from_arrays([dframe["month"], dframe["basin"]])

        # drop month and basin columns (they are already in index)
        dframe = dframe.drop(columns=["basin", "month"])

        # Save the anomaly map for the specific date
        backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        month_str = DateProcessor.pretty_date(date, "%Y-%m")
        fig, ax = plt.subplots()
        self.plot_anomaly_map(date=date, shp=self.shp, plt_ax=ax)
        filename = unidecode(self.name).replace(" ", "_") + "-" + month_str
        fig.savefig(assets_folder / f"{filename}.png")

        dframe.loc[month_str, "anomaly_map"] = f"{filename}.png"

        # return to original backend
        matplotlib.use(backend)

        # Merge the dataframes to save to disk
        if file.exists():
            # First, let's open the dataframe
            df = pd.read_parquet(file)
            # df = pd.read_csv(file, index_col=["month", "basin"], parse_dates=["time"])

            dframe = dframe.combine_first(df)

        dframe.to_parquet(file)
