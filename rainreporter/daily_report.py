"""
This module implements the monthly report class.
"""

from pathlib import Path
from typing import Union, Optional, Dict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from unidecode import unidecode

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure

import pandas as pd
import geopandas as gpd
import xarray as xr

from mergedownloader.downloader import Downloader
from mergedownloader.utils import DateProcessor, GISUtil
from mergedownloader.inpeparser import INPE_SERVER, InpeParsers, InpeTypes

from rainreporter.utils import open_json_file
from .mapper import Mapper
from .reporter import AbstractReport


class DailyReport(AbstractReport):
    """Docstring"""

    # parsers = [
    #     INPEParsers.daily_average,
    #     INPEParsers.daily_rain_parser,
    #     INPEParsers.daily_wrf,
    #     INPEParsers.hourly_wrf,
    # ]

    def __init__(
        self,
        downloader: Downloader,
        mapper: Mapper,
        shp_file: Union[str, Path],
        name: str = "",
        days_lbk: int = 30,
    ):
        super().__init__(downloader=downloader, mapper=mapper)

        # store the variables
        self.days_lbk = days_lbk
        self.name = name if name != "" else Path(shp_file).stem

        shp_file = Path(shp_file)

        if shp_file.suffix.lower() == ".parquet":
            self.shp = gpd.read_parquet(shp_file)
        else:
            self.shp = gpd.read_file(shp_file)

    def export_report_data(  # type: ignore
        self,
        date: Union[str, datetime],
        file: Path,
        assets_folder: Path,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        The export data for the report will save the daily rain and long term average that
        will be used in the powerBI report.
        In addition, we need to save the anomaly map that will also be used in the powerBI.

        Args:
            date (str): The date of the report
            file (Path): The output parquet file where the data will be saved
            assets_folder (Path): The folder where the assets (like images) will be saved

        """
        # Create the cubes with observed, average and forecast rain
        observed_cube, forecast_cube, avg_daily_cube = self._prepare_cubes(date)

        # Save the anomaly map for the specific date
        backend = mpl.get_backend()
        mpl.use("Agg")

        fig = self.plot_rain_anomaly(observed_cube, avg_daily_cube)
        filename = unidecode(self.name).replace(" ", "_") + "_30_days.png"
        fig.savefig((assets_folder / filename).as_posix())
        plt.close(fig)

        fig_forecast = self.plot_rain_forecast(forecast_cube)
        filename_forecast = unidecode(self.name).replace(" ", "_") + "_forecast.png"
        fig_forecast.savefig((assets_folder / filename_forecast).as_posix())
        plt.close(fig_forecast)

        mpl.use(backend)

        # Create the dataframe
        observed_series = self._prepare_dataframe(
            observed_cube, avg_daily_cube, forecast_cube
        )

        if file.exists():
            existing_df = pd.read_parquet(file)
            if "time" in existing_df.columns:
                existing_df = existing_df.set_index(["time", "basin"])
            observed_series = observed_series.set_index(["time", "basin"])
            observed_series = observed_series.combine_first(existing_df)
            observed_series = observed_series.reset_index()


        # One last adjust before saving the data to parquet. We need to adjust the "time" column to be in the format of "YYYY-MM-DD"
        # without the time component, since we are dealing with daily data.
        # The last file downloaded refers to the rain of the previous day, however it should be displayed as the
        # file day at 00:00. Therefore, we need to shift the time by 12 hours to match the expected format.
        observed_series["time"] = observed_series["time"] - pd.Timedelta(hours=12)

        observed_series.to_parquet(file)

        # return the cubes to match function signature return type, even though it's not used in snippet
        # But wait, signature says tuple of DataArray.
        return observed_cube, forecast_cube, avg_daily_cube

    def _prepare_dataframe(
        self,
        observed_cube: xr.DataArray,
        avg_daily_cube: xr.DataArray,
        forecast_cube: xr.DataArray,
    ) -> pd.DataFrame:
        """Prepare the dataframe with observed, average and forecast values"""
        # First cut all cubes by the geometry
        observed_cube = GISUtil.cut_cube_by_geoms(observed_cube, self.shp.geometry)
        avg_daily_cube = GISUtil.cut_cube_by_geoms(avg_daily_cube, self.shp.geometry)
        forecast_cube = GISUtil.cut_cube_by_geoms(forecast_cube, self.shp.geometry)

        # Create the observed series
        observed_series = observed_cube.mean(
            dim=["latitude", "longitude"]
        ).to_dataframe()
        observed_series = observed_series.drop(
            columns=["surface", "valid_time", "spatial_ref"], errors="ignore"
        )

        # Create the average series and concatenate with the observed series
        average_series = avg_daily_cube.mean(
            dim=["latitude", "longitude"]
        ).to_dataframe()["pmed"]
        observed_series = pd.concat([observed_series, average_series], axis=1)

        # Create the forecast series and concatenate with the observed series
        forecast_series = forecast_cube.mean(
            dim=["latitude", "longitude"]
        ).to_dataframe()
        observed_series["forecast"] = forecast_series["prec"]

        # Add the basin name to the series and reorder the columns
        observed_series["basin"] = self.name
        observed_series = observed_series[
            ["basin", "prec", "pmed", "forecast"]
        ].reset_index()

        return observed_series

    def _prepare_cubes(
        self, date: Union[str, datetime]
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Prepare the observed, forecast and average cubes"""
        # get the dates to process
        end_date = DateProcessor.parse_date(date)
        start_date = end_date - timedelta(days=self.days_lbk)

        # Create the cube of observed rain
        observed_cube = self.downloader.create_cube(
            start_date=start_date,
            end_date=end_date,
            datatype=InpeTypes.DAILY_RAIN,
        )

        # OBS: if end_date is not available, it will not be included in the cube, and it will have the correct length of
        # days as expected. However, if it is available, we will have lkb_days + 1 in the series, so we need to drop the first date.
        # the easiest way to achieve this is to get the last lkb_days from the series.
        observed_cube = observed_cube.isel(time=slice(-self.days_lbk, None))

        # another adjust we need to do refers to the exact day when the rain happened.
        # In the case of the daily rain, the rain refers to the previous day, until 9am of the current day (12pm - 3h).
        # Therefore, we need to shift the dates by one day back.
        observed_cube["time"] = observed_cube["time"] - np.timedelta64(1, "D")

        # re-adjust start and end_dates
        start_date = DateProcessor.parse_date(observed_cube["time"].values[0])
        end_date = DateProcessor.parse_date(observed_cube["time"].values[-1])

        # Now, get the forecast cube
        forecast_cube = self.downloader.open_file(
            date=end_date + timedelta(days=1),  # the first forecast day is the next day
            datatype=InpeTypes.DAILY_WRF,
        )

        if forecast_cube is None:
            raise ValueError(
                f"Could not open the forecast file for date {end_date + timedelta(days=1)}"
            )

        # Last, create the cube with the average daily rain
        avg_daily_cube = self.downloader.create_cube(
            start_date=start_date,
            end_date=forecast_cube["time"].values[-1],
            datatype=InpeTypes.DAILY_AVERAGE,
        )

        # adjust the time dimension of the forecast cube to match the observed cube
        time_dim = np.concatenate(
            [observed_cube["time"].values, forecast_cube["time"].values]
        )
        avg_daily_cube["time"] = time_dim

        return observed_cube, forecast_cube, avg_daily_cube

    def plot_rain_anomaly(
        self,
        observed_cube: xr.DataArray,
        avg_daily_cube: xr.DataArray,
    ) -> Figure:
        """Plot the rain anomaly map"""
        observed_rain = observed_cube.sum(dim="time")
        expected_rain = avg_daily_cube.sel(time=observed_cube["time"]).sum(dim="time")
        observed_rain = observed_rain.rio.reproject_match(expected_rain)
        anomaly = expected_rain.copy()
        anomaly.data = observed_rain.data - expected_rain.data

        xlims, ylims = Mapper.calc_aspects_lims(
            self.shp, aspect=1.0, percent_buffer=0.05
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        self.mapper.plot_context_layers(plt_ax=ax, z_max=0, crs=anomaly.rio.crs)

        Mapper.plot_raster_shape(
            raster=anomaly,
            plt_ax=ax,
            shp=self.shp,
            cmap="bwr_r",
            diverging=True,
            colorbar_label="Anomalia de Chuva (mm)",
            style_kwds=self.mapper.config,
        )

        self.mapper.plot_context_layers(plt_ax=ax, z_min=0, crs=expected_rain.rio.crs)

        start_date = DateProcessor.pretty_date(observed_cube.time.values[0])
        end_date = DateProcessor.pretty_date(observed_cube.time.values[-1])

        ax.set_title(
            f"Anomalia de Chuva dos últimos {self.days_lbk} dias\n{start_date} - {end_date}"
        )

        return fig

    def plot_rain_forecast(
        self,
        forecast_cube: xr.DataArray,
    ) -> Figure:
        """Plot the rain forecast map"""
        forecast_rain = forecast_cube.sum(dim="time")

        xlims, ylims = Mapper.calc_aspects_lims(
            self.shp, aspect=1.0, percent_buffer=0.05
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        self.mapper.plot_context_layers(plt_ax=ax, z_max=0, crs=forecast_rain.rio.crs)

        Mapper.plot_raster_shape(
            raster=forecast_rain,
            plt_ax=ax,
            shp=self.shp,
            cmap="Blues",
            diverging=False,
            colorbar_label="Previsão de Chuva (mm)",
            style_kwds=self.mapper.config,
        )

        self.mapper.plot_context_layers(plt_ax=ax, z_min=0, crs=forecast_rain.rio.crs)

        start_date = DateProcessor.pretty_date(forecast_cube.time.values[0])
        end_date = DateProcessor.pretty_date(forecast_cube.time.values[-1])

        ax.set_title(f"Previsão de Chuva\n{start_date} - {end_date}")

        return fig

    @classmethod
    def from_json_file(
        cls,
        downloader: Downloader,
        mapper: Mapper,
        json_file: Union[str, Path],
        bases_folder: Optional[Union[str, Path]] = None,
    ):
        """Create a MonthlyReport instance based on the json file"""
        config = open_json_file(json_file)

        return cls.from_dict(
            downloader=downloader,
            mapper=mapper,
            config=config,
            bases_folder=bases_folder,
        )

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
            days_lbk=config.get("total_dias", 30),
        )

    @staticmethod
    def create_report_layout() -> tuple:
        """Create the layout and return figure and axes as a list"""
        fig = plt.figure(num=1, constrained_layout=True, figsize=(10, 10))
        fig.clear()

        gridspec = fig.add_gridspec(
            3, 2, width_ratios=[0.3, 0.7], height_ratios=[1.2, 0.5, 0.5]
        )
        text_ax = fig.add_subplot(gridspec[0, 0])  # type: ignore
        raster_ax = fig.add_subplot(gridspec[0, 1:])  # type: ignore
        chart1_ax = fig.add_subplot(gridspec[1, :])  # type: ignore
        chart2_ax = fig.add_subplot(gridspec[2, :])  # type: ignore

        rect = patches.Rectangle(
            (-0.025, -0.025),
            1.05,
            1.15,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
        fig.add_artist(rect)

        return fig, [text_ax, raster_ax, chart1_ax, chart2_ax]

    def plot_anomaly_map(
        self, rain: xr.DataArray, avg_rain: xr.DataArray, plt_ax: plt.Axes
    ) -> None:
        """Plot the anomaly map of a given month"""

        # get the extents of the viewport
        x_lims, y_lims = Mapper.calc_aspects_lims(
            self.shp, aspect=1.0, percent_buffer=0.05
        )

        # fix the limits of the viewport
        plt_ax.set_xlim(*x_lims)
        plt_ax.set_ylim(*y_lims)

        ### Before plotting the ROI, let's plot the context shapes with z_order < 0
        self.mapper.plot_context_layers(plt_ax=plt_ax, z_max=0, crs=rain.rio.crs)

        # create the anomaly raster
        anomaly_rain = rain.sum(dim="time") - avg_rain.sum(dim="time")

        Mapper.plot_raster_shape(
            raster=anomaly_rain,
            shp=self.shp,
            plt_ax=plt_ax,
            cmap="bwr_r",
            diverging=True,
            colorbar_label="Anomalia de chuva (mm)",
            style_kwds=self.mapper.config,
        )

        self.mapper.plot_context_layers(plt_ax=plt_ax, z_min=0, crs=rain.rio.crs)

        plt_ax.legend(loc="lower right")

        # write the title of the map
        plt_ax.set_title("Anomalia últimos 30 dias")

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
        last_wet_period = rain_df[rain_df.index.month == 10].index[0]  # type: ignore

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
        date_str: Optional[str] = None,
    ):  ## pylint: disable=arguments-differ
        """Generate the daily report"""

        # first thing is to obtain start and end dates
        # for that, let's check the last available date
        today = DateProcessor.today()

        end_date = today if date_str is None else DateProcessor.parse_date(date_str)

        date_available = False
        while not date_available:
            # check if the date is available
            # if not, we will try the previous day
            file = self.downloader.get_file(end_date, InpeTypes.DAILY_RAIN)

            if file is None:
                end_date = end_date - timedelta(days=1)
            else:
                date_available = True

        start_date = end_date - timedelta(days=self.days_lbk)

        ### Create the cubes
        # create the cube with the daily rain
        rain = self.downloader.create_cube(
            start_date=start_date, end_date=end_date, datatype=InpeTypes.DAILY_RAIN
        )

        # create a cube with the average rain in the period
        avg_rain = self.downloader.create_cube(
            start_date=start_date, end_date=end_date, datatype=InpeTypes.DAILY_AVERAGE
        )

        # get the forecast
        rain_fcst = self.downloader.create_cube(
            start_date=end_date + timedelta(days=1),
            end_date=end_date + timedelta(days=7),
            datatype=InpeTypes.DAILY_WRF,
            ref_date=end_date,
        )

        # get average rain in forecast days
        avg_rain_fcst = self.downloader.create_cube(
            start_date=end_date + timedelta(days=1),
            end_date=end_date + timedelta(days=7),
            datatype=InpeTypes.DAILY_AVERAGE,
        )

        ### Align the cubes
        # Project the shapefile
        self.shp = self.shp.to_crs(rain.rio.crs)

        # correct the coordinates so rain and avg_rain are aligned
        avg_rain = avg_rain.assign_coords({"time": rain.time})
        rain = rain.sel(
            {"longitude": avg_rain.longitude, "latitude": avg_rain.latitude},
            tolerance=0.02,
            method="nearest",
        ).assign_coords(
            {"longitude": avg_rain.longitude, "latitude": avg_rain.latitude}
        )

        ### Create the layout for the report using Matplotlib Gridspec
        fig, rep_axs = DailyReport.create_report_layout()

        subtitle = f"Relatório gerado em: {DateProcessor.pretty_date(today)}\n"
        # subtitle += "MLT (INPE) calculada de 2000-2022 (23 anos)\n"
        fig.text(0.01, 1.06, subtitle, ha="left", va="top", fontsize=10)

        title = f"Bacia: {self.name} / Dia: {DateProcessor.pretty_date(end_date)}"
        fig.suptitle(title, y=1.1, fontsize=14)

        ### Plot the map
        self.plot_anomaly_map(rain=rain, avg_rain=avg_rain, plt_ax=rep_axs[1])

        ### Plot the chart
        self.plot_charts(
            rain_cube=rain,
            avg_rain_cube=avg_rain,
            rain_fcst=rain_fcst,
            avg_rain_fcst=avg_rain_fcst,
            plt_ax1=rep_axs[2],
            plt_ax2=rep_axs[3],
        )

        return fig, rep_axs, rain, avg_rain, rain_fcst, avg_rain_fcst

    def plot_daily_chart(
        self,
        rain_ts: pd.Series,
        avg_rain_ts: pd.Series,
        fcst_ts: pd.Series,
        avg_fcst_ts: pd.Series,
        plt_ax: plt.Axes,
    ):
        """Plot daily chart"""
        # plot the daily chart
        plt_ax.axvline(x=rain_ts.index[-1], color="red", label="hoje")  # type: ignore
        plt_ax.bar(rain_ts.index, rain_ts.values, color="blue", label="Chuva observada")  # type: ignore
        plt_ax.bar(fcst_ts.index, fcst_ts.values, color="orange", label="Chuva prevista")  # type: ignore

        ext_avg_rain_ts = pd.concat([avg_rain_ts, avg_fcst_ts])
        plt_ax.axhline(
            y=float(ext_avg_rain_ts.mean()), color="black", label="Chuva média"
        )

        plt_ax.set_ylabel("Chuva (mm)")
        plt_ax.set_title(f"Chuva {self.days_lbk}+7 dias")

        plt_ax.legend()

    def plot_accum_chart(
        self,
        rain_ts: pd.Series,
        avg_rain_ts: pd.Series,
        fcst_ts: pd.Series,
        avg_fcst_ts: pd.Series,
        plt_ax: plt.Axes,
    ):
        """Plot accumulated chart"""
        # plot the accum chart
        # create the extended rain+forecast
        ext_rain_ts = pd.concat([rain_ts, fcst_ts])
        plt_ax.axvline(x=rain_ts.index[-1], color="red", label="hoje")  # type: ignore
        plt_ax.plot(
            ext_rain_ts.index,
            ext_rain_ts.cumsum(),
            color="blue",
            label="Chuva observada",
        )

        ext_avg_rain_ts = pd.concat([avg_rain_ts, avg_fcst_ts])

        plt_ax.plot(
            ext_avg_rain_ts.index,
            ext_avg_rain_ts.cumsum(),
            color="black",
            label="Chuva média",
        )

        # plot the forecast in another color
        adjusted_fcst_ts = pd.concat([rain_ts.cumsum()[-1:], fcst_ts])
        plt_ax.plot(
            adjusted_fcst_ts.index,
            adjusted_fcst_ts.cumsum(),
            color="orange",
            label="Chuva prevista",
        )

        plt_ax.set_ylabel("Chuva acumulada (mm)")
        plt_ax.set_title(f"Chuva acumulada {self.days_lbk}+7 dias")

        plt_ax.legend()

    def plot_charts(
        self,
        rain_cube: xr.DataArray,
        avg_rain_cube: xr.DataArray,
        rain_fcst: xr.DataArray,
        avg_rain_fcst: xr.DataArray,
        plt_ax1: plt.Axes,
        plt_ax2: plt.Axes,
    ):
        """Plot the charts"""

        # to plot the chart we need all the time series
        # the good news is that the arrays don't need to be aligned
        # rain_ts = Downloader.get_time_series(
        #     rain_cube, self.shp, reducer=xr.DataArray.mean
        # )
        # avg_rain_ts = Downloader.get_time_series(
        #     avg_rain_cube, self.shp, reducer=xr.DataArray.mean
        # )

        # fcst_ts = Downloader.get_time_series(
        #     rain_fcst, self.shp, reducer=xr.DataArray.mean
        # )
        # avg_fcst_ts = Downloader.get_time_series(
        #     avg_rain_fcst, self.shp, reducer=xr.DataArray.mean
        # )

        # # make index equal
        # avg_fcst_ts.index = fcst_ts.index

        # # plot the daily chart
        # self.plot_daily_chart(
        #     rain_ts=rain_ts,
        #     avg_rain_ts=avg_rain_ts,
        #     fcst_ts=fcst_ts,
        #     avg_fcst_ts=avg_fcst_ts,
        #     plt_ax=plt_ax1,
        # )

        # # plot accumulated chart
        # self.plot_accum_chart(
        #     rain_ts=rain_ts,
        #     avg_rain_ts=avg_rain_ts,
        #     fcst_ts=fcst_ts,
        #     avg_fcst_ts=avg_fcst_ts,
        #     plt_ax=plt_ax2,
        # )
