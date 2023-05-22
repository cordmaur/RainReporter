"""The MapReporter class is the responsible for plotting all the maps in the reports"""
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from adjustText import adjust_text

from shapely import box
import geopandas as gpd
import rasterio as rio
import xarray as xr
import contextily as cx

from raindownloader.utils import GISUtil
from raindownloader.inpeparser import INPE


class Mapper:
    """This class is responsible for plotting the maps"""

    def __init__(self, config: Dict, shapes: Dict):
        # Load the shapes in memory and store them as a dictionary
        for shape in shapes:
            file = Path(shapes[shape]["file"])

            if not file.exists() and not file.is_absolute():
                file = (Path(__file__).parent) / file

            print(file)
            shapes[shape]["gdf"] = gpd.read_file(file)

        self.shapes = shapes
        self.config = config

    @staticmethod
    def bounds(
        shp: gpd.GeoDataFrame, percent_buffer: float = 0, fixed_buffer: float = 0.0
    ) -> tuple:
        """
        Return the total bounds of a shape file with a given buffer
        The buffer can be a fixed distance (in projection units)
        or a percentage of the maximum size
        """

        # get the bounding box of the total shape
        bbox = box(*shp.total_bounds)

        if fixed_buffer != 0:
            bbox = bbox.buffer(fixed_buffer)
        elif percent_buffer != 0:
            xmin, ymin, xmax, ymax = bbox.bounds
            delta_x = xmax - xmin
            delta_y = ymax - ymin
            diag = (delta_x**2 + delta_y**2) ** 0.5
            bbox = bbox.buffer(percent_buffer * diag)

        return bbox.bounds

    @staticmethod
    def calc_aspects_lims(
        shp: gpd.GeoDataFrame,
        aspect: float = 1.0,
        percent_buffer: float = 0,
        fixed_buffer: float = 0.0,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate the limits of a viewport given a shape, a buffer around the shape
        and an aspect ratio: aspect_ratio = lim_x/lim_y
        the buffer can be expressed as crs units or percentage
        """

        # first, let's get the bounding box
        xmin, ymin, xmax, ymax = Mapper.bounds(
            shp, percent_buffer=percent_buffer, fixed_buffer=fixed_buffer
        )

        # calc the sizes in each dimension
        size_x = xmax - xmin
        size_y = ymax - ymin

        actual_aspect = size_x / size_y

        # if actual aspect is smaller, that means width has to be increased
        if actual_aspect < aspect:
            # we have to increase X accordingly
            delta = size_y * aspect - size_x
            xmin -= delta / 2
            xmax += delta / 2

        # if actual aspect is greater, that means height has to be increased
        else:
            # we have to increase Y axis accordingly
            delta = size_x / aspect - size_y
            ymin -= delta / 2
            ymax += delta / 2

        # return the limits
        return (xmin, xmax), (ymin, ymax)

    @staticmethod
    def filter_shape(
        shp: gpd.GeoDataFrame,
        shape: Dict,
    ) -> gpd.GeoDataFrame:
        """
        Create a query based on the shape configuration.
        It is based on the fields 'count', 'column' and 'sort_order'
        If all of them appear in the shape, we create a query to filter accordingly.
        """
        # if count is specified, it is needed also the column and the sort order
        if "count" in shape and "column" in shape and "sort_order" in shape:
            # get the count and the column to apply the filter to
            column = shape["column"]
            count = shape["count"]

            # identifies if the user want the top bigger or the top smaller
            sort_order = shape["sort_order"].lower()
            order = "nlargest" if sort_order == "descending" else "nsmallest"

            # create a query statement
            qry = f"{column} >= {column}.{order}({count}).min()"

            shp = shp.query(qry)  # type: ignore

        return shp

    def plot_context_shape(
        self, plt_ax: plt.Axes, shape_name: str, crs: Optional[str] = None
    ):
        """
        Plot a shape using the configuration in the config file.
        Dispatches to the correct plotting function according to the type.
        """

        # ignore the plotting if the shape is not available
        if shape_name not in self.shapes:
            return

        # get the desired shape
        shape = self.shapes[shape_name]
        shp = shape["gdf"]

        # adjust the shape to the correct crs
        if crs is not None:
            shp = shp.to_crs(crs)

        # grab the axis bounds
        xmin, xmax, ymin, ymax = plt_ax.axis()

        # get just the features that are in the view
        rows_in_view = shp.clip_by_rect(xmin, ymin, xmax, ymax)
        rows_in_view = shp[~rows_in_view.is_empty]

        # get the filter query
        rows_in_view = Mapper.filter_shape(rows_in_view, shape)

        # get the style from the shape
        style = shape["style"] if "style" in shape else {}

        rows_in_view.plot(ax=plt_ax, **style, legend=True)

        # annotate point names
        if "text_column" in shape:
            # get the color from the plot_args (if existent)
            color = style["color"] if "color" in style else "black"

            # Annotate the city names
            texts = []
            for _, row in rows_in_view.iterrows():
                texts.append(
                    plt_ax.text(
                        x=row.geometry.x,
                        y=row.geometry.y,
                        s=row[shape["text_column"]],  # type: ignore
                        color=color,
                    )
                )
            # adjust overlapping texts
            adjust_text(texts, ax=plt_ax, expand_axes=True, ensure_inside_axes=True)

        # # dispatch to the plot function according to the shape type
        # if shape["type"].lower() == "point":
        #     self.plot_point_shape(shape=shape, plt_ax=plt_ax)

        # elif shape["type"].lower() == "polygon":
        #     self.plot_point_shape(shape=shape, plt_ax=plt_ax)

        # elif shape["type"].lower() == "linestring":
        #     self.plot_line_shape(shape=shape, plt_ax=plt_ax)

    def plot_context_layers(
        self,
        plt_ax: plt.Axes,
        z_min: Optional[int] = None,
        z_max: Optional[int] = None,
        crs: Optional[str] = None,
    ):
        """
        Plot the layers in a specific z_order range. This is usefull for plotting negative and
        positive layers
        """

        for shape in self.shapes:
            z_order = self.shapes[shape]["z_order"]
            if z_min is not None and z_order < z_min:
                continue
            if z_max is not None and z_order > z_max:
                continue

            self.plot_context_shape(
                plt_ax=plt_ax,
                shape_name=shape,
                crs=crs,
            )

    @staticmethod
    def create_colorbar(
        raster: xr.DataArray,
        plt_ax: plt.Axes,
        label: str,
        labelsize: int = 12,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: Optional[Union[str, colors.Colormap]] = None,
        diverging: bool = False,
    ):
        """Add a colorbar to the given axes based on the values of the raster"""

        # First, set the minimum and maximum limits
        if not diverging:
            vmin = float(raster.min()) if vmin is None else vmin
            vmax = float(raster.max()) * 0.8 if vmax is None else vmax
        else:
            if vmax is None or vmin is None:
                vmax = max(abs(float(raster.max())), abs(float(raster.min())))
                vmin = -vmax

        # if no cmap is defined, use the INPE version
        cmap = INPE.cmap if cmap is None else cmap

        # Create a colorbar object with the desired range of values
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt_ax  # type: ignore[attr]
        )

        # Customize the colorbar
        cbar.set_label(label)
        cbar.ax.tick_params(labelsize=labelsize)

        return cbar

    @staticmethod
    def plot_raster_shape(
        raster: xr.DataArray,
        shp: gpd.GeoDataFrame,
        plt_ax: plt.Axes,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: Optional[Union[str, colors.Colormap]] = None,
        diverging: bool = False,
        colorbar_label: str = "Chuva acumulada (mm)",
        style_kwds: Optional[Dict] = None,
    ):
        """
        Given a raster and a shapefile (loaded in geopandas),
        plot the raster within the shape.
        If diverging is True, the scale will go from -max(abs) to +max(abs)
        """
        style_kwds = {} if style_kwds is None else style_kwds

        # plot the shape using our function that keeps the aspect of the axes
        shp.plot(ax=plt_ax, **style_kwds, legend=True)

        # to use contextily, we will write the raster to a MemoryFile
        # so we don't need to write it to disk and reload it
        # first we will clip the area and create a profile

        # let's take the bounding box and apply a buffer around it
        # the buffer will prevent the raster to be cut and include all pixels in the
        # displaying area

        # first we will get the bounding box from the shape and write it to
        # a shapely box object

        xmin, xmax, ymin, ymax = plt_ax.axis()
        bbox = box(xmin, ymin, xmax, ymax)

        # 0.1 of buffer because it is the size of each pixel in merge grid.
        nxmin, nymin, nxmax, nymax = bbox.buffer(0.1).bounds

        # create a subraster within the bounds
        subraster = raster.sel(
            longitude=slice(nxmin, nxmax), latitude=slice(nymin, nymax)
        )
        subraster = subraster.expand_dims(dim="band")

        profile = GISUtil.profile_from_xarray(subraster)

        # create a memory file and use it to create a memory dataset
        with rio.MemoryFile() as memfile:
            with memfile.open(**profile) as memdset:
                # write the data to the newly created dataset
                memdset.write(subraster)

            # now, let's create a colorbar for this
            cbar = Mapper.create_colorbar(
                raster=subraster,
                plt_ax=plt_ax,
                label=colorbar_label,
                labelsize=10,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                diverging=diverging,
            )

            # if no cmap is defined, use the INPE version
            cmap = INPE.cmap if cmap is None else cmap

            # with the dataset in memory, add the basemap
            cx.add_basemap(
                ax=plt_ax,
                source=memfile,
                vmin=cbar.vmin,
                vmax=cbar.vmax,
                reset_extent=False,
                cmap=cmap,
                alpha=1,
            )  # , vmin=0, vmax=100)

        # set the axis labels
        plt_ax.set_ylabel("Latitude (deg)")
        plt_ax.set_xlabel("Longitude (deg)")

        plt_ax.set_xlim((xmin, xmax))
        plt_ax.set_ylim((ymin, ymax))  # type: ignore
        # return plt_ax
