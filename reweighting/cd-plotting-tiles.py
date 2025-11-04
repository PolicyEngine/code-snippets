import geopandas as gpd
from shapely import affinity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# https://docs.google.com/spreadsheets/d/13XkF59JKzvw4SeSq5mbgIFrJfYjK4amg9JoQE5e9grQ/edit?gid=0#gid=0
# https://www.dailykos.com/stories/2019/7/23/1873464/-We-took-the-best-map-ever-of-US-congressional-districts-and-made-it-even-better
# https://www.primarycast.us/house-races/KY_2

# ----------------------------------------------------
# read the hex-cartogram shapefile you downloaded
# ----------------------------------------------------
shp_path = r"/mnt/c/devl/data/pe/HexCDv31/HexCDv31.shp"

gdf = gpd.read_file(shp_path)

# QUICK sanity-check: take a look at the attribute names
print(gdf.columns.tolist())
print(gdf.head(3))

# ----------------------------------------------------
nc = gdf[gdf["STATEAB"] == "NC"]

# ----------------------------------------------------
#  add your 14 numbers
# ----------------------------------------------------
my_numbers = np.random.rand(14)

vals = pd.DataFrame({
    "CDLABEL": range(1, 15),
    "metric":  my_numbers
})

nc["CDLABEL"] = nc["CDLABEL"].astype(int)

# ------------------------------------------------------------------
#  merge and plot
# ------------------------------------------------------------------
nc_plot = nc.merge(vals, on="CDLABEL", how="left")

fig, ax = plt.subplots(figsize=(6, 7))

# draw boundaries so the hexes have borders
nc_plot.boundary.plot(ax=ax, edgecolor="black", linewidth=0.4)

# fill by your metric
nc_plot.plot(column="metric",
             cmap="viridis",        # pick any Matplotlib colormap
             linewidth=0,
             legend=True,
             ax=ax)

ax.set_title("North Carolina congressional districts", fontsize=13)
ax.set_axis_off()
plt.tight_layout()
plt.show()

# Now rotate to make the state flat
# pick a single pivot
pivot = nc_plot.unary_union.centroid    # nc_plot is your merged GeoDataFrame

# rotate every geometry around *that* pivot
angle = -12
nc_rot = nc_plot.copy()
nc_rot["geometry"] = nc_rot.geometry.apply(
    lambda g: affinity.rotate(g, angle, origin=pivot)
)

# 3) plot
fig, ax = plt.subplots(figsize=(6, 7))
nc_rot.boundary.plot(ax=ax, edgecolor="black", linewidth=0.4)
nc_rot.plot(column="metric", cmap="viridis", linewidth=0, legend=True, ax=ax)

ax.set_title("One Household's Weight Values Across NC's 14 Districts", fontsize=13)
ax.set_axis_off()
plt.tight_layout()
plt.show()
  

