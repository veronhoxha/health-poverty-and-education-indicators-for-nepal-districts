import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch 
import folium
import seaborn as sns
from scipy.stats import zscore, pearsonr
import plotly.express as px
from branca.colormap import linear
import json
import mapclassify

# ----------------------------------------------------------------

def choropleth_mapping(gdf, variables, title, legend_label=None, crs_epsg=3857):
    
    gdf_proj = gdf.to_crs(epsg=crs_epsg)
    
    fig, ax = plt.subplots(figsize=(10,10))

    # in case i want an univariate choropleth mapp
    if isinstance(variables, str):
        var = variables
        gdf_proj.plot(column=var, cmap="Reds", linewidth=0.5, edgecolor="grey",
            legend=True, legend_kwds={
                "label": legend_label or var,
                "orientation": "horizontal",
                "shrink": 0.6, "pad": 0.02
            },ax=ax)

    # bivariate choropleth map
    else:
        var1, var2 = variables
        k = 3

        q1 = mapclassify.Quantiles(gdf_proj[var1], k=k)
        q2 = mapclassify.Quantiles(gdf_proj[var2], k=k)
        gdf_proj[f"{var1}_q"] = q1.yb
        gdf_proj[f"{var2}_q"] = q2.yb
        gdf_proj["bivar"] = gdf_proj[f"{var2}_q"]*k + gdf_proj[f"{var1}_q"]

        # palette and bivariate map inspired from https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/
        palette = ["#e8e8e8","#ace4e4","#5ac8c8",
            "#dfb0d6","#a5add3","#5698b9",
            "#be64ac","#8c62aa","#3b4994"]
        
        cmap = ListedColormap(palette)

        gdf_proj.plot(column="bivar", cmap=cmap, linewidth=0.5, edgecolor="grey",legend=False, ax=ax)

        # matrix legend
        x0, y0, size = 0.02, 0.05, 0.06
        for i in range(k):
            for j in range(k):
                idx = i*k + j
                rect = Rectangle(
                    (x0 + j*size, y0 + i*size),
                    size, size,
                    facecolor=palette[idx],
                    edgecolor="black",
                    transform=ax.transAxes
                )
                ax.add_patch(rect)


        ax.annotate("", xy=(x0+3*size, y0-0.01), xytext=(x0, y0-0.01),
                    xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.2)
        )
        ax.text(x0, y0-0.03, "Low", transform=ax.transAxes, ha="left", va="center")
        ax.text(x0+3*size, y0-0.03, "High", transform=ax.transAxes, ha="right", va="center")

        ax.text(x0+1.5*size, y0-0.04, var1, transform=ax.transAxes, ha="center", va="top", fontweight="bold")

        ax.annotate("", xy=(x0-0.01, y0+3*size), xytext=(x0-0.01, y0), xycoords="axes fraction", 
                    arrowprops=dict(arrowstyle="->", lw=1.2))
        ax.text(x0-0.02, y0, "Low", transform=ax.transAxes, ha="center", va="bottom", rotation=90)
        ax.text(x0-0.02, y0+3*size, "High", transform=ax.transAxes, ha="center", va="top", rotation=90)
        ax.text(x0-0.045, y0+1.5*size, var2, transform=ax.transAxes, ha="right", va="center", rotation=90, fontweight="bold")

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    
    
def pearson_report(gdf, x, y):
    series = gdf[[x, y]].dropna()
    r, p = pearsonr(series[x], series[y])
    return r, p

    
def extreme_value_variables(gdf, variable, n=5):
    df = gdf[["district", variable]].dropna(subset=[variable])
    
    bottom_df = df.nsmallest(n, variable)
    top_df = df.nlargest(n, variable)
    
    print(f"Lowest {n} districts by {variable}:")
    for idx, row in bottom_df.iterrows():
        print(f"{row["district"]}: {row[variable]:.2f}")
    
    print(f"\nHighest {n} districts by {variable}:")
    for idx, row in top_df.iterrows():
        print(f"{row["district"]}: {row[variable]:.2f}")
    
    return bottom_df, top_df

