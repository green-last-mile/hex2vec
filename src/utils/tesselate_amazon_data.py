import asyncio
from functools import partial, wraps
from pathlib import Path
from random import random
from typing import Any, Dict, Generator, List, Set
import warnings

import backoff
from tqdm import tqdm
import alphashape
import geopandas as gpd
import h3
import osmnx as ox
import pandas as pd
from shapely.geometry import mapping


from ..data.download import ensure_geometry_type
from ..data.load_data import load_city_tag, load_city_tag_h3
from ..data.make_dataset import group_df_by_tag_values, h3_to_polygon, prepare_city_path
from ..data.utils import TOP_LEVEL_OSM_TAGS
from ..settings import DATA_PROCESSED_DIR, DATA_RAW_DIR



# little of a hack to get around the fact that osmnx doesn't have async
def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return run


# cleaner way to get series of points into a list
def geometry_series_to_xy(geometry_series, epgs=32633):
    g = geometry_series.to_crs(epsg=epgs).copy()
    return list(zip(g.x, g.y))


def iterate_hex_dir(parent_dir: Path) -> Generator[Path, None, None]:
    for hex_id in parent_dir.iterdir():
        if h3.h3_is_valid(hex_id.stem) and hex_id.is_dir():
            yield hex_id


def cover_point_array_w_hex(
    point_array: pd.Series,
    resolution: int,
    epgs: int = 32633,
) -> Set[str]:

    xy = geometry_series_to_xy(point_array, epgs=epgs)
    print("computing alpha shape, this may take a while...")
    res = alphashape.alphashape(xy, alpha=0.001)
    res = res.buffer(2 * h3.edge_length(resolution=resolution, unit="m"))
    convex_hull_df = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(res),
        crs=f"EPSG:{epgs}",
    )
    convex_hull_df = convex_hull_df.to_crs(epsg=4326)
    feature = mapping(convex_hull_df)

    # reverse coordinates in geojson
    for feature in feature["features"]:
        geom = feature["geometry"]
        geom["coordinates"] = [[j[::-1] for j in i] for i in geom["coordinates"]]
        hexes = list(h3.polyfill(geom, resolution))
        break

    # convert the hex ids to polygons
    return gpd.GeoDataFrame(
        data={
            "h3": hexes,
        },
        geometry=gpd.GeoSeries(map(h3_to_polygon, hexes)),
        crs="EPSG:4326",
    )

@backoff.on_exception(backoff.expo,
                      Exception,
                      max_tries=8,
                      max_time=300)
def ox_geometries(
    *args,
    **kwargs,
) -> gpd.GeoDataFrame:
    return ox.geometries_from_polygon(*args, **kwargs)

async_ox_geometries = async_wrap(ox_geometries)

def pull_hex_tags_synch(
    row: pd.Series,
    city_dir: Path,
    tag_list: str,
    simplify_data: bool = True,
    force_pull: bool = False
) -> pd.Series:

    # make the directory
    hex_dir = city_dir.joinpath(row["h3"])
    hex_dir.mkdir(parents=True, exist_ok=True)
    # print("running for hex", row["h3"])
    for tag in tag_list:
        if not hex_dir.joinpath(f"{tag}.pkl").exists() or force_pull:
            gdf = ox_geometries(row['geometry'], tags={tag: True})
            # clean the data
            if not gdf.empty:
                gdf = ensure_geometry_type(gdf)                
                gdf = gdf.reset_index()[["osmid", tag, "geometry"]] if simplify_data else gdf.reset_index()
                # save the gdf
                gdf.to_pickle(
                    hex_dir.joinpath(f"{tag}.pkl").absolute(),
                )
        else:
            print(f"{tag} already exists")



async def walk_n_queue(
    queue: asyncio.Queue,
    city_dir: Path,
    hex_gdf: gpd.GeoDataFrame,
    tag_list: str,
    resolution: int,
    simplify_data: bool = True,
    force_pull: bool = False
) -> None:

    r_dir = city_dir.joinpath(f"resolution_{resolution}")
    r_dir.mkdir(parents=True, exist_ok=True)
    for _, row in hex_gdf.iterrows():
        hex_dir = r_dir.joinpath(row["h3"])
        hex_dir.mkdir(parents=True, exist_ok=True)
        for tag in tag_list:
            if not (hex_dir.joinpath(f"{tag}.pkl").exists() 
                    or hex_dir.joinpath(f"{tag}_is_empty.txt").exists()) \
                    or force_pull:
                await queue.put(
                    (hex_dir.joinpath(f"{tag}.pkl"), row.geometry, tag)
                )
                await asyncio.sleep(0)


# async def pull_tags_for_hex(
#     row: pd.Series,
#     city_dir: Path,
#     tag_list: str,
#     simplify_data: bool = True,
#     force_pull: bool = False
# ) -> pd.Series:

async def pull_tags_for_hex(
    queue: asyncio.Queue,
    semaphore: asyncio.Semaphore
):
    simplify_data = True
    async with semaphore:
        while True:
            # this is probs bad practice
            save_path, geom, tag = await queue.get()
            print(f"pulling {save_path.parent.parent.parent.stem}/{save_path.parent.stem}/{tag}")
            gdf = await async_ox_geometries(geom, tags={tag: True})
            # clean the data
            if not gdf.empty:
                gdf = ensure_geometry_type(gdf)
                gdf = gdf.reset_index()[["osmid", tag, "geometry"]] if simplify_data else gdf.reset_index()
                # save the gdf
                gdf.to_pickle(
                    save_path.absolute()
                )
                print(f"finished {save_path.parent.parent.parent.stem}/{save_path.parent.stem}/{tag}")
            else:
                # record that the df is empty so we don't try again
                with open(save_path.parent.joinpath(f"{tag}_is_empty.txt").absolute(), 'w') as f:
                    pass
            # tell everyon the that the task is done
            queue.task_done()
        

async def pull_tags_for_hex_gdf(
    city_dir: Path,
    hex_gdf: gpd.GeoDataFrame,
    tag_list: str,
    resolution: int,
    simplify_data: bool = True,
    force_pull: bool = False
) -> None:

    # make the directory
    r_dir = city_dir.joinpath(f"resolution_{resolution}")
    r_dir.mkdir(parents=True, exist_ok=True)
    for row in hex_gdf.iterrows():
        await pull_tags_for_hex(row[1], r_dir, tag_list, simplify_data, force_pull)
        # sleep to defer to other processes
        await asyncio.sleep(0.01)
    
def join_hex_dfs(
    hex_parent_dir: Path,
    tag_list: List[str],
    target_resolution: int,
    output_dir: Path,
) -> gpd.GeoDataFrame:

    # create a map of smaller hexes to larger hexes
    for hex_id in tqdm(list(iterate_hex_dir(hex_parent_dir))):
        # create a hexagon gpd
        target_hex_ids = list(h3.h3_to_children(hex_id.stem, target_resolution))
        all_hex_polygons = list(map(h3_to_polygon, target_hex_ids))
        hexes_gdf = gpd.GeoDataFrame(
            pd.DataFrame({"h3": target_hex_ids, "geometry": all_hex_polygons}),
            crs="EPSG:4326",
        )

        # create a list of other parent-level hexagons needed. 
        # This is because the children are not always geographically contained. 
        needed_hexes = [hex_id]
        needed_hexes.extend([hex_parent_dir / h for h in h3.k_ring(hex_id.stem, 1)])
        
        # create a list of the buffered boundary hexagons. It's not necessary to have the neighbors of these hexagons.
        boundary = False
        if (hex_parent_dir.parent / "boundary.hex.txt").exists():
            with open((hex_parent_dir.parent / "boundary.hex.txt"), 'r') as f:
                boundary = hex_id.stem in f.read().split("\n")

        # check that all hexagons exist (they won't, because at some point we are at the edge of a hexagon)
        drops = []
        for i, p_hex in enumerate(needed_hexes):
            if not p_hex.exists() and not boundary:
                print(f"{p_hex.stem} is needed to completely cover {hex_id.stem} children but missing")
                drops.append(i)
        
        # drop the missing parent hexagons
        for i in drops[::-1]:
            needed_hexes.pop(i)
        
        # load the tag data. 
        for tag in tag_list:
            tag_dfs = []
            for p_hex in needed_hexes:
                tag_gdf = load_city_tag(p_hex, tag=tag, data_dir=hex_id)
                if tag_gdf is not None:
                    tag_dfs.append(tag_gdf)
            if len(tag_dfs):

                # create a hex + neighbors super df
                tag_gdf = pd.concat(tag_dfs, axis=0)
                # drop duplicated osmids
                tag_gdf = tag_gdf.drop_duplicates(subset=['osmid'])

                # spatial join of the hexes with the tag data. Only save data that is in the target hexagons.
                tag_gdf = gpd.sjoin(
                    tag_gdf, hexes_gdf, how="inner", predicate="intersects"
                )[["h3", "osmid", tag, "geometry"]]
                
                # create a save location for the data 
                save_location = output_dir.joinpath(
                    hex_id.stem,
                )
                save_location.mkdir(parents=True, exist_ok=True)
                tag_gdf.to_pickle(
                    save_location.joinpath(f"{tag}_{target_resolution}.pkl").absolute(),
                    protocol=4,
                )


def group_h3_tags(
    hex_id: str,
    resolution: int,
    tags=TOP_LEVEL_OSM_TAGS,
    filter_values: Dict[str, str] = None,
    fill_missing=True,
    data_dir: Path = DATA_RAW_DIR,
    save_dir: Path = DATA_PROCESSED_DIR,
    save=True,
) -> pd.DataFrame:
    dfs = []
    unique_h3 = list(h3.h3_to_children(hex_id, resolution))
    for tag in tags:
        # create df
        df = load_city_tag_h3(hex_id, tag, resolution, filter_values, data_path=data_dir)
        if df is not None and not df.empty:
            tag_grouped = group_df_by_tag_values(df, tag)
        else:
            tag_grouped = pd.DataFrame(data={'h3': unique_h3})
        
        # fill missing values, using pandas concat
        if fill_missing and filter_values is not None:
            columns_names = [f"{tag}_{value}" for value in filter_values[tag] if f"{tag}_{value}" not in tag_grouped.columns]
            tag_grouped = pd.concat([tag_grouped, pd.DataFrame(columns=columns_names)], axis=1, verify_integrity=True)
        dfs.append(tag_grouped)

    results = pd.concat(dfs, axis=0, )


    results = results.fillna(0).groupby("h3").sum()
    return results.reindex(index=unique_h3, fill_value=0).astype("Int16").reset_index()


def group_hex_tags(
    hex_parent_dir: Path,
    tag_list: List[str],
    output_dir: Path,
    resolution: int,
    filter_values: Dict[str, str] = None,
) -> None:

    for hex_id in tqdm(list(iterate_hex_dir(hex_parent_dir))):
        h3_grouped_df = group_h3_tags(
            hex_id=hex_id.stem,
            resolution=resolution,
            tags=tag_list,
            filter_values=filter_values,
            data_dir=hex_id.parent,
            save=False,
        )
        save_location = output_dir.joinpath(
            hex_id.stem,
        )
        save_location.mkdir(parents=True, exist_ok=True)
        h3_grouped_df.to_pickle(
            save_location.joinpath(f"{resolution}.pkl").absolute(),
            protocol=4,
        )


def create_city_from_hex(
    hex_parent_dir: Path, output_dir: Path, resolution: int, drop_all_zero=True
) -> None:
    dfs = [
        pd.read_pickle(hex_id.joinpath(f"{resolution}.pkl"))
        for hex_id in iterate_hex_dir(hex_parent_dir)
    ]

    df = pd.concat(dfs, ignore_index=True).set_index("h3")
    df.fillna(0, inplace=True)
    if drop_all_zero:
        df = df[(df.T != 0).any()]
    df.to_pickle(output_dir.joinpath(f"{resolution}.pkl").absolute(), protocol=4)
    return df


def get_buffer_hexes(hexes: Set[str], save_boundary: bool = True, save_path: Path = None) -> Set[str]:
    reported_hex = set()
    for hex in hexes:
        # find the missing and add them
        if missing := h3.k_ring(hex, 1) - hexes:
            for _h in missing:
                reported_hex.add(
                    _h
                )
    
    if save_boundary:
        if (save_path / "boundary.hex.txt").exists():
            with open(save_path / "boundary.hex.txt", 'r') as f:
                already_marked = set((h for h in f.read().split("\n") if len(h)))
        else:
            already_marked = set()

        with open(save_path / "boundary.hex.txt", 'w') as f:                        
            f.write("\n".join(reported_hex.union(already_marked)))
    
    return reported_hex

def fetch_city_polygon(city_name: str) -> Dict:
    from osmnx.geocoder import _geocode_query_to_gdf

    try:
        # bypass osmnx extras
        city_gdf = _geocode_query_to_gdf(
            city_name,
            by_osmid=False,
            which_result=None
        )
        # find the boundary polygon. Turn into geojson
        city_boundary_geojson = mapping(city_gdf.geometry.iloc[0])
        # reverse the coordinates for the h3 api
        city_boundary_geojson['coordinates'] = [[c[::-1] for c in coords] for coords in city_boundary_geojson['coordinates']]
        # return the json
        return city_boundary_geojson
    
    except (IndexError, KeyError):

        raise Exception(f"OSMNX did not find a boundary geometry for {city_name}")



