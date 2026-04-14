import cv2
import numpy as np
import rasterio as rio
import geopandas as gpd

from shapely.geometry import (box, Polygon)


def is_within_slice(polygon, slice_height, slice_width):
    """
    Returns True if the polygon is fully within the slice (not touching any edge), else False.
    """
    at_edge12 = np.any(np.any(polygon == -0.5, axis=0) == True)
    at_edge3 = np.any(polygon[:, 0] == slice_width - 0.5, axis=0)
    at_edge4 = np.any(polygon[:, 1] == slice_height - 0.5, axis=0)
    is_intersecting_edge = np.any(np.array([at_edge12, at_edge3, at_edge4]) == True)
    return (False if is_intersecting_edge else True)


def shift_polygon(polygon, shift_x, shift_y):
    """Translate polygon coordinates from slice-local to full-image coordinates."""
    return (np.stack([polygon[:, 0] + shift_x, polygon[:, 1] + shift_y], axis=-1))


def row_bbox(row):
    return(list(row.geometry.bounds))


def row_bbox_to_shapely(row):
    return(box(*row.bbox))


def add_geometries(in_raster, df):
    """Convert pixel-space polygon coordinates to georeferenced coordinates using the raster transform."""
    with rio.open(in_raster) as src:
        in_crs = src.meta["crs"]
        boulder_geometry = []
        for polygon in df.polygon.values:
            xs, ys = rio.transform.xy(src.transform, polygon[:, 1], polygon[:, 0])
            boulder_geometry.append(Polygon(np.stack([xs, ys], axis=-1)))
        gdf = gpd.GeoDataFrame(df, geometry=boulder_geometry, crs=in_crs.to_wkt())
        gdf["bbox"] = gdf.apply(row_bbox, axis=1)
    return gdf


def bboxes_to_shp(gdf, out_shp):
    """Save bounding boxes to shapefile."""
    gdf_copy = gdf.rename(columns={"category_id": "cat_id", "category_name": "cat_name", "is_within_slice": "isin_slice"})
    gdf_copy["geometry"] = gdf_copy.apply(row_bbox_to_shapely, axis=1)
    gdf_copy.drop(columns=['bbox', 'polygon']).to_file(out_shp)


def outlines_to_shp(gdf, out_shp):
    """Save polygon outlines to shapefile."""
    gdf_copy = gdf.rename(columns={"category_id": "cat_id", "category_name": "cat_name", "is_within_slice": "isin_slice"})
    gdf_copy.drop(columns=['bbox', 'polygon']).to_file(out_shp)


def binary_mask_to_polygon(binary_mask):
    """
    Convert a binary mask (2D numpy array) to a polygon outline (array of x,y coordinates).
    Uses OpenCV findContours, which is significantly faster than skimage find_contours.
    Returns the largest contour found as an (N, 2) array of (x, y) coordinates.
    """
    mask_uint8 = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in mask")
    contour = max(contours, key=cv2.contourArea)
    return contour.squeeze(1)  # (N, 1, 2) -> (N, 2)
