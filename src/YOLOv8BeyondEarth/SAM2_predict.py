import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from YOLOv8BeyondEarth.polygon import (shift_polygon, add_geometries, bboxes_to_shp, outlines_to_shp)
from YOLOv8BeyondEarth.predict import binary_mask_to_polygon_cv
from lsnms import nms

from sahi.slicing import slice_image
from tqdm import tqdm
from pathlib import Path

from rastertools_BOULDERING import raster, convert as raster_convert, metadata as raster_metadata
from shptools_BOULDERING import shp

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def refine_mask_with_obb(image_np, mask, predictor):
    """
    Re-prompt SAM2 using the oriented bounding box of an initial mask.
    Rotates the tile so the OBB is axis-aligned, re-prompts SAM2, then
    rotates the refined mask back. Skips if the mask is already within 5°
    of axis-alignment.
    Returns a bool (H, W) array.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(bool)

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    center, (w, h), angle = rect

    # Skip if already within 5° of axis-aligned (angle in [-90,0), so check both ends)
    if w == 0 or h == 0 or min(abs(angle), abs(angle + 90)) < 5:
        return mask.astype(bool)

    img_h, img_w = image_np.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image_np, M, (img_w, img_h))

    # After rotating by `angle`, w is horizontal and h is vertical — use directly
    cx, cy = center
    x1 = max(0.0, cx - w / 2)
    y1 = max(0.0, cy - h / 2)
    x2 = min(float(img_w), cx + w / 2)
    y2 = min(float(img_h), cy + h / 2)

    predictor.set_image(rotated_image)
    new_masks, _, _ = predictor.predict(
        box=np.array([[x1, y1, x2, y2]]),
        multimask_output=False,
    )

    M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
    refined = cv2.warpAffine(new_masks[0].astype(np.uint8), M_inv, (img_w, img_h))
    return refined > 0


def process_SAM2(slice_masks, slice_shift, slice_scores, slice_categories, slice_size, min_area_threshold, downscale_pred, detection_model):
    shift_x, shift_y = slice_shift
    scores, polygons, category_ids, category_names, is_within_slice_list = [], [], [], [], []

    # For each detection mask
    for idx, mask in enumerate(slice_masks):
        score = float(slice_scores[idx])
        category_id = int(slice_categories[idx])
        category_name = detection_model.category_mapping[str(category_id)]  
        
        # mask = np.squeeze(mask, axis=0)
        area = int(np.count_nonzero(mask))
        if area <= min_area_threshold:
            continue

        bool_mask_np = (np.squeeze(mask) > 0).astype(np.uint8)

        try:
            polygon = binary_mask_to_polygon_cv(bool_mask_np)
            if polygon is None or len(polygon) < 4:
                continue

            if downscale_pred:
                polygon_slice = polygon
            else:
                polygon_slice = np.stack([
                    (polygon[:, 0] / bool_mask_np.shape[0]) * slice_size,
                    (polygon[:, 1] / bool_mask_np.shape[0]) * slice_size
                ], axis=-1)

            min_edge_distance = 0.05 * slice_size
            max_edge_distance = 0.95 * slice_size
            is_within = (
                (polygon_slice[:, 0].min() >= min_edge_distance and polygon_slice[:, 0].max() <= max_edge_distance) and
                (polygon_slice[:, 1].min() >= min_edge_distance and polygon_slice[:, 1].max() <= max_edge_distance)
            )

            if not is_within:
                score = 0.10

            shifted_polygon = shift_polygon(polygon_slice, shift_x, shift_y)
            scores.append(score)
            polygons.append(shifted_polygon)
            category_ids.append(category_id)
            category_names.append(category_name)
            is_within_slice_list.append(is_within)

        except Exception:
            continue

    return pd.DataFrame({
        'score': scores,
        'polygon': polygons,
        'category_id': category_ids,
        'category_name': category_names,
        'is_within_slice': is_within_slice_list
    })


def get_sliced_prediction_SAM2(in_raster,
                              predictor,
                              detection_model=None,
                              confidence_threshold: float = 0.1,
                              output_dir=None,
                              interim_file_name=None,
                              interim_dir=None,
                              slice_size: int = None,
                              inference_size: int = None,
                              overlap_height_ratio: float = 0.2,
                              overlap_width_ratio: float = 0.2,
                              min_area_threshold: int = None,
                              downscale_pred: bool = False,
                              postprocess: bool = True,
                              postprocess_match_threshold: float = 0.5,
                              postprocess_class_agnostic: bool = False,
                              batch_size: int = 16,
                              half: bool = False,
                              refine_with_obb: bool = False):
    # Convert tiff (geospatial metadata) to PNG (expedted input to slice_image)
    in_raster = Path(in_raster)
    output_dir = Path(output_dir)
    out_png = in_raster.with_name(in_raster.stem + ".png")
    if not out_png.exists():
        raster_convert.tiff_to_png(in_raster, out_png)

    tmp_dir = (Path.home() / "tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    detection_model.image_size = inference_size
    detection_model.confidence_threshold = confidence_threshold

    # Slice (large) input image into overlapping tiles and track tile coordinates
    slice_image_result = slice_image(
        image=out_png.as_posix(),
        output_file_name=interim_file_name,
        output_dir=interim_dir,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        out_ext=".png",
    )
    num_slices = len(slice_image_result)
    shift_amounts = slice_image_result.starting_pixels
    slice_images = slice_image_result.images

    # Loop over all sliced images, in batches so GPU stays saturated
    frames = []
    for i in tqdm(range(0, num_slices, batch_size)):
        # Get current batch of images
        batch_images = slice_images[i:i + batch_size]
        batch_shifts = shift_amounts[i:i + batch_size]

        # Run detection model on batched images to get bounding boxes
        with torch.no_grad():
            prediction_results = detection_model.model(
                batch_images, imgsz=detection_model.image_size, verbose=False,
                device=detection_model.device, half=half
            )

        # This list stores one list per image slice. Each image slice's list contains
        # all bounding box predictions whose confidence is at least confidence_threshold. 
        bounding_boxes_per_slice = []
        scores_per_slice = []
        categories_per_slice = []
        for j, prediction_result in enumerate(prediction_results):
            slice_boxes = prediction_result.boxes.data

            # Filter out boxes not above the required confidence_threshold
            confidence_mask = slice_boxes[:, 4] >= confidence_threshold
            slice_boxes = slice_boxes[confidence_mask]

            # Append bounding boxes for given image slice to list
            if (len(slice_boxes) == 0):
                bounding_boxes_per_slice.append([])
                scores_per_slice.append([])
                categories_per_slice.append([])
            else:
                bounding_boxes_per_slice.append(slice_boxes[:, :4].cpu().numpy())
                scores_per_slice.append(slice_boxes[:, 4])
                categories_per_slice.append(slice_boxes[:, 5])

        # Run SAM2
        with torch.no_grad():
            non_empty_indices = [j for j, boxes in enumerate(bounding_boxes_per_slice) if len(boxes) > 0]
            existing_images = [batch_images[j] for j in non_empty_indices]
            existing_bounding_boxes = [bounding_boxes_per_slice[j] for j in non_empty_indices]
            
            if len(existing_images) > 0:
                predictor.set_image_batch(existing_images)
                masks_batch, _, _ =  predictor.predict_batch(
                    None,
                    None, 
                    box_batch=existing_bounding_boxes, 
                    multimask_output=False
                )
            else:
                masks_batch = []

        # Optional OBB refinement: re-prompt SAM2 with each mask's oriented bbox.
        # Breaks batch mode (one set_image call per mask), so use only for small runs.
        if refine_with_obb and len(masks_batch) > 0:
            for j, non_empty_idx in enumerate(non_empty_indices):
                img_np = np.array(batch_images[non_empty_idx])
                refined_masks = []
                for k in range(len(masks_batch[j])):
                    mask_2d = np.squeeze(masks_batch[j][k]) > 0
                    refined = refine_mask_with_obb(img_np, mask_2d.astype(np.uint8), predictor)
                    refined_masks.append(refined[np.newaxis].astype(masks_batch[j][k].dtype))
                masks_batch[j] = np.stack(refined_masks)

        # Process results of SAM2
        if len(masks_batch) > 0:
            for j, non_empty_idx in enumerate(non_empty_indices):
                slice_masks = masks_batch[j]
                slice_shift = batch_shifts[non_empty_idx] 
                slice_scores = scores_per_slice[non_empty_idx]
                slice_categories = categories_per_slice[non_empty_idx]

                df = process_SAM2(slice_masks, slice_shift, slice_scores, slice_categories, slice_size, min_area_threshold, downscale_pred, detection_model)

                if df.shape[0] > 0:
                    frames.append(df)

        if (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()

    if len(frames) == 0:
        df_all = pd.DataFrame(columns=['score', 'polygon', 'category_id', 'category_name', 'is_within_slice'])
    else:
        df_all = pd.concat(frames, ignore_index=True)

    df_all = df_all[df_all['polygon'].apply(lambda p: len(p) >= 4)].reset_index(drop=True)
    gdf = add_geometries(in_raster, df_all)

    in_res = raster_metadata.get_resolution(in_raster)[0]

    footprint_shp = tmp_dir / f"{in_raster.stem}-true-footprint.shp"
    footprint_line_shp = tmp_dir / f"{in_raster.stem}-true-footprint-as-a-line.shp"
    if not footprint_line_shp.exists():
        gdf_true_footprint = raster.true_footprint(in_raster, footprint_shp)
        in_meta = raster_metadata.get_profile(in_raster)
        gpd.GeoDataFrame(geometry=gdf_true_footprint.geometry.boundary.values, crs=in_meta["crs"].to_wkt()).to_file(
            footprint_line_shp)

    footprint_buffer_shp = tmp_dir / f"{in_raster.stem}-footprint-buffer-ss-{slice_size}.shp"
    if not footprint_buffer_shp.exists():
        gdf_line_buffer = shp.buffer(footprint_line_shp, slice_size * 0.10 * in_res, footprint_buffer_shp)
    else:
        gdf_line_buffer = gpd.read_file(footprint_buffer_shp)

    gdf_boulders = gdf.copy()
    gdf_boulders["id"] = gdf_boulders.index
    gdf_intersected = gpd.overlay(gdf_boulders, gdf_line_buffer, how="intersection", keep_geom_type=True)

    gdf["is_at_edge"] = False
    gdf.loc[gdf_intersected.id.values, "is_at_edge"] = True

    gdf = gdf.drop_duplicates(subset="geometry", ignore_index=True)
    gdf["id"] = gdf.index

    bbox_filename = in_raster.stem + "-predictions-ct-" + str(int(confidence_threshold * 100)).zfill(3) + "-ss-" + str(
        slice_size) + "-is-" + str(inference_size) + "-ov-" + str(int(overlap_height_ratio * 100)).zfill(3) + "-bbox.shp"
    mask_filename = bbox_filename.replace("-bbox.shp", "-mask.shp")

    if downscale_pred:
        bbox_filename = bbox_filename.replace("-bbox.shp", "-downscaled-bbox.shp")
        mask_filename = mask_filename.replace("-mask.shp", "-downscaled-mask.shp")

    out_bbox_shp = output_dir / bbox_filename
    out_mask_shp = output_dir / mask_filename
    bboxes_to_shp(gdf, out_bbox_shp)
    outlines_to_shp(gdf, out_mask_shp)

    if postprocess:
        if postprocess_class_agnostic:
            keep = nms(boxes=np.stack(gdf.bbox.values), scores=gdf.score.values,
                       iou_threshold=postprocess_match_threshold, class_ids=None, rtree_leaf_size=32)
        else:
            keep = nms(boxes=np.stack(gdf.bbox.values), scores=gdf.score.values,
                       iou_threshold=postprocess_match_threshold, class_ids=gdf.category_id.values, rtree_leaf_size=32)

        gdf_nms = gdf.loc[keep]
        bboxes_to_shp(gdf_nms, out_bbox_shp.with_name(out_bbox_shp.stem + "-nms.shp"))
        outlines_to_shp(gdf_nms, out_mask_shp.with_name(out_mask_shp.stem + "-nms.shp"))
        return gdf, gdf_nms
    else:
        return gdf, None


def get_sliced_prediction_SAM2_auto(
    in_raster,
    mask_generator,
    output_dir=None,
    interim_file_name=None,
    interim_dir=None,
    slice_size: int = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_threshold: int = None,
    postprocess: bool = True,
    postprocess_match_threshold: float = 0.5,
):
    in_raster  = Path(in_raster)
    output_dir = Path(output_dir)
    out_png = in_raster.with_name(in_raster.stem + ".png")
    if not out_png.exists():
        raster_convert.tiff_to_png(in_raster, out_png)

    tmp_dir = Path.home() / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    slice_image_result = slice_image(
        image=out_png.as_posix(),
        output_file_name=interim_file_name,
        output_dir=interim_dir,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        out_ext=".png",
    )
    num_slices    = len(slice_image_result)
    shift_amounts = slice_image_result.starting_pixels
    slice_images  = slice_image_result.images

    frames = []
    for i in tqdm(range(num_slices)):
        shift_x, shift_y = shift_amounts[i]

        with torch.no_grad():
            masks_data = mask_generator.generate(slice_images[i])

        scores, polygons, is_within_list = [], [], []
        for m in masks_data:
            bool_mask = (m["segmentation"]).astype(np.uint8)
            if int(bool_mask.sum()) <= min_area_threshold:
                continue

            try:
                polygon = binary_mask_to_polygon_cv(bool_mask)
                if polygon is None or len(polygon) < 4:
                    continue

                min_edge = 0.05 * slice_size
                max_edge = 0.95 * slice_size
                is_within = (
                    polygon[:, 0].min() >= min_edge and polygon[:, 0].max() <= max_edge and
                    polygon[:, 1].min() >= min_edge and polygon[:, 1].max() <= max_edge
                )

                score = float(m["predicted_iou"]) if is_within else 0.10
                shifted = shift_polygon(polygon, shift_x, shift_y)
                scores.append(score)
                polygons.append(shifted)
                is_within_list.append(is_within)

            except Exception:
                continue

        if scores:
            frames.append(pd.DataFrame({
                "score":           scores,
                "polygon":         polygons,
                "category_id":     [0] * len(scores),
                "category_name":   ["boulder"] * len(scores),
                "is_within_slice": is_within_list,
            }))

        if i % 50 == 0:
            torch.cuda.empty_cache()

    if len(frames) == 0:
        df_all = pd.DataFrame(columns=["score", "polygon", "category_id", "category_name", "is_within_slice"])
    else:
        df_all = pd.concat(frames, ignore_index=True)

    df_all = df_all[df_all["polygon"].apply(lambda p: len(p) >= 4)].reset_index(drop=True)
    gdf = add_geometries(in_raster, df_all)

    in_res = raster_metadata.get_resolution(in_raster)[0]

    footprint_shp      = tmp_dir / f"{in_raster.stem}-true-footprint.shp"
    footprint_line_shp = tmp_dir / f"{in_raster.stem}-true-footprint-as-a-line.shp"
    if not footprint_line_shp.exists():
        gdf_true_footprint = raster.true_footprint(in_raster, footprint_shp)
        in_meta = raster_metadata.get_profile(in_raster)
        gpd.GeoDataFrame(geometry=gdf_true_footprint.geometry.boundary.values,
                         crs=in_meta["crs"].to_wkt()).to_file(footprint_line_shp)

    footprint_buffer_shp = tmp_dir / f"{in_raster.stem}-footprint-buffer-ss-{slice_size}.shp"
    if not footprint_buffer_shp.exists():
        gdf_line_buffer = shp.buffer(footprint_line_shp, slice_size * 0.10 * in_res, footprint_buffer_shp)
    else:
        gdf_line_buffer = gpd.read_file(footprint_buffer_shp)

    gdf_boulders = gdf.copy()
    gdf_boulders["id"] = gdf_boulders.index
    gdf_intersected = gpd.overlay(gdf_boulders, gdf_line_buffer, how="intersection", keep_geom_type=True)

    gdf["is_at_edge"] = False
    gdf.loc[gdf_intersected.id.values, "is_at_edge"] = True

    gdf = gdf.loc[np.logical_or(gdf.is_at_edge == True, gdf.is_within_slice == True)]
    gdf = gdf.drop_duplicates(subset="geometry", ignore_index=True)
    gdf["id"] = gdf.index

    ov = str(int(overlap_height_ratio * 100)).zfill(3)
    bbox_filename = f"{in_raster.stem}-sam2auto-ss-{slice_size}-ov-{ov}-bbox.shp"
    mask_filename = bbox_filename.replace("-bbox.shp", "-mask.shp")

    out_bbox_shp = output_dir / bbox_filename
    out_mask_shp = output_dir / mask_filename
    bboxes_to_shp(gdf, out_bbox_shp)
    outlines_to_shp(gdf, out_mask_shp)

    if postprocess:
        keep = nms(
            boxes=np.stack(gdf.bbox.values),
            scores=gdf.score.values,
            iou_threshold=postprocess_match_threshold,
            class_ids=None,
            rtree_leaf_size=32,
        )
        gdf_nms = gdf.loc[keep]
        bboxes_to_shp(gdf_nms, out_bbox_shp.with_name(out_bbox_shp.stem + "-nms.shp"))
        outlines_to_shp(gdf_nms, out_mask_shp.with_name(out_mask_shp.stem + "-nms.shp"))
        return gdf, gdf_nms
    else:
        return gdf, None