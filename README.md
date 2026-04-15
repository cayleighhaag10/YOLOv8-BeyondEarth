# YOLOv8-BeyondEarth
YOLOv8-BeyondEarth is a repository that contains tools and scripts to create YOLOv8 custom datasets, train the model, and post-process the obtained results. There is a strong focus on the use of satellite imagery and the application of the YOLOv8 model on solid planetary bodies in our Solar system.


## Installation

Create a new environment if wanted. Then you can install the rastertools by writing the following in your terminal. Before installing YOLOv8-BeyondEarth, you need to install `rastertools` and `shptools.


Use the following Python version.
```bash
conda install python=3.10.15
```   

#### rastertools

```bash
git clone https://github.com/astroNils/rastertools.git
cd rastertools
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps rastertools_BOULDERING
pip install -r requirements.txt
```

#### shptools

```bash
git clone https://github.com/astroNils/shptools.git
cd shptools
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps shptools_BOULDERING
pip install -r requirements.txt
```

#### YOLOv8-BeyondEarth

````bash
git clone https://github.com/astroNils/YOLOv8-BeyondEarth
cd YOLOv8-BeyondEarth
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps YOLOv8BeyondEarth
pip install -r requirements.txt
````

#### sahi

To get the last version of sahi.

```bash
git clone https://github.com/obss/sahi.git
cd sahi
pip install -e . 
```

You should now have access to this module in Python.

```bash
python
```

```python
from YOLOv8BeyondEarth.predict import get_sliced_predictionfast
```

## Running on Sherlock (Stanford HPC)

Inference is GPU-bound, so getting the right GPU allocation is critical for performance. On a shared GPU slice (~6GB), a full NAC image with 4 slice sizes takes ~35 minutes. On a full NVIDIA A30 (24GB), the same run takes ~7.5 minutes.

### Requesting a full A30 on Sherlock OnDemand

When launching a JupyterLab session via [Sherlock OnDemand](https://ondemand.sherlock.stanford.edu):

| Field | Value |
|---|---|
| Partition | `gpu` |
| #GPUs | `1` |
| Memory (GB) | `32` |
| Additional node features | `GPU_MEM:24GB` |

The `GPU_MEM:24GB` constraint ensures you get a full A30 rather than a shared slice.

### Recommended inference settings

```python
slice_sizes = [256, 512, 768, 1024]
confidence_threshold = 0.10
inference_size = 1024
overlap_height_ratio = 0.20

for slice_size in slice_sizes:
    __, __ = get_sliced_predictionfast(in_raster,
                        detection_model=detection_model,
                        confidence_threshold=confidence_threshold,
                        has_mask=True,
                        output_dir=output_dir,
                        slice_size=slice_size,
                        inference_size=inference_size,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                        min_area_threshold=6,
                        downscale_pred=True,
                        postprocess=True,
                        postprocess_match_threshold=0.2,
                        postprocess_class_agnostic=False,
                        batch_size=16)
```

## Bugs

If in your coordinate system, the latitude of origin is different than 0 degrees and this is expressed by the parameter `["Latitude of natural origin"`, there is a bug in the alignment of the prediction shapefiles (i.e., the shapefiles do not align with the raster/image). This bug does not happen for the Lunar Reconnaissance Orbiter NAC images. I wonder if this is due to the previously mentioned coordinate system/projection parameter and some bugs with rasterio/shapely? (need to do more digging there). When this parameter is not defined (i.e., a latitude of origin equal to 0), **then there is no bug**. Note that in NAC images, an equirectangular projection can be centered on a latitude without causing a bug. Looking at the coordinate system, the centering on the latitude seems to be described by another parameter (`["Latitude of 1st standard parallel"`). Therefore,  one solution could be to convert the coordinate system so that the centered latitude is described by the 1st standard parallel (or just keep the center latitude at 0 degrees). You can switch between coordinate systems with the help of `gdalwarp` (https://gdal.org/programs/gdalwarp.html).

```bash
gdalwarp -s_src <old_coordinate_system> -t_srs <wkt_string_where_only_latitude_1st_standard_parallel_is_used> <src_raster> <dst_raster>
```

You can get information about the old coordinate system with the help of `gdal_info`. 

```bash
gdalinfo <src_raster>
```

## Getting Started

A jupyter notebook is provided as a tutorial ([GET STARTED HERE](./resources/nb/GETTING_STARTED.ipynb)).

### Few links (YOLO, YOLOv8, anchor-free object detection)

https://github.com/ultralytics/ultralytics

https://medium.com/cord-tech/yolov8-for-object-detection-explained-practical-example-23920f77f66a

https://keylabs.ai/blog/comparing-yolov8-and-yolov7-whats-new/

https://learnopencv.com/fcos-anchor-free-object-detection-explained/

https://learnopencv.com/centernet-anchor-free-object-detection-explained/#What-is-Anchor-Free-Object-Detection

https://www.datacamp.com/blog/yolo-object-detection-explained
