# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from ... import logger
from ...osm.tiling import TileManager
from ...osm.viz import GeoPlotter, ImagePlotter
from ...utils.geo import BoundaryBox, Projection
from ...utils.io import DATA_URL, download_file #maploc/utils/io.py
from .dataset import BimDataModule
from .utils import parse_gps_file

split_files = ["test1_files.txt", "test2_files.txt", "train_files.txt"]

def prepare_bim(csv_path, image_path, output_path):
    all_xy = pd.read_csv(csv_path).dropna(subset=['x', 'y'])[['x', 'y']].to_numpy()
    if all_xy.size == 0:
        raise ValueError(f"No valid data found in {csv_path}.")

    # plotter = ImagePlotter(image_path, x_min=-9.15, x_max=24.59, y_min=-16.85, y_max=16.88)
    plotter = ImagePlotter()
    plotter.points(all_xy, color='red', label='ViewPoints')
    plotter.bbox([-9, -16, 24, 16], color='blue', label='tiling bounding box')
    plotter.fig.write_html(image_path, output_path)  # Adjust path as needed

    return plotter

def prepare_osm(
    data_dir,
    csv_path,
    #osm_path,
    output_path,
    tile_margin=512,
    ppm=2,
):
    all_xy = pd.read_csv(csv_path).dropna(subset=['x', 'y'])[['x', 'y']].to_numpy() # Read X and Y coordinates from a CSV file & Filter out rows if either 'x' or 'y' might be empty
    if all_xy.size == 0:
        raise ValueError(f"Cannot find any valid GPS data in {csv_path}.")
    projection = all_xy
    bbox_map = BoundaryBox(all_xy.min(0), all_xy.max(0)) + tile_margin

    plotter = ImagePlotter()
    plotter.points(all_xy, "red", name="GPS")
    plotter.bbox(projection.unproject(bbox_map), color='blue', label='tiling bounding box')
    plotter.fig.write_html(data_dir / "split_bim.html")

    tile_manager = TileManager.from_bbbox( # original function is from_bbox
        projection,
        bbox_map,
        ppm,
        data_dir / 'tiles.pkl',
        tile_size=128
        # path=osm_path,
    )

    png_files = {
        (1, 1): {
            "Column_output": data_dir / "corplab/bev/Column_output.png",
            "Door_output": data_dir / "corplab/bev/Door_output.png",
            "FlowFitting_output": data_dir / "corplab/bev/FlowFitting_output.png",
            "FlowSegment_output": data_dir / "corplab/bev/FlowSegment_output.png",
            "Slab_output": data_dir / "corplab/bev/Slab_output.png",
            "Wall_output": data_dir / "corplab/bev/Wall_output.png",
            "Window_output": data_dir / "corplab/bev/Window_output.png"
        }
    }
    # png_files = {
    #         "Column_output": data_dir / "corplab/bev/Column_output.png",
    #         "Door_output": data_dir / "corplab/bev/Door_output.png",
    #         "FlowFitting_output": data_dir / "corplab/bev/FlowFitting_output.png",
    #         "FlowSegment_output": data_dir / "corplab/bev/FlowSegment_output.png",
    #         "Slab_output": data_dir / "corplab/bev/Slab_output.png",
    #         "Wall_output": data_dir / "corplab/bev/Wall_output.png",
    #         "Window_output": data_dir / "corplab/bev/Window_output.png"
    # }

    tile_manager.save_png(output_path, png_files)
    return tile_manager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, default=Path(BimDataModule.default_cfg["data_dir"])
    )
    parser.add_argument("--pixel_per_meter", type=int, default=2)
    parser.add_argument("--generate_tiles", action="store_true")
    args = parser.parse_args()
    
    args.data_dir.mkdir(exist_ok=True, parents=True)

    data_dir = args.data_dir
    csv_path = data_dir/"location_data.csv" # I added this 
    image_path = data_dir / "corplab/bev/merged_output.png"
    tiles_path = data_dir / BimDataModule.default_cfg["tiles_filename"]

    if args.generate_tiles:
        logger.info("Generating the map tiles.")
        
        prepare_osm(csv_path, image_path, tiles_path)
        (args.data_dir / ".downloaded").touch()
    else:
        print(f"Not generating tiles, kitti.pkl file downloaded incorrectly. from maploc/data/bim/prepare.py")
        logger.info("Downloading pre-generated map tiles.")
        download_file(DATA_URL + "/tiles/kitti.pkl", tiles_path)


