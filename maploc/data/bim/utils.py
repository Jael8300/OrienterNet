# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import json
import numpy as np
from scipy.spatial.transform import Rotation

from ...utils.geo import Projection

split_files = ["test1_files.txt", "test2_files.txt", "train_files.txt"]


def parse_gps_file(path, projection: Projection = None):
    with open(path, "r") as fid:
        lat, lon, _, roll, pitch, yaw, *_ = map(float, fid.read().split())
    latlon = np.array([lat, lon])
    R_world_gps = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    t_world_gps = None if projection is None else np.r_[projection.project(latlon), 0]
    return latlon, R_world_gps, t_world_gps


def parse_split_file(path: Path):
    with open(path, "r") as fid:
        info = fid.read()
    names = []
    shifts = []
    for line in info.split("\n"):
        if not line:
            continue
        name, *shift = line.split()
        names.append(tuple(name.split("/")))
        if len(shift) > 0:
            assert len(shift) == 3
            shifts.append(np.array(shift, float))
    shifts = None if len(shifts) == 0 else np.stack(shifts)
    return names, shifts


def parse_calibration_file(path):
    calib = {}
    with open(path, "r") as fid:
        for line in fid.read().split("\n"):
            if not line:
                continue
            key, *data = line.split(" ")
            key = key.rstrip(":")
            if key.startswith("R"):
                data = np.array(data, float).reshape(3, 3)
            elif key.startswith("T"):
                data = np.array(data, float).reshape(3)
            elif key.startswith("P"):
                data = np.array(data, float).reshape(3, 4)
            calib[key] = data
    return calib


def get_camera_calibration(info_dir):
    with open(info_dir, 'r') as f:
        calib_data = json.load(f)
    
    # Extract relevant camera parameters
    width = calib_data["width"]
    height = calib_data["height"]
    focal_length = calib_data["camera_parameters"]["focal_length"]
    sensor_width = calib_data["camera_parameters"]["sensor_width"]
    sensor_height = calib_data["camera_parameters"]["sensor_height"]

    # Calculate camera matrix (K) based on focal length and sensor dimensions
    fx = focal_length * (width / sensor_width)
    fy = focal_length * (height / sensor_height)
    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # Calculate camera parameters for PINHOLE model
    size = [int(width), int(height)]
    params = K[[0, 1, 0, 1], [0, 1, 2, 2]]
    
    # Format the camera information
    camera = {
        "model": "PERSPECTIVE",  # Update model to "Perspective" for perspective projection
        "width": size[0],
        "height": size[1],
        "params": params.tolist(),  # Convert numpy array to list for JSON serialization
    }

    return camera