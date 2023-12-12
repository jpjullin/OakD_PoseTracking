#!/usr/bin/env python3

from pathlib import Path
import depthai as dai
import numpy as np
import json
import cv2
import tools_osc


def create_mesh(res):
    return [(j * res['w'], i * res['h']) for i in range(2) for j in range(2)]


def save_mesh(path):
    with open(path, 'w') as filehandle:
        json.dump(tools_osc.warp_pos.tolist(), filehandle)


def create_pipeline(mesh, params):
    res = params['res']
    fps = params['fps']
    median = params['median']
    lrcheck = params['lrcheck']
    extended = params['extended']
    subpixel = params['subpixel']

    # Create pipeline
    pipeline = dai.Pipeline()

    # Low light tuning
    tuning_path = Path(__file__).parent.joinpath('utils/tuning_mono_low_light.bin')
    pipeline.setCameraTuningBlobPath(tuning_path)

    # Mono left camera
    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setCamera("left")

    # Mono right camera
    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_right.setCamera("right")

    # Set resolution and fps
    for mono_cam in (cam_left, cam_right):
        mono_cam.setResolution(res['res'])
        mono_cam.setFps(fps)

    # Create stereo pipeline
    stereo = pipeline.create(dai.node.StereoDepth)
    cam_left.out.link(stereo.left)
    cam_right.out.link(stereo.right)

    # Stereo settings
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(median)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    # Stream out rectified right
    xout_rectif_right = pipeline.create(dai.node.XLinkOut)
    xout_rectif_right.setStreamName("rectifiedRight")
    stereo.rectifiedRight.link(xout_rectif_right.input)

    # Create warp pipeline
    warp = pipeline.create(dai.node.Warp)
    stereo.rectifiedRight.link(warp.inputImage)

    # Warp settings
    warp.setWarpMesh(mesh, 2, 2)
    warp.setOutputSize(res['w'], res['h'])
    warp.setMaxOutputFrameSize(res['w'] * res['h'] * 3)
    warp.setHwIds([1])
    warp.setInterpolation(dai.Interpolation.NEAREST_NEIGHBOR)

    # Stream out warped
    xout_warped = pipeline.create(dai.node.XLinkOut)
    xout_warped.setStreamName("warped")
    warp.out.link(xout_warped.input)

    return pipeline


def find_corners(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, tools_osc.corners_min, tools_osc.corners_max, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # In case there is no good result
    result = np.array([
        [0, 0],
        [image.shape[1], 0],
        [0, image.shape[0]],
        [image.shape[1], image.shape[0]]
    ], dtype=int)

    for contour in contours:
        # Approximate the contour as a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(color_image, [approx], 0, (0, 0, 255), 2)

            cv2.imshow('Rectangle', color_image)

            # Reshape to a 2D array and reorder the corners
            result = np.array(approx, dtype=int).reshape(-1, 2)[[0, 3, 1, 2]]

    tools_osc.find_corners = False
    return result
