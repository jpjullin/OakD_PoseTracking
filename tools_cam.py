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

    cv2.imshow("Thresh", image)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_corners = []

    for c in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(c)

        if area > 5000:
            # Find the minimum area rectangle
            rect = cv2.minAreaRect(c)

            # Get the four corners
            corners = cv2.boxPoints(rect)
            corners = np.int0(corners)
            all_corners.append(corners)

    if all_corners:
        result = np.array(all_corners[0], dtype=int)
        result = result[[0, 1, 3, 2]]

    else:
        result = [
            [0, 0],
            [image.shape[1], 0],
            [0, image.shape[0]],
            [image.shape[1], image.shape[0]]
        ]

    tools_osc.find_corners = False
    return result
