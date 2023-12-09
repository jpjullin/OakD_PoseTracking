#!/usr/bin/env python3

from pathlib import Path
import depthai as dai
import numpy as np
import cv2


def create_mesh(res):
    mesh = []
    for i in range(2):
        for j in range(2):
            x = j * res['w']
            y = i * res['h']
            mesh.append((x, y))

    return mesh


def get_mesh(calibData, resolution):
    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, resolution['w'], resolution['h']))
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
    R1 = np.array(calibData.getStereoLeftRectificationRotation())
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, resolution['w'], resolution['h']))
    d2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
    R2 = np.array(calibData.getStereoRightRectificationRotation())
    mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, M2, (resolution['w'], resolution['h']), cv2.CV_32FC1)
    mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, M2, (resolution['w'], resolution['h']), cv2.CV_32FC1)

    meshCellSize = 16
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    return meshLeft, meshRight


def save_mesh(meshLeft, meshRight, outputPath):
    meshLeft.tofile(outputPath + "/left.mesh")
    meshRight.tofile(outputPath + "/right.mesh")


def create_pipeline(mesh, params, meshPath):
    res = params['res']
    fps = params['fps']
    median = params['median']
    lrcheck = params['lrcheck']
    extended = params['extended']
    subpixel = params['subpixel']

    # Create pipeline
    pipeline = dai.Pipeline()

    # Low light tuning
    tuning_path = Path(__file__).with_name('tuning_mono_low_light.bin')
    pipeline.setCameraTuningBlobPath(tuning_path)

    # Mono left camera
    camLeft = pipeline.create(dai.node.MonoCamera)
    camLeft.setCamera("left")

    # Mono right camera
    camRight = pipeline.create(dai.node.MonoCamera)
    camRight.setCamera("right")

    # Set resolution and fps
    for monoCam in (camLeft, camRight):
        monoCam.setResolution(res['res'])
        monoCam.setFps(fps)

    # Create stereo pipeline
    stereo = pipeline.create(dai.node.StereoDepth)
    camLeft.out.link(stereo.left)
    camRight.out.link(stereo.right)

    # Stereo settings
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(median)  # KERNEL_7x7 default
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    # Stream out rectified right
    xoutRectifRight = pipeline.create(dai.node.XLinkOut)
    xoutRectifRight.setStreamName("rectifiedRight")
    stereo.rectifiedRight.link(xoutRectifRight.input)

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
    xoutWarped = pipeline.create(dai.node.XLinkOut)
    xoutWarped.setStreamName("warped")
    warp.out.link(xoutWarped.input)

    # Load custom mesh
    if meshPath is not None:
        # --------------------- NOT WORKING
        stereo.loadMeshFiles(
            Path(__file__).with_name('left.mesh'),
            Path(__file__).with_name('right.mesh'))

        print("Custom mesh loaded")
        print("Not really I lied")

    return pipeline
