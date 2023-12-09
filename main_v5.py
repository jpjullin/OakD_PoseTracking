#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import mediapipe as mp
from pathlib import Path

import tools_osc
import tools_vis
import tools_cam

# DELETE IF NO NDI
import NDIlib as ndi

# OSC
oscReceive = tools_osc.osc_in("127.0.0.1", 2223)
oscSender = tools_osc.osc_out("127.0.0.1", 2222)

# Main parameters
resolution = "720"  # Options: 800 | 720 | 400
fps = 30            # Frame/s (mono cameras)
# Show the output frame (+fps)
tools_osc.showFrame = False

# Output to NDI
out_ndi = False     # Output the frame to NDI (adds latency)

# Tracking
tracking = True     # Activate OpenPose Tracking
model = 0           # Options: 0=lite | 1=full | 2=heavy

# Night vision
laserDot = False    # Project dots for active depth
laserVal = 765      # in mA, 0..1200, don't go beyond 765
irFlood = True      # IR brightness
irVal = 750        # in mA, 0..1500

# Stereo parameters
lrcheck = True      # Better handling for occlusions
extended = False    # Closer-in minimum depth, disparity range is doubled
subpixel = False    # Better accuracy for longer distance
median = "7x7"      # Options: OFF | 3x3 | 5x5 | 7x7

# Path to save mesh files
meshDir = str(Path(__file__).parent.resolve())
# Load custom mesh on startup
if Path(__file__).with_name('left.mesh').is_file() and Path(__file__).with_name('right.mesh').is_file():
    loadMesh = True
else:
    loadMesh = False
    print("No custom mesh")

# Verbose
verbose = False     # Print info

RES_MAP = {
    '800': {'w': 1280, 'h': 800, 'res': dai.MonoCameraProperties.SensorResolution.THE_800_P},
    '720': {'w': 1280, 'h': 720, 'res': dai.MonoCameraProperties.SensorResolution.THE_720_P},
    '400': {'w':  640, 'h': 400, 'res': dai.MonoCameraProperties.SensorResolution.THE_400_P}
}
resolution = tools_osc.resforMesh = RES_MAP[resolution]

MEDIAN_MAP = {
    "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}
median = MEDIAN_MAP[median]

tools_osc.warp_pos = tools_cam.create_mesh(resolution)

# OpenPose
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=model,  # 0-2 = lite, full, heavy
                   enable_segmentation=True,
                   smooth_segmentation=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)

params = {
    'res': resolution, 'fps': fps, 'median': median,
    'lrcheck': lrcheck, 'extended': extended, 'subpixel': subpixel
}

if out_ndi:
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = 'ndi-python'
    ndi_send = ndi.send_create(send_settings)
    video_frame = ndi.VideoFrameV2()

running = True

while running:
    pipeline = tools_cam.create_pipeline(tools_osc.warp_pos, params, meshDir if loadMesh else None)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Verbose
        if verbose:
            device.setLogLevel(dai.LogLevel.DEBUG)
            device.setLogOutputLevel(dai.LogLevel.DEBUG)

        print("Starting device")

        # Dot brightness
        if laserDot:
            device.setIrLaserDotProjectorBrightness(laserVal)  # in mA, 0..1200

        # IR brightness
        if irFlood:
            device.setIrFloodLightBrightness(irVal)  # in mA, 0..1500

        # Output queues
        q_rectified = device.getOutputQueue(name="rectifiedRight", maxSize=4, blocking=False)
        q_warped = device.getOutputQueue(name="warped", maxSize=4, blocking=False)

        # Tracking values (no init)
        nose = np.empty(3)
        x = np.empty(33)
        y = np.empty(33)

        # Empty tracking (init 0)
        nose_empty = np.zeros(3)
        x_empty = np.zeros(33)
        y_empty = np.zeros(33)
        sendEmptyTracking = False

        restart_device = False

        while not restart_device:
            # Draw the mesh
            if tools_osc.showFrame:
                frame = q_rectified.get()
                if frame is not None:
                    source = frame.getCvFrame()
                    source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
                    color = (0, 0, 255)
                    for i in range(4):
                        cv2.circle(source, (tools_osc.warp_pos[i][0], tools_osc.warp_pos[i][1]), 4, color, -1)

                        if i % 2 != 2 - 1:
                            cv2.line(source, (tools_osc.warp_pos[i][0], tools_osc.warp_pos[i][1]),
                                     (tools_osc.warp_pos[i + 1][0], tools_osc.warp_pos[i + 1][1]), color, 2)

                        if i + 2 < 4:
                            cv2.line(source, (tools_osc.warp_pos[i][0], tools_osc.warp_pos[i][1]),
                                     (tools_osc.warp_pos[i + 2][0], tools_osc.warp_pos[i + 2][1]), color, 2)

                    cv2.imshow("Source", source)
            else:
                cv2.destroyWindow("Source")

            frame_warped = q_warped.get()
            if tools_osc.showFrame and not tracking:
                if frame_warped is not None:
                    cv2.imshow("Warped", frame_warped.getCvFrame())
            else:
                cv2.destroyWindow("Warped")

            if tracking:
                if frame_warped is not None:
                    # OpenPose
                    frame_warped = frame_warped.getCvFrame()
                    frame_warped = cv2.cvtColor(frame_warped, cv2.COLOR_GRAY2RGB)
                    results = pose.process(frame_warped)

                    # Get tracking values + Send OSC
                    if results.pose_landmarks:
                        for id, lm in enumerate(results.pose_landmarks.landmark):
                            if tools_osc.showFrame:
                                h, w, c = frame_warped.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame_warped, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                            if id == 0:
                                nose = [lm.x, -lm.y+1, lm.z+1]
                            else:
                                if lm.visibility > 0.8:
                                    x[id] = lm.x
                                    y[id] = -lm.y+1
                                else:
                                    x[id] = 0
                                    y[id] = 0

                        oscSender.send_message("/nose", nose)
                        oscSender.send_message("/x", x)
                        oscSender.send_message("/y", y)
                        sendEmptyTracking = True

                    else:
                        # No new tracking values
                        if sendEmptyTracking:
                            oscSender.send_message("/nose", nose_empty)
                            oscSender.send_message("/x", x_empty)
                            oscSender.send_message("/y", y_empty)
                            sendEmptyTracking = False

                    # Show fps on out frame
                    if tools_osc.showFrame:
                        cv2.imshow("Warped and tracked", tools_vis.show_frame(frame_warped))
                    else:
                        cv2.destroyWindow("Warped and tracked")

            # NDI
            if out_ndi:
                img = cv2.cvtColor(frame_warped, cv2.COLOR_BGR2BGRA)
                video_frame.data = img
                video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
                ndi.send_send_video_v2(ndi_send, video_frame)

            # Restart the device if mesh has changed
            if tools_osc.sendWarpConfig:
                print("Mesh changed, restarting...")
                tools_osc.sendWarpConfig = False
                restart_device = True

            # Save mesh files
            if tools_osc.saveMeshConfig:
                meshLeft, meshRight = tools_cam.get_mesh(device.readCalibration(), resolution)
                tools_cam.save_mesh(meshLeft, meshRight, meshDir)
                print("Mesh saved to:", meshDir)
                tools_osc.saveMeshConfig = False

            key = cv2.waitKey(1)
            # Exit
            if key == 27 or key == ord('q'):
                running = False
                tools_osc.osc_stop()
                if out_ndi:
                    ndi.send_destroy(ndi_send)
                    ndi.destroy()
                break
