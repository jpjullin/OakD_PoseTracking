#!/usr/bin/env python3

from pathlib import Path
import mediapipe as mp
import depthai as dai
import numpy as np
import json
import cv2

import NDIlib as Ndi

import tools_osc
import tools_vis
import tools_cam

# OSC
oscReceive = tools_osc.osc_in("127.0.0.1", 2223)
oscSender = tools_osc.osc_out("127.0.0.1", 2222)

# Main parameters
resolution = "720"  # Options: 800 | 720 | 400
fps = 30            # Frame/s (mono cameras)

# Show the output frame (+fps)
tools_osc.show_frame = False

# Output to NDI
out_ndi = False     # Output the frame to NDI (adds latency)

# Tracking
tracking = True     # Activate OpenPose Tracking
model = 0           # Options: 0=lite | 1=full | 2=heavy

# Night vision
laser_dot = False   # Project dots for active depth
laser_val = 765     # in mA, 0..1200, don't go beyond 765
ir_flood = True     # IR brightness
ir_val = 1500       # in mA, 0..1500

# Stereo parameters
lrcheck = True      # Better handling for occlusions
extended = False    # Closer-in minimum depth, disparity range is doubled
subpixel = False    # Better accuracy for longer distance
median = "7x7"      # Options: OFF | 3x3 | 5x5 | 7x7

# Verbose
verbose = False     # Print (some) info about cam

RES_MAP = {
    '800': {'w': 1280, 'h': 800, 'res': dai.MonoCameraProperties.SensorResolution.THE_800_P},
    '720': {'w': 1280, 'h': 720, 'res': dai.MonoCameraProperties.SensorResolution.THE_720_P},
    '400': {'w':  640, 'h': 400, 'res': dai.MonoCameraProperties.SensorResolution.THE_400_P}
}
resolution = tools_osc.res_for_mesh = RES_MAP[resolution]

MEDIAN_MAP = {
    "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}
median = MEDIAN_MAP[median]

# Path to mesh
meshPath = Path(__file__).parent.joinpath('utils/mesh.json')

# Load custom mesh
if meshPath.is_file():
    if meshPath is not None:
        with open(str(meshPath), 'r') as data:
            mesh = json.loads(data.read())
        tools_osc.warp_pos = np.array(mesh)
        print("Custom mesh loaded")
else:
    tools_osc.warp_pos = tools_cam.create_mesh(resolution)
    print("No custom mesh")

# OpenPose
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=model,  # 0-2 = lite, full, heavy
                   enable_segmentation=True,
                   smooth_segmentation=True,
                   min_detection_confidence=0.7,
                   min_tracking_confidence=0.7)

params = {
    'res': resolution, 'fps': fps, 'median': median,
    'lrcheck': lrcheck, 'extended': extended, 'subpixel': subpixel
}

if out_ndi:
    send_settings = Ndi.SendCreate()
    send_settings.ndi_name = 'ndi-python'
    ndi_send = Ndi.send_create(send_settings)
    video_frame = Ndi.VideoFrameV2(FourCC=Ndi.FOURCC_VIDEO_TYPE_BGRX)

running = True


# Stop program
def stop_program():
    global running
    running = False
    if out_ndi:
        Ndi.send_destroy(ndi_send)
        Ndi.destroy()


# GUI button
gui_bg = np.ones((100, 200, 3), np.uint8) * 255  # Last one is color
cv2.rectangle(gui_bg, (0, 0), (gui_bg.shape[1], gui_bg.shape[0]), (0, 0, 0), -1)

gui_text = "Press q to stop"
gui_font = cv2.FONT_HERSHEY_SIMPLEX
gui_font_scale = 0.7
gui_font_thickness = 1
gui_text_size = cv2.getTextSize(gui_text, gui_font, gui_font_scale, gui_font_thickness)[0]

# Center text
gui_text_x = (gui_bg.shape[1] - gui_text_size[0]) // 2
gui_text_y = (gui_bg.shape[0] + gui_text_size[1]) // 2
cv2.putText(gui_bg, gui_text, (gui_text_x, gui_text_y), gui_font, gui_font_scale,
            (255, 255, 255), gui_font_thickness, cv2.LINE_AA)

# Display the GUI
cv2.namedWindow("Oak-D Tracking", cv2.WINDOW_NORMAL)
cv2.imshow("Oak-D Tracking", gui_bg)

while running:
    pipeline = tools_cam.create_pipeline(tools_osc.warp_pos, params)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Verbose
        if verbose:
            device.setLogLevel(dai.LogLevel.DEBUG)
            device.setLogOutputLevel(dai.LogLevel.DEBUG)

        print("Starting device")

        # Dot brightness
        if laser_dot:
            device.setIrLaserDotProjectorBrightness(laser_val)  # in mA, 0..1200

        # IR brightness
        if ir_flood:
            device.setIrFloodLightBrightness(ir_val)  # in mA, 0..1500

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
        send_empty_tracking = False

        restart_device = False
        new_start = True

        while not restart_device:
            if new_start:
                print("Device started")
                new_start = False

            # Draw the mesh
            if tools_osc.show_frame:
                frame = q_rectified.get()
                tools_vis.show_source_frame(frame)

            frame_warped = q_warped.get()
            if tools_osc.show_frame and not tracking:
                if frame_warped is not None:
                    cv2.imshow("Warped", frame_warped.getCvFrame())

            if tracking:
                if frame_warped is not None:
                    # OpenPose
                    frame_warped = frame_warped.getCvFrame()
                    frame_warped = cv2.cvtColor(frame_warped, cv2.COLOR_GRAY2RGB)
                    results = pose.process(frame_warped)

                    # Get tracking values + Send OSC
                    if results.pose_landmarks:
                        for i, lm in zip(range(33), results.pose_landmarks.landmark):  # 33 landmarks
                            if tools_osc.show_frame or out_ndi:
                                h, w, c = frame_warped.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame_warped, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                            if i == 0:
                                nose = [lm.x, -lm.y + 1, lm.z + 1]

                            x[i] = lm.x
                            y[i] = -lm.y + 1

                            # if lm.visibility > 0.8:
                            #     x[i] = lm.x
                            #     y[i] = -lm.y + 1
                            # else:
                            #     x[i] = 0
                            #     y[i] = 0

                        oscSender.send_message("/nose", nose)
                        oscSender.send_message("/x", x)
                        oscSender.send_message("/y", y)
                        send_empty_tracking = True

                    else:
                        # No new tracking values
                        if send_empty_tracking:
                            oscSender.send_message("/nose", nose_empty)
                            oscSender.send_message("/x", x_empty)
                            oscSender.send_message("/y", y_empty)
                            send_empty_tracking = False

                    # Show fps on out frame
                    if tools_osc.show_frame:
                        cv2.imshow("Warped and tracked", tools_vis.show_frame(frame_warped))

            # Find corners
            if tools_osc.find_corners:
                corners = tools_cam.find_corners(q_rectified.get().getCvFrame())
                tools_osc.warp_pos = corners

            # NDI
            if out_ndi:
                img = cv2.cvtColor(frame_warped, cv2.COLOR_BGR2BGRA)
                video_frame.data = img
                Ndi.send_send_video_v2(ndi_send, video_frame)

            # Restart the device if mesh has changed
            if tools_osc.send_warp_config:
                print("Mesh changed, restarting...")
                tools_osc.send_warp_config = False
                restart_device = True

            # Save mesh files
            if tools_osc.save_mesh_config:
                tools_cam.save_mesh(meshPath)
                print("Mesh saved to:", str(Path(meshPath)))
                tools_osc.save_mesh_config = False

            # Exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                stop_program()
                break

cv2.destroyAllWindows()
