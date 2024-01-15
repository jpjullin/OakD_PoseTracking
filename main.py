#!/usr/bin/env python3

from pathlib import Path
import mediapipe as mp
import depthai as dai
import numpy as np
import tools
import cv2

# Load configuration
config = tools.Config(model=0, ip="192.168.3.1")  # Options: 0=lite | 1=full | 2=heavy

config.show_frame = False   # Show the output frame (+fps)
config.ir_val = 1           # IR brightness (0 to 1)
config.depth = False        # Track on depth image

# Initialize OSC and load custom mesh
tools.initialize_osc(config)
tools.load_custom_mesh(config)

# Initialize OpenPose
mpPose = mp.solutions.pose
pose = mpPose.Pose(
    model_complexity=config.mp_pose_model_complexity,
    enable_segmentation=config.mp_pose_enable_segmentation,
    smooth_segmentation=config.mp_pose_smooth_segmentation,
    min_detection_confidence=config.mp_pose_min_detection_confidence,
    min_tracking_confidence=config.mp_pose_min_tracking_confidence
)

config.running = True

# Display the GUI
cv2.namedWindow("Oak-D Tracking", cv2.WINDOW_NORMAL)
cv2.imshow("Oak-D Tracking", tools.create_gui_bg())

while config.running:
    # Create pipeline using warp_pos from tools module and config parameters
    pipeline = tools.create_pipeline(config)

    # Connect to device and start pipeline
    with (dai.Device(pipeline) as device):

        # Verbose
        if config.verbose:
            device.setLogLevel(dai.LogLevel.DEBUG)
            device.setLogOutputLevel(dai.LogLevel.DEBUG)

        print("Starting device")

        # Dot brightness
        device.setIrLaserDotProjectorIntensity(config.laser_val)

        # IR brightness
        device.setIrFloodLightIntensity(config.ir_val)

        # Output queues
        if config.depth:
            q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        q_rectified = device.getOutputQueue(name="rectifiedRight", maxSize=4, blocking=False)
        q_warped = device.getOutputQueue(name="warped", maxSize=4, blocking=False)

        # Tracking values
        nose = np.zeros(3)
        x = np.zeros(33)
        y = np.zeros(33)

        restart_device = False
        new_start = True

        while not restart_device:
            if new_start:
                print("Device started")
                new_start = False

            # Draw the mesh
            if config.show_frame:
                frame = q_rectified.get()
                tools.show_source_frame(frame, config)

            frame_warped = q_warped.get().getCvFrame()
            if frame_warped is not None and config.depth:
                frame_warped = tools.get_disparity_frame(frame_warped, config).astype(np.uint8)

            if config.show_frame and not config.tracking:
                if frame_warped is not None:
                    cv2.imshow("Warped", frame_warped)

            if config.tracking:
                if frame_warped is not None:
                    # OpenPose
                    if not config.depth:
                        frame_warped = cv2.cvtColor(frame_warped, cv2.COLOR_GRAY2RGB)
                    results = pose.process(frame_warped)

                    # Get tracking values + Send OSC
                    if results.pose_landmarks:
                        for i, lm in zip(range(33), results.pose_landmarks.landmark):  # 33 landmarks
                            if config.show_frame:
                                h, w, c = frame_warped.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame_warped, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                            lm.y = -lm.y + 1

                            if 0 < lm.y < 1 and 0 < lm.x < 1:
                                x[i] = lm.x
                                y[i] = lm.y

                                if i == 0:
                                    lm.z = lm.z + 1
                                    nose = [lm.x, lm.y, lm.z]

                        # Check if any valid values were found
                        if any(0 <= xi <= 1 for xi in x) and any(0 <= yi <= 1 for yi in y):
                            config.osc_sender.send_message("/nose", nose)
                            config.osc_sender.send_message("/x", x)
                            config.osc_sender.send_message("/y", y)
                        else:
                            # If no valid values found, send the last valid values
                            config.osc_sender.send_message("/nose", nose)
                            config.osc_sender.send_message("/x", x)
                            config.osc_sender.send_message("/y", y)

                    # Show fps on out frame
                    if config.show_frame:
                        cv2.imshow("Warped and tracked", tools.show_frame(frame_warped))

            # Find corners
            if config.find_corners:
                corners = tools.find_corners(q_rectified.get().getCvFrame(), config)
                config.warp_pos = corners

            # Restart the device if mesh has changed
            if config.send_warp_config:
                print("Mesh changed, restarting...")
                config.send_warp_config = False
                restart_device = True

            # Save mesh files
            if config.save_mesh_config:
                tools.save_mesh(config.mesh_path, config.warp_pos)
                print("Mesh saved to:", str(Path(config.mesh_path)))
                config.save_mesh_config = False

            # Exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                tools.stop_program(config)
                break

cv2.destroyAllWindows()
