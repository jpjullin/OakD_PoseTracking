#!/usr/bin/env python3

from utils import *

# Load configuration
config = Config(nn_model='lightning', ip="192.168.3.1")

# Initialize OSC and load custom mesh
initialize_osc(config)
load_custom_mesh(config)

# Display the GUI
cv2.namedWindow("Oak-D Tracking", cv2.WINDOW_NORMAL)
cv2.imshow("Oak-D Tracking", create_gui_bg())

# Program
config.running = True
while config.running:
    # Create pipeline using warp_pos
    pipeline = create_pipeline(config)

    # Connect to device and start pipeline
    with (dai.Device(pipeline) as device):

        # Verbose
        if config.verbose:
            device.setLogLevel(dai.LogLevel.DEBUG)
            device.setLogOutputLevel(dai.LogLevel.DEBUG)

        print('...')
        print("Starting device")

        # Laser Dot + Infrared
        device.setIrLaserDotProjectorIntensity(config.laser_val)
        device.setIrFloodLightIntensity(config.ir_val)

        # Output queues
        q_rectified = device.getOutputQueue(name="rectified", maxSize=4, blocking=False)
        q_warped = device.getOutputQueue(name="warped", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        # Tracking values
        nose = np.zeros(3)
        x = np.zeros(17)
        y = np.zeros(17)

        restart_device = False
        new_start = True

        previous_time = 0

        while not restart_device:
            if new_start:
                print("Device started")
                new_start = False

            # Get predictions
            in_nn = q_nn.get().getLayerFp16('Identity')

            if len(in_nn) > 0:

                for i in range(17):
                    kpt_x = in_nn[3 * i + 1]
                    kpt_y = in_nn[3 * i] * -1 + 1

                    if 0 < kpt_x < 1 and 0 < kpt_y < 1:
                        x[i] = kpt_x
                        y[i] = kpt_y

                        if i == 0:
                            nose = [kpt_x, kpt_y]

                if config.show_frame:
                    # Source frame
                    frame = q_rectified.get()
                    show_source_frame(frame, config)

                    # Warped and tracked frame
                    frame_warped = q_warped.get().getCvFrame()
                    frame_warped = cv2.cvtColor(frame_warped, cv2.COLOR_GRAY2BGR)

                    frame = draw_kpts(frame_warped, [x, y], config)
                    cv2.imshow('Warped & Tracked', draw_fps(frame_warped))

                if not config.show_frame:
                    cv2.destroyWindow('Source')
                    cv2.destroyWindow('Mesh')
                    cv2.destroyWindow('Warped & Tracked')

            config.osc_sender.send_message("/nose", nose)
            config.osc_sender.send_message("/x", x)
            config.osc_sender.send_message("/y", y)

            # Find corners
            if config.find_corners:
                corners = find_corners(q_rectified.get().getCvFrame(), config)
                config.warp_pos = corners

            # Restart the device if mesh has changed
            if config.send_warp_config:
                print("Mesh changed, restarting...")
                config.send_warp_config = False
                restart_device = True

            # Save mesh files
            if config.save_mesh_config:
                save_mesh(config.mesh_path, config.warp_pos)
                print("Mesh saved to:", str(Path(config.mesh_path)))
                config.save_mesh_config = False

            # Exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                stop_program(config)
                break

cv2.destroyAllWindows()
