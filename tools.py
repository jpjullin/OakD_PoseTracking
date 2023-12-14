#!/usr/bin/env python3

from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from threading import Thread
from pathlib import Path
import depthai as dai
import NDIlib as Ndi
import numpy as np
import time
import json
import cv2


class Config:
    def __init__(self):
        # OSC
        self.osc_receive_ip = "127.0.0.1"
        self.osc_receive_port = 2223

        self.osc_send_ip = "127.0.0.1"
        self.osc_send_port = 2222
        self.osc_sender = None

        # Main parameters
        self.resolution = "720"  # Options: 800 | 720 | 400
        self.fps = 30            # Frame/s (mono cameras)
        self.show_frame = False  # Show the output frame (+fps)

        # Output to NDI
        self.out_ndi = False     # Output the frame to NDI (adds latency)

        # Tracking
        self.tracking = True     # Activate OpenPose Tracking
        self.model = 0           # Options: 0=lite | 1=full | 2=heavy

        # Depth tracking
        self.depth = False       # Track on depth image
        self.max_disparity = 0.  # Maximum disparity (get from camera later)

        # Color map
        self.cv_color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_BONE)
        self.cv_color_map[0] = [0, 0, 0]

        # Night vision
        self.laser_dot = False   # Project dots for active depth
        self.laser_val = 765     # in mA, 0..1200, don't go beyond 765
        self.ir_flood = True     # IR brightness
        self.ir_val = 1500       # in mA, 0..1500

        # Stereo parameters
        self.lrcheck = True      # Better handling for occlusions
        self.extended = False    # Closer-in minimum depth, disparity range is doubled
        self.subpixel = True     # Better accuracy for longer distance
        self.median = "7x7"      # Options: OFF | 3x3 | 5x5 | 7x7

        # Verbose
        self.verbose = False     # Print (some) info about cam

        # Mesh
        self.mesh_path = Path(__file__).parent.joinpath('utils/mesh.json')
        self.save_mesh_config = False

        # OpenPose
        self.mp_pose_model_complexity = self.model
        self.mp_pose_enable_segmentation = True
        self.mp_pose_smooth_segmentation = True
        self.mp_pose_min_detection_confidence = 0.7
        self.mp_pose_min_tracking_confidence = 0.7

        # Resolution
        self.res_map = {
            '800': {'w': 1280, 'h': 800, 'res': dai.MonoCameraProperties.SensorResolution.THE_800_P},
            '720': {'w': 1280, 'h': 720, 'res': dai.MonoCameraProperties.SensorResolution.THE_720_P},
            '400': {'w': 640, 'h': 400, 'res': dai.MonoCameraProperties.SensorResolution.THE_400_P}
        }
        self.resolution = self.res_map[self.resolution]

        # Median kernel
        self.median_map = {
            "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
            "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
            "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
            "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
        }
        self.median = self.median_map[self.median]

        # Warp
        self.warp_pos = np.zeros(8, dtype=int)
        self.corners_min = 0
        self.corners_max = 255
        self.find_corners = False
        self.send_warp_config = False

        # Running state
        self.running = False


# --------------------------------------- OSC ---------------------------------------
def initialize_osc(config):
    # Receiver
    disp = Dispatcher()
    disp.map("/*", lambda osc_address, *msg: handle_msg(osc_address, msg, config))

    server = ThreadingOSCUDPServer((config.osc_receive_ip, config.osc_receive_port), disp)
    print("Listening on {}".format(server.server_address))

    global_thread = Thread(target=server.serve_forever, daemon=True)
    global_thread.start()

    # Sender
    config.osc_sender = SimpleUDPClient(config.osc_send_ip, config.osc_send_port)


def handle_msg(osc_address, msg, config):
    address_handlers = {
        "/show_frame": lambda: setattr(config, 'show_frame', bool(msg[0])) or (
            cv2.destroyWindow("Source") if not bool(msg[0]) else None,
            cv2.destroyWindow("Warped") if not bool(msg[0]) else None,
            cv2.destroyWindow("Warped and tracked") if not bool(msg[0]) else None,
            cv2.destroyWindow("Rectangle") if not bool(msg[0]) else None,
            setattr(config, 'show_frame', bool(msg[0]))
        ),
        "/warp_pos": lambda: setattr(config, 'warp_pos', [
            (int(msg[i * 2] * config.resolution['w']), int(msg[(i * 2) + 1] * config.resolution['h']))
            for i in range(4)
        ]),
        "/warp_go": lambda: setattr(config, 'send_warp_config', True),
        "/warp_save": lambda: setattr(config, 'save_mesh_config', True),
        "/corners_find": lambda: setattr(config, 'find_corners', True),
        "/corners_thresh": lambda: setattr(config, 'corners_min', msg[0]) and setattr(config, 'corners_max', msg[1]),
    }
    handler = address_handlers.get(osc_address)
    if handler:
        handler()


# --------------------------------------- CAMERA ---------------------------------------
def create_mesh(res):
    return [(j * res['w'], i * res['h']) for i in range(2) for j in range(2)]


def save_mesh(path, warp_pos):
    with open(path, 'w') as filehandle:
        json.dump(warp_pos, filehandle)


def load_custom_mesh(config):
    if config.mesh_path.is_file():
        if config.mesh_path is not None:
            with open(str(config.mesh_path), 'r') as data:
                mesh = json.loads(data.read())
            config.warp_pos = np.array(mesh)
            print("Custom mesh loaded")
    else:
        config.warp_pos = create_mesh(config.resolution)
        print("No custom mesh")


def get_disparity_frame(frame, config):
    disp = (frame * (255.0 / config.max_disparity)).astype(np.uint8)
    disp = cv2.applyColorMap(disp, config.cv_color_map)
    return disp


def create_pipeline(config):
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
        mono_cam.setResolution(config.resolution['res'])
        mono_cam.setFps(config.fps)

    # Create stereo pipeline
    stereo = pipeline.create(dai.node.StereoDepth)
    cam_left.out.link(stereo.left)
    cam_right.out.link(stereo.right)

    # Stereo settings
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(config.median)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setLeftRightCheck(config.lrcheck)
    stereo.setExtendedDisparity(config.extended)
    stereo.setSubpixel(config.subpixel)

    # Get max disparity
    config.max_disparity = stereo.initialConfig.getMaxDisparity()

    # Stream out depth (actually disparity)
    if config.depth:
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.disparity.link(xout_depth.input)

    # Stream out rectified right
    xout_rectif_right = pipeline.create(dai.node.XLinkOut)
    xout_rectif_right.setStreamName("rectifiedRight")
    stereo.rectifiedRight.link(xout_rectif_right.input)

    # Create warp pipeline
    warp = pipeline.create(dai.node.Warp)
    if config.depth:
        stereo.disparity.link(warp.inputImage)
    else:
        stereo.rectifiedRight.link(warp.inputImage)

    # Warp settings
    warp.setWarpMesh(config.warp_pos, 2, 2)
    warp.setOutputSize(config.resolution['w'], config.resolution['h'])
    warp.setMaxOutputFrameSize(config.resolution['w'] * config.resolution['h'] * 3)
    warp.setHwIds([1])
    warp.setInterpolation(dai.Interpolation.NEAREST_NEIGHBOR)

    # Stream out warped
    xout_warped = pipeline.create(dai.node.XLinkOut)
    xout_warped.setStreamName("warped")
    warp.out.link(xout_warped.input)

    return pipeline


def find_corners(image, config):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, config.corners_min, config.corners_max, cv2.THRESH_BINARY)

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

    config.find_corners = False
    return result


# --------------------------------------- VISUALISATION DEF ---------------------------------------
def show_frame(frame):
    current_time = time.time()
    fps = 1 / (current_time - show_frame.previous_time)
    show_frame.previous_time = current_time

    cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    return frame


show_frame.previous_time = time.time()


def show_source_frame(q_rectified, config):
    if config.show_frame:
        frame = q_rectified
        if frame is not None:
            source = frame.getCvFrame()
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            color = (0, 0, 255)
            for i in range(4):
                cv2.circle(source, (config.warp_pos[i][0], config.warp_pos[i][1]), 4, color, -1)

                if i % 2 != 2 - 1:
                    cv2.line(source, (config.warp_pos[i][0], config.warp_pos[i][1]),
                             (config.warp_pos[i + 1][0], config.warp_pos[i + 1][1]), color, 2)

                if i + 2 < 4:
                    cv2.line(source, (config.warp_pos[i][0], config.warp_pos[i][1]),
                             (config.warp_pos[i + 2][0], config.warp_pos[i + 2][1]), color, 2)

            cv2.imshow("Source", source)


def create_gui_bg():
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

    return gui_bg


def stop_program(config, ndi_send=None):
    config.running = False
    if config.out_ndi and ndi_send:
        Ndi.send_destroy(ndi_send)
        Ndi.destroy()
