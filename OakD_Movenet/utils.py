#!/usr/bin/env python3

from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from threading import Thread
from pathlib import Path
import depthai as dai
import numpy as np
import json
import time
import cv2


class Config:
    def __init__(self, nn_model='lightning', ip='127.0.0.1'):
        # OSC
        self.osc_receive_ip = '0.0.0.0'
        self.osc_receive_port = 2223

        self.osc_send_ip = ip
        self.osc_send_port = 2222
        self.osc_sender = None

        # Main parameters
        self.cam_source = 'left'  # Options: left | right
        self.resolution = '720'  # Options: 800 | 720 | 400
        self.fps = 30  # Frame/s (mono cameras)
        self.show_frame = False  # Show the output frame (+fps)

        # Night vision
        self.laser_val = 0  # Project dots for active depth (0 to 1)
        self.ir_val = 1  # IR Brightness (0 to 1)

        # Stereo parameters
        self.lrcheck = True  # Better handling for occlusions
        self.extended = False  # Closer-in minimum depth, disparity range is doubled
        self.subpixel = True  # Better accuracy for longer distance
        self.median = "7x7"  # Options: OFF | 3x3 | 5x5 | 7x7

        # Verbose
        self.verbose = False  # Print (some) info about cam

        # Mesh
        self.mesh_path = Path(__file__).parent.joinpath('utils/mesh.json')
        self.save_mesh_config = False

        # Tracking model
        self.nn_model = nn_model
        self.check_consistency = True
        self.consistency_threshold = 2.2

        self.nn_models = {
            'lightning': {'path': 'utils/movenet_singlepose_lightning_U8_transpose.blob', 'input': 192},
            'thunder': {'path': 'utils/movenet_singlepose_thunder_U8_transpose.blob', 'input': 256}
        }
        self.nn_model = self.nn_models[self.nn_model]

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
    print("Listening on port", config.osc_receive_port)

    global_thread = Thread(target=server.serve_forever, daemon=True)
    global_thread.start()

    # Sender
    config.osc_sender = SimpleUDPClient(config.osc_send_ip, config.osc_send_port)
    print("Sending on", config.osc_send_ip, config.osc_send_port)


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
            if i * 2 < len(msg) and (i * 2) + 1 < len(msg)
            else (0, 0)
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
    if not isinstance(warp_pos, list):
        warp_pos = warp_pos.tolist()
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

    # Stream out rectified
    xout_rectif = pipeline.create(dai.node.XLinkOut)
    xout_rectif.setStreamName("rectified")
    if config.cam_source == 'left':
        stereo.rectifiedLeft.link(xout_rectif.input)
    elif config.cam_source == 'right':
        stereo.rectifiedRight.link(xout_rectif.input)

    # Create warp pipeline
    warp = pipeline.create(dai.node.Warp)
    if config.cam_source == 'left':
        stereo.rectifiedLeft.link(warp.inputImage)
    elif config.cam_source == 'right':
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

    # Neural network
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(Path(__file__).parent.joinpath(config.nn_model['path']))
    # detection_nn.passthrough.link(xout_warped.input)

    # Stream out nn
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    xout_nn.input.setBlocking(False)

    detection_nn.out.link(xout_nn.input)

    # Image manip
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(config.nn_model['input'], config.nn_model['input'])
    manip.setKeepAspectRatio(False)
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(1228800)
    warp.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

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

            cv2.imshow('Mesh', color_image)

            # Reshape to a 2D array and reorder the corners
            result = np.array(approx, dtype=int).reshape(-1, 2)[[0, 3, 1, 2]]

    config.find_corners = False
    return result


# --------------------------------------- TRACKING UTILITIES ---------------------------------------
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


def check_spatial_consistency(x_values, y_values, conf_scores, threshold=1.0):
    """
    Check spatial consistency of joint positions.

    Parameters:
    - x_values (list): List of x coordinates for all joints.
    - y_values (list): List of y coordinates for all joints.
    - threshold (float): Threshold for spatial consistency check.

    Returns:
    - consistent (bool): True if spatially consistent, False otherwise.
    """

    # Define joint pairs for spatial consistency check
    joint_pairs = [('nose', 'left_eye'), ('left_eye', 'right_eye'), ('right_eye', 'left_ear'),
                   ('left_ear', 'right_ear'), ('left_shoulder', 'right_shoulder'),
                   ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
                   ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
                   ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                   ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
                   ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')]

    # Combine x, y, and confidence values into a list of (x, y, confidence) tuples
    joints_with_confidence = list(zip(x_values, y_values, conf_scores))

    # Dynamic thresholding based on confidence scores
    threshold = np.mean(conf_scores) * threshold

    # Check spatial consistency for each joint pair
    for joint_pair in joint_pairs:
        joint1, joint2 = KEYPOINT_DICT[joint_pair[0]], KEYPOINT_DICT[joint_pair[1]]
        distance = np.linalg.norm(
            np.array(joints_with_confidence[joint1][:2]) - np.array(joints_with_confidence[joint2][:2]))
        if distance > threshold:
            return False

    return True


# --------------------------------------- VISUALISATION ---------------------------------------
def draw_fps(frame):
    color = (0, 255, 0)

    current_time = time.time()
    elapsed_time = current_time - draw_fps.previous_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0

    # Moving average
    alpha = 0.9  # Smoothing factor
    draw_fps.fps_average = alpha * draw_fps.fps_average + (1 - alpha) * fps

    cv2.putText(frame, str(int(draw_fps.fps_average)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    draw_fps.previous_time = current_time

    return frame


draw_fps.previous_time = time.time()
draw_fps.fps_average = 0.0


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


def draw_kpts(frame, kpts, config):
    # Parameters
    color = (0, 255, 0)
    keypoint_radius = 3
    line_thickness = 2

    # Interleave and scale keypoints
    kpts = np.column_stack(
        (kpts[0] * config.resolution['w'],
         config.resolution['h'] - kpts[1] * config.resolution['h'])
    )

    skeleton = np.array([
        [16, 14], [14, 12], [17, 15], [15, 13],
        [12, 13], [6, 12], [7, 13], [6, 7],
        [6, 8], [7, 9], [8, 10], [9, 11],
        [2, 3], [1, 2], [1, 3], [2, 4],
        [3, 5], [4, 6], [5, 7]
    ]) - 1  # Convert to 0-based indexing

    # Draw keypoints
    for kpt in kpts:
        cv2.circle(frame, tuple(kpt.astype(int)), keypoint_radius, color, thickness=-1, lineType=cv2.FILLED)

    # Draw lines
    for pair in skeleton:
        pt1, pt2 = kpts[pair].astype(int)
        cv2.line(frame, tuple(pt1), tuple(pt2), color, line_thickness)

    return frame


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


# --------------------------------------- PROGRAM ---------------------------------------
def stop_program(config):
    config.running = False
