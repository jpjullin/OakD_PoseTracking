#!/usr/bin/env python3

from threading import Thread

import cv2
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
import numpy as np

# Constants
OSC_ADDRESS_SHOW_FRAME = "/show_frame"
OSC_ADDRESS_WARP_POS = "/warp_pos"
OSC_ADDRESS_WARP_GO = "/warp_go"
OSC_ADDRESS_WARP_SAVE = "/warp_save"
OSC_ADDRESS_CORNERS_FIND = "/corners_find"
OSC_ADDRESS_CORNERS_THRESH = "/corners_thresh"

# Global variables
warp_pos = np.zeros(8, dtype=int)
res_for_mesh = None
send_warp_config = False
save_mesh_config = False
show_frame = False
find_corners = False
corners_min = 0
corners_max = 255

# Threading
global_thread = Thread(daemon=True)
stop_thread = False


def osc_in(ip, port):
    disp = Dispatcher()
    disp.map("/*", handle_msg)

    server = ThreadingOSCUDPServer((ip, port), disp)
    print("Listening on {}".format(server.server_address))
    global global_thread
    global_thread = Thread(target=server.serve_forever, daemon=True)
    global_thread.start()


def osc_out(ip, port):
    return SimpleUDPClient(ip, port)


def handle_msg(osc_address, *msg):
    # print(osc_address, msg)

    global show_frame, send_warp_config, save_mesh_config, warp_pos, res_for_mesh, \
        find_corners, corners_min, corners_max

    if osc_address == OSC_ADDRESS_SHOW_FRAME:
        cv2.destroyWindow("Source")
        cv2.destroyWindow("Warped")
        cv2.destroyWindow("Warped and tracked")
        show_frame = bool(msg[0])

    elif osc_address == OSC_ADDRESS_WARP_POS:
        if len(msg) != 8:
            print("Not enough values, need 8")
        else:
            for i in range(4):
                warp_pos[i] = (int(msg[i * 2] * res_for_mesh['w']), int(msg[(i * 2) + 1] * res_for_mesh['h']))

    elif osc_address == OSC_ADDRESS_WARP_GO:
        send_warp_config = True

    elif osc_address == OSC_ADDRESS_WARP_SAVE:
        save_mesh_config = True

    elif osc_address == OSC_ADDRESS_CORNERS_FIND:
        find_corners = True

    elif osc_address == OSC_ADDRESS_CORNERS_THRESH:
        corners_min = msg[0]
        corners_max = msg[1]

