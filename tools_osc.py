#!/usr/bin/env python3

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import threading
import numpy as np

# Global warp positions
warp_pos = np.empty(8)
resforMesh = None
sendWarpConfig = False
saveMeshConfig = False
showFrame = False

# Threading
global_thread = threading.Thread()


def osc_in(ip, port):
    disp = Dispatcher()
    disp.map("/*", handle_msg)

    server = ThreadingOSCUDPServer((ip, port), disp)
    print("Listening on {}".format(server.server_address))
    global global_thread
    global_thread = threading.Thread(target=server.serve_forever)
    global_thread.start()


def osc_out(ip, port):
    return SimpleUDPClient(ip, port)


def osc_stop():
    global global_thread
    global_thread.join()


def handle_msg(addr, *msg):
    # print(addr, msg)

    if addr == "/show_frame":
        global showFrame
        if msg[0] == 0:
            showFrame = False
        if msg[0] == 1:
            showFrame = True

    if addr == "/warp_pos":
        # Check if 8 values
        if len(msg) != 8:
            print("Not enough values, need 8")
        else:
            global warp_pos
            for i in range(4):
                warp_pos[i] = (int(msg[i*2] * resforMesh['w']), int(msg[(i*2)+1] * resforMesh['h']))

    if addr == "/warp_go":
        global sendWarpConfig
        sendWarpConfig = True

    if addr == "/warp_save":
        global saveMeshConfig
        saveMeshConfig = True
