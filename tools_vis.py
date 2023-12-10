#!/usr/bin/env python3

import time
import cv2
import tools_osc


def show_frame(frame):
    current_time = time.time()

    # Calculate frames per second (fps)
    fps = 1 / (current_time - show_frame.previous_time)
    show_frame.previous_time = current_time

    # Display the FPS on the frame
    cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    return frame


show_frame.previous_time = time.time()


def show_source_frame(q_rectified):
    if tools_osc.show_frame:
        frame = q_rectified
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
