from __future__ import division

import sys
import numpy as np
import cv2
from utils import Conf
from select_roi import SelectROI


def select_object(video_capture):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            sys.exit(0)

        cv2.imshow("Pres 's' for pause", frame)

        key = cv2.waitKey(40) & 0xFF
        if key == ord('s'):
            break

    roi_selector = SelectROI('circ')
    roi_selector.select(frame)
    roi = roi_selector.roi

    if roi_selector.roi_type == 'rect':
        img_roi = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    else:
        img_roi = frame[roi[0][1] - roi[1]:roi[0][1] + roi[1], roi[0][0] - roi[1]:roi[0][0] + roi[1]]

    return frame, roi_selector.mask, img_roi, roi


def create_model(img, mask, img_roi, roi):
    cv2.imshow('img', img_roi)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)


def visualization(frame, mask):
    mask = np.where(mask > 50, 255, 0).astype(np.uint8)

    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


if __name__ == '__main__':
    conf = Conf('conf.json')
    data_path = '/home/tomas/Dropbox/Data/tracking_game/ball_bowl.mp4'
    video_capture = cv2.VideoCapture(data_path)
    # video_capture = cv2.VideoCapture(0)

    img, mask, img_roi, roi = select_object(video_capture)

    model = create_model(img, mask, img_roi, roi)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            sys.exit(0)

        # downscaling
        frame = cv2.resize(frame, None, fx=conf['scale'], fy=conf['scale'])

        # get mask
        # TODO: najit meze pro inrange
        # TODO: zkusit grabcut

        # visualization
        frame_vis = frame.copy()
        visualization(frame_vis, mask)
        # if tracker.found:
        #     x, y, w, h = tracker.track_window
        #     cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im_vis = np.hstack((frame_vis, cv2.cvtColor(bp.heat_map, cv2.COLOR_GRAY2BGR)))

        # upscaling
        im_vis = cv2.resize(im_vis, None, fx=1/conf['scale'], fy=1/conf['scale'])

        # video_writer.write(im_vis)
        cv2.imshow('CamShift tracker', im_vis)

        key = cv2.waitKey(40) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            select_object(frame)