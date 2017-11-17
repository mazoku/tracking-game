from __future__ import division

import cv2
import numpy as np
from back_projector import BackProjector
from select_roi import SelectROI
from utils import Conf
import sys


class Tracker(object):
    def __init__(self, window=None):
        self.center = None
        self._track_window = window
        self.frame = None
        self.track_space = None
        # self.ret = False
        self.prev_track_window = None
        self.found = False
        self.score = 0

    @property
    def track_window(self):
        return self._track_window

    @track_window.setter
    def track_window(self, window):
        # self.prev_track_window = self._track_window[:]
        self._track_window = window
        if window is None:
            self.center = None
        else:
            self.center = (window[0] + window[2] / 2, window[1] + window[1] / 2)

    # def get_track_window(self):
    #     print 'track window getter'
    #     return self._track_window
    #
    # def set_track_window(self, window):
    #     print 'track window setter'
    #     self._track_window = window
    #     if window is None:
    #         self.center = None
    #     else:
    #         self.center = (window[0] + window[2] / 2, window[1] + window[1] / 2)
    #
    # track_window = property(get_track_window, set_track_window)

    def trackbox_score(self, track_space):
        mask_box = np.zeros(track_space.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask_box, self.track_box, 1, thickness=-1)
        mask_space = track_space > 0

        values_space = track_space[np.nonzero(mask_box & mask_space)]
        if values_space.any():
            self.score = values_space.mean()
        else:
            self.score = 0

        # values_box = track_space[np.nonzero(mask_box)]
        # score_box = values_box.mean()

        # print 'box: {:.2f}, space:{:.2f}'.format(prob_box, prob_space)

        # cv2.imshow('masks', np.hstack((255 * mask_box, 255 * (mask_space & mask_box))))
        # cv2.waitKey(0)

    def track(self, frame, track_space=None, track_window=None):
        self.frame = frame

        if track_space is None:
            self.track_space = frame.copy()
        else:
            self.track_space = track_space

        if track_window is not None:
            self.track_window = track_window
            # self.center = (self.track_window[0] + self.track_window[2] / 2, self.track_window[1] + self.track_window[1] / 2)

        if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
            self.prev_track_window = self.track_window[:]

            # Setup the termination criteria, either 10 iteration or move by at least 1 pt
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            self.track_box, self.track_window = cv2.CamShift(self.track_space, self.track_window, term_crit)
            # track_box = elipse: ((center_x, center_y), (width, height), angle)

            # print 'box: {}, window:{}'.format(self.track_box, self.track_window)
            if self.track_window[2] == 0 or self.track_window[3] == 0:
                self.found = False
            else:
                self.found = True
                self.trackbox_score(track_space)
        else:
            self.found = False

            # if track_window[2] == 0 or track_window[3] == 0:
            #     # self.ret = True
            #     print 'in'
            #     track_window = (0, 0, 50, 50)
            #     self.found = False
            # if track_box:
            #     self.track_window = track_window
            #     # self.center = (self.track_window[0] + self.track_window[2] / 2, self.track_window[1] + self.track_window[1] / 2)
            # else:
            #     self.track_window = None


def select_object(video_capture):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            sys.exit(0)

        cv2.imshow("Pres 's' for pause", frame)

        key = cv2.waitKey(40) & 0xFF
        if key == ord('s'):
            break

    roi_selector = SelectROI()
    roi_selector.select(frame)
    roi_rect = roi_selector.roi_rect

    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

    return roi_rect, img_roi


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

    # # update the points queue
    # pts.appendleft(center)
    #
    # # loop over the set of tracked points
    # for i in xrange(1, len(pts)):
    #     # if either of the tracked points are None, ignore them
    #     if pts[i - 1] is None or pts[i] is None:
    #         continue
    #
    #     # otherwise, compute the thickness of the line and draw the connecting lines
    #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


if __name__ == '__main__':
    conf = Conf('conf.json')
    data_path = '/home/tomas/Dropbox/Data/tracking_game/ball_bowl.mp4'
    # video_capture = cv2.VideoCapture(data_path)
    video_capture = cv2.VideoCapture(0)

    # output_fname = '/home/tomas/temp/cv_seminar/backproj_tracker_in.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # video writer initialization
    # video_writer = cv2.VideoWriter(output_fname, fourcc, 30.0, (2 * frame.shape[1], frame.shape[0]), True)

    roi_rect, img_roi = select_object(video_capture)

    bp = BackProjector(space='hsv', channels=[0, 1])
    bp.model_im = img_roi
    bp.calc_model_hist(bp.model_im)

    # tracker = Tracker()
    # tracker.track_window = roi_rect

    # bp.calc_heatmap(frame, convolution=True, morphology=False)
    # cv2.rectangle(frame, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 2)
    # im_vis = np.hstack((frame, cv2.cvtColor(bp.heat_map, cv2.COLOR_GRAY2BGR)))
    # for i in range(10):
    #     video_writer.write(im_vis)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            sys.exit(0)

        # downscaling
        frame = cv2.resize(frame, None, fx=conf['scale'], fy=conf['scale'])

        # back projection
        bp.calc_heatmap(frame, convolution=True, morphology=False)

        # tracking
        # tracker.track(frame, bp.heat_map)

        # visualization
        frame_vis = frame.copy()
        visualization(frame_vis, bp.heat_map)
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