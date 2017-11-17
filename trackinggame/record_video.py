from __future__ import division

from utils import Conf
import os
import cv2


def run(conf):
    video_capture = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # warming up camera
    for i in range(10):
        ret, frame = video_capture.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # initializing video writer
    output_fname = os.path.join(conf['dropbox_data_dir'], 'ball_bowl.mp4')
    video_writer = cv2.VideoWriter(output_fname, fourcc, 15.0, (frame.shape[1], frame.shape[0]), True)

    while True:
        ret, frame = video_capture.read()

        # show image
        cv2.imshow('video stream', frame)

        # write frame
        video_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break


if __name__ == '__main__':
    # configuration variables - config file
    conf_path = 'conf.json'
    conf = Conf(conf_path)

    # start the recording script
    run(conf)
