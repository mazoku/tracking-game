import cv2
import numpy as np


class SelectROI:
    def __init__(self, roi_type):
        self.pt1 = None
        self.pt2 = None
        self.marking = False  # whether we are currently marking
        self.marked = False  # whether we already marked a roi
        self.valid_roi_types = ('rect', 'circ')
        self.img = None
        self.mask = None
        if roi_type in self.valid_roi_types:
            self.roi_type = roi_type
        else:
            raise ValueError('Unknown roi type. Valid choices are {}'.format(self.valid_roi_types))
        self.roi = None  # marked ROI, rect=(top-let.x, top-left.y, width, height); circ= (center, radius), center=(x, y)

    def mouse_callback(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt1 = (x, y)
            self.pt2 = (x, y)
            self.marking = True

        if event == cv2.EVENT_MOUSEMOVE and self.marking:
            self.pt2 = (x, y)

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            self.marking = False
            self.marked = True
            if self.roi_type == 'rect':
                # record the ending (x, y) coordinates and indicate that the cropping operation is finished
                # findig top-left and bottom-right points
                if x < self.pt1[0]:
                    if y < self.pt1[1]:
                        tl = (x, y)
                        br = self.pt1
                    else:
                        tl = (x, self.pt1[1])
                        br = (self.pt1[0], y)
                else:
                    if y < self.pt1[1]:
                        tl = (self.pt1[0], y)
                        br = (x, self.pt1[1])
                    else:
                        tl = self.pt1
                        br = (x, y)
                self.pt1 = tl
                self.pt2 = br
                self.roi = (self.pt1[0], self.pt1[1], self.pt2[0] - self.pt1[0], self.pt2[1] - self.pt1[1])
                print 'Selected ROI = rect: {}'.format((self.pt1, self.pt2))
            elif self.roi_type == 'circ':
                radius = int(np.linalg.norm(np.array(self.pt1) - np.array(self.pt2)))
                self.roi = (self.pt1, radius)
                self.mask = np.zeros(self.img.shape[:2])
                cv2.circle(self.mask, self.pt1, radius, 255, -1)
                print 'Selected ROI = circ: {}'.format((self.pt1, radius))


    def reset(self):
        self.pt1 = None
        self.pt2 = None
        self.marking = False  # whether we are currently marking
        self.marked = False  # whether we already marked a roi
        self.roi = None
        self.img = None

    def select(self, img, title='Specify ROI'):
        self.img = img
        cv2.namedWindow(title)
        # cv2.imshow(title, img)
        cv2.setMouseCallback(title, self.mouse_callback)

        while True:
            if self.marking or self.marked:
                img_vis = img.copy()
                if self.roi_type == 'rect':
                    cv2.rectangle(img_vis, self.pt1, self.pt2, (0, 255, 0), 2)
                elif self.roi_type == 'circ':
                    radius = int(np.linalg.norm(np.array(self.pt1) - np.array(self.pt2)))
                    # center = (int(np.mean((self.pt1[0], self.pt2[0]))), int(np.mean((self.pt1[1], self.pt2[1]))))
                    cv2.circle(img_vis, self.pt1, radius, (0, 255, 0), 2)
                    # cv2.circle(img_vis, center, radius, (0, 255, 0), 2)
            else:
                img_vis = img
            cv2.imshow(title, img_vis)
            k = cv2.waitKey(30) & 0xFF

            if k == 27:  # Esc
                cv2.destroyAllWindows()
                break

            if k == ord('r'):  # reset
                self.reset()

        return self.pt1, self.pt2


# --------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np
    img = np.zeros((500, 500, 3))
    img[200:400, 200:400, :] = 255
    selector = SelectROI()
    roi = selector.select(img)
    print roi
