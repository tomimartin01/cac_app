import cv2
import time
import numpy as np
from utils.misc.misc import check_multiple_detection, validate_video_format

class Bar:

  def __init__(self, video, minDist, param1, param2, minRadius, maxRadius, record):
    self.video = video
    self.minDist = minDist
    self.param1 = param1
    self.param2 = param2
    self.minRadius = minRadius
    self.maxRadius = maxRadius
    self.record = record
    self.count_multiple_detection = 0


  def process(self, count_frames, total_frames, width, height, x_bar, y_bar, frame):

    cv2.putText(frame, f'{count_frames} of {total_frames}', (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25) #cv2.bilateralFilter(gray,10,50,50)
    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, self.minDist, param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)
    if circles is not None:
      if len(circles[0,:]) > 2:
          self.count_multiple_detection += 1
      circles = np.uint16(np.around(circles))
      # draw the circle
      cv2.circle(frame, (circles[0,:][0][0], circles[0,:][0][1]), circles[0,:][0][2], (0, 0, 255), 2)
      
      # draw the center of the circle
      cv2.circle(frame,(circles[0,:][0][0],circles[0,:][0][1]),2,(0, 255, 0),3)
      x_bar.append(circles[0,:][0][0])
      y_bar.append(circles[0,:][0][1])

    for i in range(0,len(x_bar)):
      cv2.circle(frame,(x_bar[i],y_bar[i]),7,(0, 255, 0),-1)

    
    return frame, x_bar, y_bar, self.count_multiple_detection