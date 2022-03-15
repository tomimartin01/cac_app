import cv2
import time
import numpy as np
from utils import image_resize

class Hough:

  def __init__(self, video, minDist, param1, param2, minRadius, maxRadius, record):
    self.video = video
    self.minDist = minDist
    self.param1 = param1
    self.param2 = param2
    self.minRadius = minRadius
    self.maxRadius = maxRadius
    self.record = record

  def bar_analysis(self, st, stframe):

    cap = cv2.VideoCapture(self.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    fps = 0
    count_frames= 0
    prevTime = 0


    while cap.isOpened():
        count_frames+=1
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 25) #cv2.bilateralFilter(gray,10,50,50)
        # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, self.minDist, param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
        
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        if self.record:
            out.write(frame)

        frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
        frame = image_resize(image = frame, width = 640)
        stframe.image(frame,channels = 'BGR',use_column_width=True)



    cap.release()
    out. release()
