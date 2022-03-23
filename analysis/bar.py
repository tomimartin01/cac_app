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

  def analysis(self, st, stframe):

    cap = cv2.VideoCapture(self.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not validate_video_format(total_frames, height, width):
      return None, None, None

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    count_multiple_detection = 0
    count_frames= 0
    prevTime = 0
    x_graph, y_graph = [], []

    #blank_frame = np.full((height, width, 3) , (35, 35, 35), np.uint8)
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
        if len(circles[0,:]) > 2:
            count_multiple_detection += 1
        circles = np.uint16(np.around(circles))
        # draw the circle
        cv2.circle(frame, (circles[0,:][0][0], circles[0,:][0][1]), circles[0,:][0][2], (0, 0, 255), 2)
        
        # draw the center of the circle
        cv2.circle(frame,(circles[0,:][0][0],circles[0,:][0][1]),2,(0, 255, 0),3)
        x_graph.append(circles[0,:][0][0])
        y_graph.append(circles[0,:][0][1])

        for i in range(0,len(x_graph)):
          cv2.circle(frame,(x_graph[i],y_graph[i]),7,(0, 255, 0),-1)
          #cv2.circle(blank_frame,(x_graph[i],y_graph[i]),7,(0, 255, 0),-1)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        
        #frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
        #frame = image_resize(image = frame, width = 640)
      # draw the frame number
      cv2.putText(frame, f'{count_frames} of {total_frames}', (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
      stframe.image(frame,channels = 'BGR',use_column_width='auto')
      if self.record:
          out.write(frame)

    cap.release()
    out. release()
    stframe.empty()

    # blank_frame = cv2.resize(blank_frame,(0,0),fx = 0.8 , fy = 0.8)
    # blank_frame = image_resize(image = blank_frame, width = 640)

    multiple_detection = check_multiple_detection(count_multiple_detection, total_frames)

    return x_graph, y_graph, multiple_detection

