import cv2
import time
import mediapipe as mp
from utils import image_resize, get_position_xy, get_asimetry_y, get_asimetry_x, get_position_pair_xy
from const import keypoints, keypoints_pair

class Analyzer:

  def __init__(self, video, detection_confidence, tracking_confidence, model, record):
    self.video = video
    self.detection_confidence = detection_confidence
    self.tracking_confidence = tracking_confidence
    self.model = model
    self.record = record
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_pose = mp.solutions.pose

    
  def simple_analysis(self, st, stframe, keypoints_options):

    cap = cv2.VideoCapture(self.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    fps = 0
    count_frames= 0
    drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


    with self.mp_pose.Pose(
    min_detection_confidence= self.detection_confidence,
    min_tracking_confidence= self.tracking_confidence,
    model_complexity = self.model)as pose:
        prevTime = 0
        x_graph, y_graph = [], []

        while cap.isOpened():
            count_frames+=1
            print(count_frames)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results:
                self.mp_drawing.draw_landmarks(
                image = frame,
                landmark_list=results.pose_landmarks,
                connections= self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            for keypoint in keypoints:
                x, y = get_position_xy(results, width, height, keypoint, keypoints)

                if keypoint == keypoints_options:
                    x_graph.append(x)
                    y_graph.append(y)

            
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if self.record:
                out.write(frame)
            
            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,channels = 'BGR',use_column_width=True)


    stframe.empty()
    cap.release()
    out. release()

    return x_graph, y_graph


  def frontal_analysis(self, st, stframe, keypoints_options):

    cap = cv2.VideoCapture(self.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    fps = 0
    count_frames= 0
    drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    with self.mp_pose.Pose(
    min_detection_confidence= self.detection_confidence,
    min_tracking_confidence= self.tracking_confidence,
    model_complexity = self.model)as pose:
        prevTime = 0
        asimmetry_x_graph, asimmetry_y_graph = [], []
        while cap.isOpened():
            count_frames+=1
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results:
                self.mp_drawing.draw_landmarks(
                image = frame,
                landmark_list=results.pose_landmarks,
                connections= self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
                
                if count_frames == 1:
                    asimmetry_y0 = get_asimetry_y(results, width, height,keypoints_pair, 'HEELS')
                    asimmetry_x0 = get_asimetry_x(results, width, height,keypoints_pair, 'HEELS')
                for pair in keypoints_pair:

                    xl, yl = get_position_pair_xy(results, width, height, pair, keypoints_pair, 'l')
                    xr, yr = get_position_pair_xy(results, width, height, pair, keypoints_pair, 'r')
                    
                    asimmetry_y = get_asimetry_y(results, width, height,keypoints_pair, pair)
                    asimmetry_x = get_asimetry_x(results, width, height,keypoints_pair, pair)
                    if asimmetry_x  > 7 or asimmetry_y > 7:
                    #if asimmetry_x  > 7 + asimmetry_x0 or asimmetry_y > 7 + asimmetry_y0:
                        cv2.circle(frame,(xl, yl), 5, (0, 0, 255), -1)
                        cv2.circle(frame,(xr, yr), 5, (0, 0, 255), -1)
                    else:
                        cv2.circle(frame,(xl, yl), 5, (0, 255, 0), -1)
                        cv2.circle(frame,(xr, yr), 5, (0, 255, 0), -1)


                    if pair == keypoints_options:
                        asimmetry_x_graph.append(asimmetry_x)
                        asimmetry_y_graph.append(asimmetry_y)


            currTime = time.time()
            fps += 1 / (currTime - prevTime)
            prevTime = currTime

            if self.record:
                out.write(frame)

            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,channels = 'BGR',use_column_width=True)
            
    stframe.empty()
    cap.release()
    out. release()

    return asimmetry_x_graph, asimmetry_y_graph

