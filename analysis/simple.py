import cv2
import time
import mediapipe as mp
from utils.cv2.cv22 import image_resize, get_position_xy
from utils.misc.misc import  validate_video_format
from const.const import keypoints

class Simple:

  def __init__(self, video, detection_confidence, tracking_confidence, model, record):
    self.video = video
    self.detection_confidence = detection_confidence
    self.tracking_confidence = tracking_confidence
    self.model = model
    self.record = record
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_pose = mp.solutions.pose

    
  def analysis(self, stframe, keypoints_options):

    cap = cv2.VideoCapture(self.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not validate_video_format(total_frames, height, width):
      return None, None
    
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

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
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # draw the frame number
            cv2.putText(frame, f'{count_frames} of {total_frames}', (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            if results.pose_landmarks:
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


    cap.release()
    out. release()
    stframe.empty()

    return x_graph, y_graph
