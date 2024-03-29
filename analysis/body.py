import cv2
import mediapipe as mp
from utils.cv22.cv22 import get_position_xy

from const.const import keypoints

class Body:

  def __init__(self, video, detection_confidence, tracking_confidence, model):
    self.video = video
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_pose = mp.solutions.pose
    self.pose = self.mp_pose.Pose(
    min_detection_confidence= detection_confidence,
    min_tracking_confidence= tracking_confidence,
    model_complexity = model)
    self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


  def process(self, keypoints_options, count_frames, total_frames, width, height, x_body, y_body, frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.pose.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # draw the frame number
    cv2.putText(frame, f'{count_frames} of {total_frames}', (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
    if results.pose_landmarks:
      self.mp_drawing.draw_landmarks(
      image = frame,
      landmark_list=results.pose_landmarks,
      connections= self.mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=self.drawing_spec,
      connection_drawing_spec= self.drawing_spec)

      for keypoint in keypoints:
        x, y = get_position_xy(results, width, height, keypoint, keypoints)

        if keypoint == keypoints_options:
          x_body.append(x)
          y_body.append(y)
    
    return frame, x_body, y_body