import cv2
import mediapipe as mp
from utils.cv22.cv22 import  get_asimetry_y, get_asimetry_x, get_position_pair_xy
from const.const import keypoints_pair

class Asimmetry:

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
        self.asimmetry_threshold = False
        self.asimmetry_y0 = 0
        self.asimmetry_y0 = 0
        

    def process(self, keypoints_options, count_frames, total_frames, width, height, x_graph, y_graph, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame, f'{count_frames} of {total_frames}', (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image = frame,
                landmark_list=results.pose_landmarks,
                connections= self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec= self.drawing_spec)

            if not self.asimmetry_threshold:
                self.asimmetry_y0 = get_asimetry_y(results, width, height,keypoints_pair, 'HEELS')
                self.asimmetry_x0 = get_asimetry_x(results, width, height,keypoints_pair, 'HEELS')
                self.asimmetry_threshold = True

            for pair in keypoints_pair:

                xl, yl = get_position_pair_xy(results, width, height, pair, keypoints_pair, 'l')
                xr, yr = get_position_pair_xy(results, width, height, pair, keypoints_pair, 'r')
                
                asimmetry_y = get_asimetry_y(results, width, height,keypoints_pair, pair)
                asimmetry_x = get_asimetry_x(results, width, height,keypoints_pair, pair)
                #if asimmetry_x  > 7 or asimmetry_y > 7:
                if asimmetry_x  > 7 + self.asimmetry_x0 or asimmetry_y > 7 + self.asimmetry_y0:
                    cv2.circle(frame,(xl, yl), 5, (0, 0, 255), -1)
                    cv2.circle(frame,(xr, yr), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(frame,(xl, yl), 5, (0, 255, 0), -1)
                    cv2.circle(frame,(xr, yr), 5, (0, 255, 0), -1)

                if pair == keypoints_options:
                    x_graph.append(asimmetry_x)
                    y_graph.append(asimmetry_y)
        
        return frame, x_graph, y_graph


