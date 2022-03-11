import cv2
import time
import mediapipe as mp
from utils import image_resize, get_position_xy, get_asimetry_y, get_asimetry_x
from const import keypoints, keypoints_pair

class Analyzer:

  def __init__(self, video, detection_confidence, tracking_confidence, model, start, record):
    self.video = video
    self.detection_confidence = detection_confidence
    self.tracking_confidence = tracking_confidence
    self.model = model
    self.start = start
    self.record = record
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_pose = mp.solutions.pose

    
  def simple_analysis(self, st, stframe):

    cap = cv2.VideoCapture(self.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    fps = 0
    count_frames= 0
    drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Count Frames**")
        kpi3_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Size**")
        kpi2_text = st.markdown("0 x 0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with self.mp_pose.Pose(
    min_detection_confidence= self.detection_confidence,
    min_tracking_confidence= self.tracking_confidence,
    model_complexity = self.model)as pose:
        prevTime = 0

        while cap.isOpened() and self.start:
            count_frames+=1
            ret, frame = cap.read()
            if not ret:
                continue

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

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if self.record:
                #st.checkbox("Recording", value=True)
                out.write(frame)
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{count_frames}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width} x {height}</h1>", unsafe_allow_html=True)
            

            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,channels = 'BGR',use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    cap.release()
    out. release()

  def frontal_analysis(self, st, stframe):

    cap = cv2.VideoCapture(self.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    fps = 0
    count_frames= 0
    drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Count Frames**")
        kpi3_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Size**")
        kpi2_text = st.markdown("0 x 0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with self.mp_pose.Pose(
    min_detection_confidence= self.detection_confidence,
    min_tracking_confidence= self.tracking_confidence,
    model_complexity = self.model)as pose:
        prevTime = 0

        while cap.isOpened() and self.start:
            count_frames+=1
            ret, frame = cap.read()
            if not ret:
                continue

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

                    xl, yl = get_position_xy(results, width, height, pair, keypoints_pair, 'l')
                    xr, yr = get_position_xy(results, width, height, pair, keypoints_pair, 'r')
                    
                    asimmetry_y = get_asimetry_y(results, width, height,keypoints_pair, pair)
                    asimmetry_x = get_asimetry_x(results, width, height,keypoints_pair, pair)
                    if asimmetry_x  > 7 or asimmetry_y > 7:
                    #if asimmetry_x  > 7 + asimmetry_x0 or asimmetry_y > 7 + asimmetry_y0:
                        cv2.circle(frame,(xl, yl), 5, (0, 0, 255), -1)
                        cv2.circle(frame,(xr, yr), 5, (0, 0, 255), -1)
                    else:
                        cv2.circle(frame,(xl, yl), 5, (0, 255, 0), -1)
                        cv2.circle(frame,(xr, yr), 5, (0, 255, 0), -1)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if self.record:
                #st.checkbox("Recording", value=True)
                out.write(frame)
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{count_frames}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width} x {height}</h1>", unsafe_allow_html=True)
            

            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,channels = 'BGR',use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    cap.release()
    out. release()

    