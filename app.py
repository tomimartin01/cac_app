import streamlit as st
import mediapipe as mp
import tempfile
from os.path import exists


from const.const import keypoints, keypoints_pair, OUTPUT_VIDEO, OUTPUT_CSV
from analysis.body import Body
from analysis.asimmetry import Asimmetry
from analysis.bar import Bar
from utils.misc.misc import mp_validate_detection, hgh_validate_detection
from utils.csvv.csvv import write_csv_bar,write_csv_body
from utils.st.stt import sidebar_format, video_options, mp_detection_parameters, export_options, hgh_detection_parameters, plot_graph
from utils.cv22.cv22 import analysis_mp, analysis_sim, analysis_bar

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

ph_text = st.empty()
ph_video = st.empty()
ph_graphx = st.empty()
ph_graphy = st.empty()
ph_export = st.empty()

sidebar_format(st)
st.sidebar.title('App Options')
app_mode = st.sidebar.selectbox('Choose App mode ',
                                ['Home', 'Body Analysis', 'Asimmetry Analysis', 'Bar Analysis'])

if app_mode =='Home':
    
    ph_graphx = st.empty()
    ph_graphy = st.empty()
    ph_export = st.empty()
    with ph_text.container():
        st.title('Crossfit Assistan Couch App')

        st.markdown ('''CrossFit Coach Assistant (CCA) is a software tool that facilitates the coach's analysis of strength \
                    and gymnastic exercises and the trainee's correction by displaying the results on a User Interface.
                    ''') 
        st.markdown ('''To achieve these goals, CCA offers 3 different types of analysis using Machine Learning and computer\
                    vision techniques:
                    ''')
        st.markdown('''
                    - Body analysis 
                    - Asimmetry analysis 
                    - Bar analysis
                ''')

    ph_video = st.empty()
    sidebar_format(st)

elif app_mode =='Body Analysis':
    
    ph_graphx = st.empty()
    ph_graphy = st.empty()
    ph_export = st.empty()
    with ph_text.container():
        st.subheader('Body analysis')
        st.markdown ('''With this analysis, you can see the keypoints postition in pixels of the body. \
                        You have to upload a video.''')

    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_format(st)
    ## set Body analysis options and parameters
    video_file_buffer = video_options(st)
    keypoints_options, detection_confidence, tracking_confidence, model = mp_detection_parameters(st, keypoints)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    
    ## Only process if there is a video loaded
    if video_file_buffer:

        tfflie.write(video_file_buffer.read())
        ## create an instance of analyzer and execute frontal analysis
        body = Body(tfflie.name, detection_confidence, tracking_confidence, model)
        x_body, y_body = analysis_mp(keypoints_options, body, ph_graphx)
        if (not mp_validate_detection(x_body, y_body)):
            with ph_video.container():
                st.error('Inaccurate Detection. Please change Detection Parametrs.')
                ph_graphx.empty()
        else:
            with ph_graphx.expander('X-axis Graph'):
                with st.container():
                    plot_graph(x_body, keypoints_options, st)
            with ph_graphy.expander('Y-axis Graph'):
                with st.container():
                    plot_graph(y_body, keypoints_options, st)
            with ph_video.expander("Video"):
                with open(OUTPUT_VIDEO, 'rb') as fvideo:
                    video_bytes = fvideo.read()
                    st.video(video_bytes)

            with ph_export.expander("Export options"):
                write_csv_body(x_body, y_body)
                if exists(OUTPUT_VIDEO) and exists(OUTPUT_CSV):
                    export_options(st)
    else:
         with ph_video.container():
                st.warning('No loaded video. \n Please load a video file.')

elif app_mode =='Asimmetry Analysis':

    ph_graphx = st.empty()
    ph_graphy = st.empty()
    ph_export = st.empty()
    with ph_text.container():
        st.subheader('Asimmetry Analysis')
        st.markdown ('''With this analysis, you can see the asimmetry in pixels between the right keypoints and the left keypoints \
                        of the body. You have to upload a frontal plane video.''')

    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_format(st)
    ## set Body analysis options and parameters
    video_file_buffer = video_options(st)
    keypoints_options, detection_confidence, tracking_confidence, model = mp_detection_parameters(st, keypoints_pair)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    
    ## Only process if there is a video loaded
    if video_file_buffer:

        tfflie.write(video_file_buffer.read())
        ## create an instance of analyzer and execute frontal analysis
        sim = Asimmetry(tfflie.name, detection_confidence, tracking_confidence, model)
        x_body, y_body = analysis_sim(keypoints_options, sim, ph_graphx)
        if (not mp_validate_detection(x_body, y_body)):
            with ph_video.container():
                st.error('Inaccurate Detection. Please change Detection Parametrs.')
                ph_graphx.empty()
        else:
            with ph_graphx.expander('Asimmetry in X-axis Graph'):
                with st.container():
                    plot_graph(x_body, keypoints_options, st)
            with ph_graphy.expander('Asimmetry in Y-axis Graph'):
                with st.container():
                    plot_graph(y_body, keypoints_options, st)
            with ph_video.expander("Video"):
                with open(OUTPUT_VIDEO, 'rb') as fvideo:
                    video_bytes = fvideo.read()
                    st.video(video_bytes)
            with ph_export.expander("Export options"):
                write_csv_body(x_body, y_body)
                if exists(OUTPUT_VIDEO) and exists(OUTPUT_CSV):
                    export_options(st)
    else:
         with ph_video.container():
                st.warning('No loaded video. \n Please load a video file.')

elif app_mode =='Bar Analysis':

    ph_graphx = st.empty()
    ph_graphy = st.empty()
    ph_export = st.empty()
    with ph_text.container():
        st.subheader('Bar analysis')
        st.markdown ('''With this analysis, you can see the bar postition and the keypoints postition of the body in pixels . \
                        You have to upload a side plane video.''')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_format(st)
    video_file_buffer = video_options(st)
    keypoints_options, detection_confidence, tracking_confidence, model = mp_detection_parameters(st, keypoints)
    maxRadius, minRadius, param2, param1, minDist = hgh_detection_parameters(st)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer:
        
        tfflie.write(video_file_buffer.read())
        body = Body(tfflie.name, detection_confidence, tracking_confidence, model)
        bar = Bar(tfflie.name, minDist, param1, param2, minRadius, maxRadius)
        x_body, y_body, x_bar, y_bar, multiple_detection = analysis_bar(keypoints_options,body,bar, ph_graphx)
        # if (not mp_validate_detection(x_body, y_body)):
        #     with ph_video.container():
        #         st.error('Inaccurate Detection. Please change Detection Parametrs.')
        #         ph_graphx.empty()
        # else:
        with ph_graphx.expander('X-axis Graph'):
            with st.container():
                plot_graph(x_body, keypoints_options, st)
        with ph_graphy.expander('Y-axis Graph'):
            with st.container():
                plot_graph(y_body, keypoints_options, st)
        with ph_video.expander("Video"):
            with open(OUTPUT_VIDEO, 'rb') as fvideo:
                video_bytes = fvideo.read()
                st.video(video_bytes)
        with ph_export.expander("Export options"):
            write_csv_bar(x_body, y_body, x_bar, y_bar)
            if exists(OUTPUT_VIDEO) and exists(OUTPUT_CSV):
                export_options(st)
    else:
        with ph_video.container():
            st.warning('No loaded video. \n Please load a video file.')

