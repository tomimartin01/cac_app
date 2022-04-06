import streamlit as st
import mediapipe as mp
import tempfile
from os.path import exists

from const.const import keypoints, keypoints_pair, OUTPUT_VIDEO, OUTPUT_CSV
from analysis.body import Body
from analysis.asimmetry import Asimmetry
from analysis.bar import Bar
from utils.misc.misc import mp_validate_detection, hgh_validate_detection
from utils.csvv.csvv import write_csv
from utils.st.stt import create_components, sidebar_format, video_options, hide_components, mp_detection_parameters, export_options, hgh_detection_parameters, plot_graph, write_video
from utils.cv22.cv22 import analysis

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


sidebar_format(st)

st.sidebar.subheader('App Options')


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['Home', 'Body Analysis', 'Asimmetry Analysis', 'Bar Analysis'])

if app_mode =='Home':
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

    st.subheader('Body analysis')
    st.markdown ('''With this analysis, you can see the keypoints postition in pixels of the body. \
                    You have to upload a video.''')

    st.subheader('Asimmetry Analysis')
    st.markdown ('''With this analysis, you can see the asimmetry in pixels between the right keypoints and the left keypoints \
                    of the body. You have to upload a frontal plane video.''')

    st.subheader('Bar analysis')
    st.markdown ('''With this analysis, you can see the bar postition and the keypoints postition of the body in pixels . \
                    You have to upload a side plane video.''')

    
    sidebar_format(st)

elif app_mode =='Body Analysis':

    st.title('Body Analysis')
    st.markdown('---')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_format(st)
    ## Create dinamyc components, it allows to hide them 
    stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy = create_components(st)
    ## set Body analysis options and parameters
    video_file_buffer, is_writting = video_options(st)
    keypoints_options, detection_confidence, tracking_confidence, model = mp_detection_parameters(st, keypoints)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    
    ## Only process if there is a video loaded
    if video_file_buffer:
        
        ## Hide componets, they are shown after the processing
        hide_components(stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy)

        tfflie.write(video_file_buffer.read())
        ## create an instance of analyzer and execute frontal analysis
        body = Body(tfflie.name, detection_confidence, tracking_confidence, model, is_writting)
        x_body, y_body, x_bar, y_bar, multiple_detection = analysis(stframe, keypoints_options, body, None, None)
        
        if (not mp_validate_detection(x_body, y_body)):
            ststatus.error('Inaccurate Detection. Please change Detection Parametrs.')
        else:

            plot_graph('X-axis Graph', x_body, keypoints_options, stgraphtitle, stgraphxlabel, stgraphx)
            plot_graph('Y-axis Graph', y_body, keypoints_options, stgraphtitle, stgraphylabel, stgraphy)
            print(exists(OUTPUT_VIDEO), exists(OUTPUT_CSV))
            if is_writting:
                write_csv(x_body, y_body, x_bar, y_bar)
                write_video(stframe)
                if exists(OUTPUT_VIDEO) and exists(OUTPUT_CSV):
                    export_options(st)

    else:
        ststatus.warning('No loaded video. \n Please load a video file.')

elif app_mode =='Asimmetry Analysis':

    st.title('Asimmetry Analysis')
    st.markdown('---')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_format(st)
    ## Create dinamyc components, it allows to hide them 
    stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy = create_components(st)
    ## set Body analysis options and parameters
    video_file_buffer, is_writting = video_options(st)
    keypoints_options, detection_confidence, tracking_confidence, model = mp_detection_parameters(st, keypoints_pair)
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer:

        hide_components(stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy)
        tfflie.write(video_file_buffer.read())
        asimmetry = Asimmetry(tfflie.name, detection_confidence, tracking_confidence, model, is_writting)
        x_body, y_body, x_bar, y_bar, multiple_detection = analysis(stframe, keypoints_options, None, asimmetry, None)
        
        if (not mp_validate_detection(x_body, y_body)):
            ststatus.error('Inaccurate Detection. Please change Detection Parametrs.')
        else:
            plot_graph('X-axis Asimmetry Graph', x_body, keypoints_options, stgraphtitle, stgraphxlabel, stgraphx)
            plot_graph('Y-axis Asimmetry Graph', y_body, keypoints_options, stgraphtitle, stgraphylabel, stgraphy)

            if is_writting:
                write_csv(x_body, y_body, x_bar, y_bar)
                write_video(stframe)
                if exists(OUTPUT_VIDEO) and exists(OUTPUT_CSV):
                    export_options(st)
    else:
        ststatus.warning('No loaded video. \n Please load a video file.')

elif app_mode =='Bar Analysis':

    st.title('Bar Analysis')
    st.markdown('---')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_format(st)
    stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy = create_components(st)
    video_file_buffer, is_writting = video_options(st)
    keypoints_options, detection_confidence, tracking_confidence, model = mp_detection_parameters(st, keypoints_pair)
    maxRadius, minRadius, param2, param1, minDist = hgh_detection_parameters(st)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer:
        
        tfflie.write(video_file_buffer.read())
        Body = Body(tfflie.name, detection_confidence, tracking_confidence, model, is_writting)
        bar = Bar(tfflie.name, minDist, param1, param2, minRadius, maxRadius, is_writting)
        x_body, y_body, x_bar, y_bar, multiple_detection = analysis(stframe, None, Body, None, bar)
        if (not hgh_validate_detection(x_body, y_body, multiple_detection)):
            ststatus.error('Inaccurate Detection. Please change Detection Parametrs.')
        else:
            plot_graph('X-axis Graph', x_body, keypoints_options, stgraphtitle, stgraphxlabel, stgraphx)
            plot_graph('Y-axis Graph', y_body, keypoints_options, stgraphtitle, stgraphylabel, stgraphy)
            if is_writting:
                write_csv(x_body, y_body, x_bar, y_bar)
                write_video(stframe)
                if exists(OUTPUT_VIDEO) and exists(OUTPUT_CSV):
                    export_options(st)
    else:
        ststatus.warning('No loaded video. \n Please load a video file.')
        

