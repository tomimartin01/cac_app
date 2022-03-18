import streamlit as st
import mediapipe as mp
import cv2
import tempfile
from analyzer import Analyzer
from hough import Hough
from const import keypoints, keypoints_pair
import altair as alt
import pandas as pd
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'



st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.subheader('App Options')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['Home', 'Simple Body Analysis', 'Frontal Body Analysis', 'Bar Analysis'])

if app_mode =='Home':
    st.title('Crossfit Assistan Couch App')
    st.markdown('Crossfit Assistan Couch (CAC) App  herramienta de software para el análisis de \
                algunos de los ejercicios de halterofilia y gimnasia de la actividad deportiva CrossFit, basada\
                en técnicas de inteligencia artificial y visión por computadora. El entrenador podra hacer foco en \
                tres aspectos diferentes para poder corregir al atleta: Trajectoria de puntos de referencia del \
                cuerpo del atleta, simetria y trajectoria de la barra olimpica.')
       
    st.markdown('En esta aplicacion, estamos usando **MediaPipe** y **OPenCV** para el seguimiento de las poses del cuerpo humano. **StreamLit** para crear la interfaz grafica de usuario.')
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

elif app_mode =='Simple Body Analysis':

    st.title('Simple Body Analysis')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,)

    ## Create dinamyc components, it allows to hide them 
    stframe = st.empty()
    stwait = st.empty()
    stgraphtitle= st.empty()
    stgraphxlabel= st.empty()
    stgraphx = st.empty()
    stgraphylabel= st.empty()
    stgraphy = st.empty()

    ## set simple analysis options and parameters
    st.sidebar.markdown('---')
    st.sidebar.markdown('Video options') 
    keypoints_options = st.sidebar.selectbox(
    'Graph keypoint position',
    [key for key, _ in keypoints.items()])
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    record = st.sidebar.checkbox("Record Video")
    download = st.download_button('Download Zip', f, file_name='archive.zip')

    st.sidebar.markdown('---')
    st.sidebar.markdown('Detection Parameters')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    model = st.sidebar.slider('Model Complexity', min_value = 0,max_value = 2,value = 1)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    
    ## Only process if there is a video loaded
    if video_file_buffer:
        
        ## Hide componets, they are shown after the processing
        stwait.empty()
        stgraphtitle.empty()
        stgraphx.empty()
        stgraphxlabel.empty()
        stgraphylabel.empty()
        stgraphy.empty()

        tfflie.write(video_file_buffer.read())
        ## create an instance of analyzer and execute frontal analysis
        cam_analyzer = Analyzer(tfflie.name, detection_confidence, tracking_confidence, model, record)
        x_graph, y_graph = cam_analyzer.simple_analysis(st, stframe, keypoints_options)
        
        ## Remove image component 
        stframe.empty()
        stgraphtitle.subheader(f"{keypoints_options.replace('_',' ')} keypoint")

        stgraphxlabel.markdown('X-axis Graph')
        #stgraphx.line_chart({"X": x_graph})
        d = {'Frames': list(range(0,len(x_graph))), 'Pixels': x_graph}
        df = pd.DataFrame(data=d)
        fig_rec = alt.Chart(df).mark_line().encode(
        alt.X("Frames"),
        alt.Y("Pixels")
        ).properties(
            width=700,
            height=500
        )
        stgraphx.altair_chart(fig_rec)

        stgraphylabel.markdown('Y-axis Graph')
        #stgraphy.line_chart({"Y": y_graph})
        d = {'Frames': list(range(0,len(y_graph))), 'Pixels': y_graph}
        df = pd.DataFrame(data=d)
        fig_rec = alt.Chart(df).mark_line().encode(
        alt.X("Frames"),
        alt.Y("Pixels")
        ).properties(
            width=700,
            height=500
        )
        stgraphy.altair_chart(fig_rec)

    else:
        stwait.subheader('Please load a video file...')

elif app_mode =='Frontal Body Analysis':

    st.title('Frontal Body Analysis')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,)

    stframe = st.empty() 
    stwait = st.empty()
    stgraphtitle= st.empty()
    stgraphxlabel= st.empty()
    stgraphx = st.empty()
    stgraphylabel= st.empty()
    stgraphy = st.empty()
    
    st.sidebar.markdown('---')
    st.sidebar.markdown('Video options') 
    keypoints_options = st.sidebar.selectbox(
    'Graph keypoint position',
    [key for key, _ in keypoints_pair.items()])
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    record = st.sidebar.checkbox("Record Video")

    st.sidebar.markdown('---')
    st.sidebar.markdown('Detection Parameters')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    model = st.sidebar.slider('Model Complexity', min_value = 0,max_value = 2,value = 1)
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer:

        stwait.empty()
        stgraphtitle.empty()
        stgraphx.empty()
        stgraphxlabel.empty()
        stgraphylabel.empty()
        stgraphy.empty()

        tfflie.write(video_file_buffer.read())
        cam_analyzer = Analyzer(tfflie.name, detection_confidence, tracking_confidence, model, record)
        asimmetry_x_graph,asimmetry_y_graph = cam_analyzer.frontal_analysis(st, stframe, keypoints_options)
        stframe.empty()

        stgraphtitle.subheader(f"{keypoints_options.replace('_',' ')} keypoint")
        stgraphxlabel.markdown('Asimmetry X Graph')
        # stgraphx.line_chart({"Asimetry X": asimmetry_x_graph})
        d = {'Frames': list(range(0,len(asimmetry_x_graph))), 'Pixels': asimmetry_x_graph}
        df = pd.DataFrame(data=d)
        fig_rec = alt.Chart(df).mark_line().encode(
        alt.X("Frames"),
        alt.Y("Pixels")
        ).properties(
            width=700,
            height=500
        )
        stgraphx.altair_chart(fig_rec)

        stgraphylabel.markdown('Asimmetry Y Graph')
        # stgraphy.line_chart({"Asimmetry Y": asimmetry_y_graph})
        d = {'Frames': list(range(0,len(asimmetry_y_graph))), 'Pixels': asimmetry_y_graph}
        df = pd.DataFrame(data=d)
        fig_rec = alt.Chart(df).mark_line().encode(
        alt.X("Frames"),
        alt.Y("Pixels")
        ).properties(
            width=700,
            height=500
        )
        stgraphy.altair_chart(fig_rec)
        
    else:
        stwait.subheader('Please load a video file...')

elif app_mode =='Bar Analysis':

    st.title('Bar Analysis')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,)

    stframe = st.empty()
    stwait = st.empty()
    stgraphtitle= st.empty()
    stgraphlabel= st.empty()
    stgraph = st.empty()

    st.sidebar.markdown('---')  
    st.sidebar.markdown('Video options')     
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    record = st.sidebar.checkbox("Record Video")

    st.sidebar.markdown('---')
    st.sidebar.markdown('Detection Parameters')
    minDist = st.sidebar.slider('Min Distance', min_value =0,max_value = 1000,value = 20)
    param1 = st.sidebar.slider('Parameter 1', min_value =0,max_value = 1000,value = 50)
    param2 = st.sidebar.slider('Parameter 2', min_value =0,max_value = 1000,value = 48)
    minRadius = st.sidebar.slider('Min Radius', min_value =0,max_value = 1000,value = 51)
    maxRadius = st.sidebar.slider('Max Radius', min_value =0,max_value = 1000,value = 87)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    
    if video_file_buffer:
        
        stwait.empty()
        tfflie.write(video_file_buffer.read())
        cam_analyzer = Hough(tfflie.name, minDist, param1, param2, minRadius, maxRadius, record)
        x_graph, y_graph, multiple_detection, blank_frame = cam_analyzer.bar_analysis(st, stframe)

        ## Hide image component in post processing
        stframe.empty()
        stframe.image(blank_frame,channels = 'RGB',use_column_width='auto')

        ## GRaph xy
        #d = {'X': x_graph, 'Y': y_graph}
        # df = pd.DataFrame(data=d)
        # fig_rec = alt.Chart(df).mark_line().encode(
        # alt.X("X"),
        # alt.Y("Y")
        # ).properties(
        #     width=900,
        #     height=500
        # )
        # st.altair_chart(fig_rec)

      
    else:
        stwait.subheader('Please load a video file...')
        

