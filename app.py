import streamlit as st
import mediapipe as mp
import cv2
import tempfile

from const.const import keypoints, keypoints_pair
from analysis.simple import Simple
from analysis.simmetry import Simmetry
from analysis.bar import Bar
from utils.misc.misc import mp_validate_detection, hgh_validate_detection
from utils.csv.csvv import write_csv
from utils.st.stt import create_components, sidebar_format, video_options, hide_components, mp_detection_parameters, export_options, hgh_detection_parameters, plot_graph, write_video
from utils.cv22.cv22 import analysis

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'


sidebar_format(st)

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
    st.markdown('---')
    st.markdown('Crossfit Assistan Couch (CAC) App  herramienta de software para el análisis de \
                algunos de los ejercicios de halterofilia y gimnasia de la actividad deportiva CrossFit, basada\
                en técnicas de inteligencia artificial y visión por computadora. El entrenador podra hacer foco en \
                tres aspectos diferentes para poder corregir al atleta: Trajectoria de puntos de referencia del \
                cuerpo del atleta, simetria y trajectoria de la barra olimpica.')
       
    st.markdown('En esta aplicacion, estamos usando **MediaPipe** y **OPenCV** para el seguimiento de las poses del cuerpo humano. **StreamLit** para crear la interfaz grafica de usuario.')
    
    sidebar_format(st)

elif app_mode =='Simple Body Analysis':

    st.title('Simple Body Analysis')
    st.markdown('---')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_format(st)
    ## Create dinamyc components, it allows to hide them 
    stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy = create_components(st)
    ## set simple analysis options and parameters
    video_file_buffer, is_writting = video_options(st)
    keypoints_options, detection_confidence, tracking_confidence, model = mp_detection_parameters(st, keypoints)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    export_options(st)
    
    ## Only process if there is a video loaded
    if video_file_buffer:
        
        ## Hide componets, they are shown after the processing
        hide_components(stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy)

        tfflie.write(video_file_buffer.read())
        ## create an instance of analyzer and execute frontal analysis
        simple = Simple(tfflie.name, detection_confidence, tracking_confidence, model, is_writting)
        x_body, y_body, x_bar, y_bar, multiple_detection = analysis(stframe, keypoints_options, simple, None, None)
        
        if (not mp_validate_detection(x_body, y_body)):
            ststatus.error('Inaccurate Detection. Please change Detection Parametrs.')
        else:

            plot_graph('X-axis Graph', x_body, keypoints_options, stgraphtitle, stgraphxlabel, stgraphx)
            plot_graph('Y-axis Graph', y_body, keypoints_options, stgraphtitle, stgraphylabel, stgraphy)

            if is_writting:
                write_csv(x_body, y_body, x_bar, y_bar)
                write_video(stframe)

    else:
        ststatus.warning('No loaded video. \n Please load a video file.')

elif app_mode =='Frontal Body Analysis':

    st.title('Frontal Body Analysis')
    st.markdown('---')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_format(st)
    ## Create dinamyc components, it allows to hide them 
    stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy = create_components(st)
    ## set simple analysis options and parameters
    video_file_buffer, is_writting = video_options(st)
    keypoints_options, detection_confidence, tracking_confidence, model = mp_detection_parameters(st, keypoints_pair)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    export_options(st)

    if video_file_buffer:

        hide_components(stframe, ststatus, stgraphtitle, stgraphxlabel, stgraphx, stgraphylabel, stgraphy)
        tfflie.write(video_file_buffer.read())
        simmetry = Simmetry(tfflie.name, detection_confidence, tracking_confidence, model, is_writting)
        x_body, y_body, x_bar, y_bar, multiple_detection = analysis(stframe, keypoints_options, None, simmetry, None)
        
        if (not mp_validate_detection(x_body, y_body)):
            ststatus.error('Inaccurate Detection. Please change Detection Parametrs.')
        else:
            plot_graph('X-axis Graph', x_body, keypoints_options, stgraphtitle, stgraphxlabel, stgraphx)
            plot_graph('Y-axis Graph', y_body, keypoints_options, stgraphtitle, stgraphylabel, stgraphy)

            if is_writting:
                write_csv(x_body, y_body, x_bar, y_bar)
                write_video(stframe)
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
    export_options(st)

    if video_file_buffer:
        
        tfflie.write(video_file_buffer.read())
        simple = Simple(tfflie.name, detection_confidence, tracking_confidence, model, is_writting)
        bar = Bar(tfflie.name, minDist, param1, param2, minRadius, maxRadius, is_writting)
        x_body, y_body, x_bar, y_bar, multiple_detection = analysis(stframe, None, simple, None, bar)
        if (not hgh_validate_detection(x_body, y_body, multiple_detection)):
            ststatus.error('Inaccurate Detection. Please change Detection Parametrs.')
        else:
            plot_graph('X-axis Graph', x_body, keypoints_options, stgraphtitle, stgraphxlabel, stgraphx)
            plot_graph('Y-axis Graph', y_body, keypoints_options, stgraphtitle, stgraphylabel, stgraphy)
            if is_writting:
                write_csv(x_body, y_body, x_bar, y_bar)
                write_video(stframe)
    else:
        ststatus.warning('No loaded video. \n Please load a video file.')
        

