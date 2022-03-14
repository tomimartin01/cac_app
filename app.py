import streamlit as st
import mediapipe as mp
import cv2
import tempfile
from analyzer import Analyzer
from hough import Hough
from const import keypoints, keypoints_pair

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'



st.title('CAC App')

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

st.sidebar.title('CAC App')
st.sidebar.subheader('Parameters')

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
                                ['Home', 'Simple', 'Frontal', 'Lateral'])

if app_mode =='Home':
    st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
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
    st.video('https://www.youtube.com/watch?v=FMaNNXgB_5c&ab_channel=AugmentedStartups')

    st.markdown('''
          # About Me \n 
            Hey this is ** Ritesh Kanjee ** from **Augmented Startups**. \n
           
            If you are interested in building more Computer Vision apps like this one then visit the **Vision Store** at
            www.augmentedstartups.info/visionstore \n
            
            Also check us out on Social Media
            - [YouTube](https://augmentedstartups.info/YouTube)
            - [LinkedIn](https://augmentedstartups.info/LinkedIn)
            - [Facebook](https://augmentedstartups.info/Facebook)
            - [Discord](https://augmentedstartups.info/Discord)
        
            If you are feeling generous you can buy me a **cup of  coffee ** from [HERE](https://augmentedstartups.info/ByMeACoffee)
             
            ''')
elif app_mode =='Simple':

    st.set_option('deprecation.showfileUploaderEncoding', False)
    record = st.sidebar.checkbox("Record Video")

    ## if record:
    ##    st.checkbox("Recording", value=True)
    
    ## if start:
    ##    st.checkbox("Starting", value=True)

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

    # st.markdown(' ## Output')

    stframe = st.empty()
    
    st.sidebar.markdown('---')
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    st.sidebar.markdown('---')
    keypoints_options = st.sidebar.multiselect(
    'Graph keypoint position',
    [key for key, _ in keypoints.items()])
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    model = st.sidebar.slider('Model Complexity', min_value = 0,max_value = 2,value = 1)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    
    if video_file_buffer:

        tfflie.write(video_file_buffer.read())
        cam_analyzer = Analyzer(tfflie.name, detection_confidence, tracking_confidence, model, record)
        x_graph, y_graph = cam_analyzer.simple_analysis(st, stframe, keypoints_options)
        st.line_chart({"X": x_graph})
        st.line_chart({"Y": y_graph})

elif app_mode =='Frontal':

    st.set_option('deprecation.showfileUploaderEncoding', False)
    record = st.sidebar.checkbox("Record Video")


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
    st.sidebar.markdown('---')
    keypoints_options = st.sidebar.multiselect(
    'Graph keypoint position',
    [key for key, _ in keypoints_pair.items()])
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    model = st.sidebar.slider('Model Complexity', min_value = 0,max_value = 2,value = 1)
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer:
        tfflie.write(video_file_buffer.read())
        cam_analyzer = Analyzer(tfflie.name, detection_confidence, tracking_confidence, model, record)
        asimmetry_x_graph,asimmetry_y_graph = cam_analyzer.frontal_analysis(st, stframe, keypoints_options)
        st.line_chart({"Asimetry X": asimmetry_x_graph})
        st.line_chart({"Asimetry Y": asimmetry_y_graph})

elif app_mode =='Lateral':

    st.set_option('deprecation.showfileUploaderEncoding', False)
    record = st.sidebar.checkbox("Record Video")

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
            
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    st.sidebar.markdown('---')
    minDist = st.sidebar.slider('Min Distancia', min_value =0,max_value = 1000,value = 100)
    param1 = st.sidebar.slider('Parameter 1', min_value =0,max_value = 1000,value = 100)
    param2 = st.sidebar.slider('Parameter 2', min_value =0,max_value = 1000,value = 100)
    minRadius = st.sidebar.slider('Min Radius', min_value =0,max_value = 1000,value = 100)
    maxRadius = st.sidebar.slider('Min Distance', min_value =0,max_value = 1000,value = 100)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    
    if video_file_buffer:
        tfflie.write(video_file_buffer.read())
        cam_analyzer = Hough(tfflie.name, minDist, param1, param2, minRadius, maxRadius, record)
        cam_analyzer.lateral_analysis(st, stframe)
        

