import altair as alt
import pandas as pd
from const.const import OUTPUT_VIDEO, OUTPUT_CSV

def sidebar_format(st):

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

def video_options(st):

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])

    return video_file_buffer

def mp_detection_parameters(st, body):

    st.sidebar.markdown('---')
    st.sidebar.subheader('Body Detection Parameters')
    keypoints_options = st.sidebar.selectbox('Graph keypoint position', [key for key, _ in body.items()])
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    model = st.sidebar.slider('Model Complexity', min_value = 0,max_value = 2,value = 1)

    return keypoints_options, detection_confidence, tracking_confidence, model

def export_options(st):
    st.subheader('Export options')
    with open(OUTPUT_VIDEO, 'rb') as fvideo:
        st.download_button('Video as MP4', fvideo, file_name=OUTPUT_VIDEO)

    with open(OUTPUT_CSV) as fcsv:
        st.download_button('Data as CSV', fcsv)


def hgh_detection_parameters(st):

    st.sidebar.markdown('---')
    st.sidebar.subheader('Bar Detection Parameters')
    minDist = st.sidebar.slider('Min Distance', min_value =0,max_value = 1000,value = 20)
    param1 = st.sidebar.slider('Parameter 1', min_value =0,max_value = 1000,value = 50)
    param2 = st.sidebar.slider('Parameter 2', min_value =0,max_value = 1000,value = 48)
    minRadius = st.sidebar.slider('Min Radius', min_value =0,max_value = 1000,value = 51)
    maxRadius = st.sidebar.slider('Max Radius', min_value =0,max_value = 1000,value = 200 )
    
    return maxRadius, minRadius, param2, param1, minDist


def plot_graph(axis, keypoints_options, st):

    st.subheader(f"{keypoints_options.replace('_',' ')} keypoint")
    #stgraphx.line_chart({"X": x_graph})
    d = {'Frames': list(range(0,len(axis))), 'Pixels': axis}
    df = pd.DataFrame(data=d)
    fig_rec = alt.Chart(df).mark_line().encode(
    alt.X("Frames"),
    alt.Y("Pixels")
    ).properties(
        width=700,
        height=500
    )
    st.altair_chart(fig_rec, use_container_width=True)

