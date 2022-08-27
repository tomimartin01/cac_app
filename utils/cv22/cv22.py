import cv2
from utils.misc.misc import  validate_video_format, check_multiple_detection
from const.const import OUTPUT_VIDEO

import threading
import queue
from streamlit.script_run_context import add_script_run_ctx

read_frames_buffer = queue.Queue(50)
process_frames_buffer = queue.Queue(1000)


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

def get_asimetry_y(results, width, height,keypoints_pair,pair):

    _, ylh = get_position_pair_xy(results, width, height, pair, keypoints_pair, 'l')
    _, yrh = get_position_pair_xy(results, width, height, pair, keypoints_pair, 'r')
    return abs(yrh - ylh)

def get_asimetry_x(results, width, height,keypoints_pair, pair):
    center_x0 = get_center_x(results, width, height,keypoints_pair)
    xlh, _ = get_position_pair_xy(results, width, height, pair, keypoints_pair, 'l')
    xrh, _ = get_position_pair_xy(results, width, height, pair, keypoints_pair, 'r')
    distl, distr = abs(xlh - center_x0), abs(xrh - center_x0)

    return abs(distl - distr)

def get_center_x(results, width, height,keypoints_pair):

    xls, _ = get_position_pair_xy(results, width, height, 'SHOULDERS', keypoints_pair, 'l')
    xrs, _ = get_position_pair_xy(results, width, height, 'SHOULDERS', keypoints_pair, 'r')
    return round((abs(xls - xrs)/2) + xrs)

def get_position_pair_xy(results, width, height, pair, keypoints_pair, side):

    if side == 'l':
        return (int(results.pose_landmarks.landmark[int(keypoints_pair[pair][0])].x * width), int(results.pose_landmarks.landmark[int(keypoints_pair[pair][0])].y * height))
    return (int(results.pose_landmarks.landmark[int(keypoints_pair[pair][1])].x * width), int(results.pose_landmarks.landmark[int(keypoints_pair[pair][1])].y * height))

def get_position_xy(results, width, height, part, keypoints):

    return (int(results.pose_landmarks.landmark[int(keypoints[part])].x * width), int(results.pose_landmarks.landmark[int(keypoints[part])].y * height))

def read_frames(cap,width,height):

    global finished, read_frames_buffer
    while not finished:
        ret, frame = cap.read()
        
        if not ret:
            finished = True
            cap.release()
            break
       
        try:
            frame = cv2.resize(frame, (width,height), interpolation = cv2.INTER_AREA)
            read_frames_buffer.put(frame, timeout=1)
        except queue.Full:
            pass

def write_frames( width, height,fps_input, progress_bar, total_frames,ph_graphx ):

    global finished, process_frames_buffer
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter(OUTPUT_VIDEO, codec, fps_input, (width, height))
    i=0
    progress_bar = ph_graphx.progress(0)
    while True:
        try:
            progress_bar.progress(round((i*100)/total_frames))
            frame = process_frames_buffer.get(timeout=1)
            out.write(frame)
            i += 1
        except queue.Empty:
            if finished:
                out.release()
                break

def processing_bar_frames(bar, total_frames, width, height, mp, keypoints_options):

    global finished, process_frames_buffer, read_frames_buffer
    global x_body, y_body, x_bar, y_bar, count_multiple_detection, multiple_detection
    
    
    x_body_input, y_body_input, x_bar_input, y_bar_input = [], [], [], []
    count_multiple_detection = 0
    multiple_detection = False
    count_frames= 0

    while True:
        
        count_frames+=1
        try:
            
            frame = read_frames_buffer.get(timeout=1)
            frame, x_bar, y_bar, multiple_detection = bar.process(count_frames, total_frames, width, height, x_bar_input, y_bar_input, frame)
            frame, x_body, y_body = mp.process(keypoints_options, count_frames, total_frames, width, height, x_body_input, y_body_input, frame)
            process_frames_buffer.put(frame, timeout=1)
        except queue.Empty:
            if finished:
                break
        except queue.Full:
            pass

def processing_body_frames(total_frames, width, height, mp, keypoints_options):

    global finished, process_frames_buffer, read_frames_buffer
    global x_body, y_body

    
    x_array, y_array = [], []
    count_frames= 0

    while True:
        
        count_frames+=1
        try:
            
            frame = read_frames_buffer.get(timeout=1)
            frame, x_body, y_body = mp.process(keypoints_options, count_frames, total_frames, width, height, x_array, y_array, frame)
            process_frames_buffer.put(frame, timeout=1)
        except queue.Empty:
            if finished:
                break
        except queue.Full:
            pass

def processing_sim_frames(total_frames, width, height, sim, keypoints_options):

    global finished, process_frames_buffer, read_frames_buffer
    global x_body, y_body

    count_frames= 0
    x_array, y_array = [], []
    while True:
        
        count_frames+=1
        try:
            
            frame = read_frames_buffer.get(timeout=1)
            frame, x_body, y_body = sim.process(keypoints_options, count_frames, total_frames, width, height,frame, x_array, y_array)
            process_frames_buffer.put(frame, timeout=1)
        except queue.Empty:
            if finished:
                break
        except queue.Full:
            pass

def analysis_mp(keypoints_options,mp, ph_graphx):
    
    global x_body, y_body, finished
    cap = cv2.VideoCapture(mp.video)
    finished = False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = ph_graphx.progress(0)

    tReadingFrames = threading.Thread(target=read_frames, args=(cap,width, height))
    add_script_run_ctx(tReadingFrames)
    tReadingFrames.start()

    tProcessingFrames = threading.Thread(target=processing_body_frames, args=(total_frames, width, height, mp, keypoints_options))
    add_script_run_ctx(tProcessingFrames)
    tProcessingFrames.start()

    tWritingFrames = threading.Thread(target=write_frames, args=(width,height,fps_input, progress_bar,total_frames,ph_graphx ))
    add_script_run_ctx(tWritingFrames)
    tWritingFrames.start()

    tWritingFrames.join()
    tProcessingFrames.join()
    tReadingFrames.join()
    
    return x_body, y_body

def analysis_sim(keypoints_options, sim, ph_graphx):

    
    global x_body, y_body, x_bar, y_bar, finished
    cap = cv2.VideoCapture(sim.video)
    finished = False
    width = 640
    height = 480
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = ph_graphx.progress(0)

    tReadingFrames = threading.Thread(target=read_frames, args=(cap,width,height))
    add_script_run_ctx(tReadingFrames)
    tReadingFrames.start()

    tProcessingFrames = threading.Thread(target=processing_sim_frames, args=(total_frames, width, height, sim, keypoints_options))

    add_script_run_ctx(tProcessingFrames)
    tProcessingFrames.start()

    tWritingFrames = threading.Thread(target=write_frames, args=(width,height,fps_input, progress_bar,total_frames,ph_graphx ))
    add_script_run_ctx(tWritingFrames)
    tWritingFrames.start()

    tWritingFrames.join()
    tProcessingFrames.join()
    tReadingFrames.join()
    
    return x_body, y_body

def analysis_bar(keypoints_options, mp, bar, ph_graphx):
    
    global x_body, y_body, x_bar, y_bar, count_multiple_detection, multiple_detection, finished
    cap = cv2.VideoCapture(bar.video)

    finished = False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = ph_graphx.progress(0)

    tReadingFrames = threading.Thread(target=read_frames, args=(cap,width, height))
    add_script_run_ctx(tReadingFrames)
    tReadingFrames.start()

    tProcessingFrames = threading.Thread(target=processing_bar_frames, args=( bar,total_frames, width, height, mp, keypoints_options))
    add_script_run_ctx(tProcessingFrames)
    tProcessingFrames.start()

    tWritingFrames = threading.Thread(target=write_frames, args=(width,height,fps_input, progress_bar,total_frames,ph_graphx ))
    add_script_run_ctx(tWritingFrames)
    tWritingFrames.start()

    tWritingFrames.join()
    tProcessingFrames.join()
    tReadingFrames.join()
    
    return x_body, y_body, x_bar, y_bar, multiple_detection




