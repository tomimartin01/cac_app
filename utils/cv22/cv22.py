import cv2
from utils.misc.misc import  validate_video_format, check_multiple_detection
from const.const import OUTPUT_VIDEO

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

def analysis( keypoints_options, mp, sim, bar, ph_graphx):
    
    if mp:
        cap = cv2.VideoCapture(mp.video)
    elif sim:
        cap = cv2.VideoCapture(sim.video)
    elif bar:
        cap = cv2.VideoCapture(bar.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not validate_video_format(total_frames, height, width):
        return None, None
    
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter(OUTPUT_VIDEO, codec, fps_input, (width, height))
    count_frames= 0
    x_body, y_body, x_bar, y_bar = [], [], [], []
    count_multiple_detection = 0
    multiple_detection = False
    progress_bar = ph_graphx.progress(0)
    while cap.isOpened():
        progress_bar.progress(round((count_frames*100)/total_frames))
        count_frames+=1
        ret, frame = cap.read()

        if not ret:
            break
        if mp and not bar:
            frame, x_body, y_body = mp.process(keypoints_options, count_frames, total_frames, width, height, x_body, y_body, frame)
            out.write(frame)
        elif sim:
            frame, x_body, y_body = sim.process(keypoints_options, count_frames, total_frames, width, height, x_body, y_body, frame)
            out.write(frame)
        elif bar:
            frame, x_bar, y_bar, multiple_detection = bar.process(count_frames, total_frames, width, height, x_bar, y_bar, frame)
            frame, x_body, y_body = mp.process(keypoints_options, count_frames, total_frames, width, height, x_bar, y_bar, frame)
            out.write(frame)

    if bar:
        multiple_detection = check_multiple_detection(count_multiple_detection, total_frames)
    else:
        multiple_detection = False
    cap.release()
    out. release()
    
    return x_body, y_body, x_bar, y_bar, multiple_detection




