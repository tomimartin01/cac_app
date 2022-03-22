import cv2
import csv

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
    

def write_csv(x, y):
    
    data=[]
    data.append(x)
    data.append(y)

    with open('data.csv', 'w', encoding='UTF8', newline='') as f:
        
        writer = csv.writer(f)
        writer.writerows(data)