
import argparse
import os
from skimage import io
import json
import numpy as np
import cv2


# parse args
parser = argparse.ArgumentParser(description='Input path and output path')
parser.add_argument('input_path', help='the path to the input images and .json files')
parser.add_argument('output_path', help='the path where the output images and .json files should appear')

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

display = False # do you want to display one image?
which_image = 1 # which image to display


# read in input data
files = os.listdir(input_path)
image_files = [input_path + '\\' + x for x in files if x.endswith('.jpg')]
info_files = [input_path + '\\' + x for x in files if x.endswith('.info.json')] # we don't really need this
images = []
infos = []
for image_file in image_files:
    info_file = image_file + '.info.json'
    images.append(io.imread(image_file))
    with open(info_file) as f:
        data = json.load(f)
        infos.append(data)
edges = [x['canonical_board']['tl_tr_br_bl'] for x in infos]
bar_widths = [x['canonical_board']['bar_width_to_checker_width'] for x in infos]
board_widths = [x['canonical_board']['board_width_to_board_height'] for x in infos]
pip_lengths = [x['canonical_board']['pip_length_to_board_height'] for x in infos]

if display:
    # visualize one input image
    cv2.imshow('one input image', images[0])


# remove perspective
def four_point_transform(image, rect):
    rect = np.array(rect, dtype=np.float32)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

no_persps = []
for i, image in enumerate(images):
    # umat = cv2.UMat(np.array(image, dtype=np.uint8))
    no_persp = four_point_transform(image, edges[i])
    no_persps.append(no_persp)
# convert to grayscale for hough circles
grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in no_persps]
    
if display:
    # visualize one image with the perspective removed in grayscale
    cv2.imshow('one perspectiveless image', grays[which_image])


# account for distortion of image
resizeds = []
for i, img in enumerate(grays):
    scale_percent_x = 100
    # scale_percent_x = board_widths[i] * 100 #- 20 # percent of original size
    # print(scale_percent_x)
    width = int(img.shape[1] * scale_percent_x / 100)
    height = int(img.shape[0] * 100 / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resizeds.append(resized)
    
    
# use hough circles to detect the circles
processed = []
processing = []
centres = []
for i, gray in enumerate(resizeds):
    
    for _ in range(9):
        gray = cv2.blur(gray, (3, 3))
    # gray = cv2.medianBlur(gray, 5)
    # gray = cv2.Laplacian(gray, cv2.CV_64F).astype(np.uint8)
    gray = cv2.Canny(gray, 30, 100)
    # sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # for _ in range(5):
    #     gray = cv2.filter2D(gray, -1, sharpening_kernel)
    
    # output = gray.copy()
    output = no_persps[i].copy()
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, param2=60, minRadius=20, maxRadius=90)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=60, minRadius=20, maxRadius=90)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=20, minRadius=20, maxRadius=60)
    centres.append([])
    if circles is not None:
        print(len(circles[0]), 'circles found')
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (255, 0, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), -1)
            centres[i].append((x, y))
    else:
        print('0 circles found')
    processed.append(output)
    processing.append(gray)
    print('image', i, 'processed')
    # break
       
if display:     
    cv2.imshow('one image with circles', processed[which_image])
    cv2.imshow('what the algorithm sees', processing[which_image])


# output results
for i in range(len(image_files)):
    fname = output_path + '\\' + image_files[i].split('\\')[1]
    processed_filename = fname[:-4] + '.visual_feedback.jpg'
    io.imsave(processed_filename, processed[i])
    
    output_filename = '.'.join(processed_filename.split('.')[:-2]) + '.checkers.json'
    
    total_width = images[i].shape[0]
    total_height = images[i].shape[1]
    pip_width = total_width / 12
    out_dir = {}
    top = np.zeros(12)
    bottom = np.zeros(12)
    for centre in centres[i]:
        # should calculate which pip it is located on based on x coordinate:
        # for that, calculate the width of a pip by dividing the board width by the number of pips
        # then based on the y coordinate, tell whether it's in the top or bottom row
        pip_index = int(centre[0] // pip_width)
        if pip_index >= 12: # then we made a mistake
            pip_index = 11
        top_or_bottom = 'top'
        if centre[1] > total_height / 2:
            top_or_bottom = 'bottom'
        if top_or_bottom == 'top':
            top[pip_index] += 1
        if top_or_bottom == 'bottom':
            bottom[pip_index] += 1
    out_dir['top'] = list(top)
    out_dir['bottom'] = list(bottom)
    out_json = json.dumps(out_dir, indent=4)
    with open(output_filename, 'w') as f:
        f.write(out_json)