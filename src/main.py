from os.path import dirname, abspath
from os import listdir
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src import processing as ps

# Directories to grab from/save to
OUTPUT_DIR = dirname(dirname(abspath(__file__))) + "/output_images/"
TEST_IMG_DIR = dirname(dirname(abspath(__file__))) + "/test_images/"
VIDEO_DIR = dirname(dirname(abspath(__file__))) + "/"


# Step 1: Read in calibration coefficients for undistorting images
# (Calibration is an independent step, please see src > calibration.py)
#
# Grab camera matrix from file
with open(OUTPUT_DIR + "/camera_matrix.txt", 'r') as f:
    mtx = np.array([[float(num) for num in line.split(' ')] for line in f])
print("Camera matrix:\r\n", mtx)

# Grab distortion coefficients from file
with open(OUTPUT_DIR + "/distort_coeffs.txt", 'r') as f:
    dist = np.array([[float(num) for num in line.split(' ')] for line in f])
print("Distortion coefficients:\r\n", dist)


# Step 2: Apply processing pipeline to test images
#
# Grab all JPG files in the directory
for fname in listdir(TEST_IMG_DIR):
    if fname.endswith(".jpg"):
        # Read in one image with applicable filename
        image = mpimg.imread(TEST_IMG_DIR + fname)
        print("Input:", fname)

        # Note: left_fit/right_fit outputs not used for single-image processing
        out_img, left_fit, right_fit = ps.process_image(image, mtx, dist)

        # Save to output folder
        outfile = fname.split('.')[0]
        plt.imsave(OUTPUT_DIR + outfile + "_output.jpg", out_img)


# Step 3: Apply processing pipeline to video
#
# Read in video clip
video = cv2.VideoCapture(VIDEO_DIR + "harder_challenge_video.mp4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_DIR + 'processed_video.mp4', fourcc, 24.9, (1280,720))

has_frame = True        # flag reporting success of frame grab
count = 1               # frame counter (for display only)
left_fit = None         # initial "None" toggles X-window pixel searching
right_fit = None        # subsequent values are returned by the pipeline itself
while has_frame:
    has_frame, frame = video.read()

    if has_frame:
        # Video frames are read in as BGR, convert to RGB for pipeline to work
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        out_img, left_fit, right_fit = ps.process_image(frame, mtx, dist,
                                                        left_fit=left_fit, right_fit=right_fit)

        # Convert back to BGR before writing to video file
        out.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        print("writing frame #%d" % count)

    count += 1

out.release()
cv2.destroyAllWindows()
