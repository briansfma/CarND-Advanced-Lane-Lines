from os.path import dirname, abspath
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Global constants
CAL_DIR = dirname(dirname(abspath(__file__))) + "/camera_cal/"
OUTPUT_DIR = dirname(dirname(abspath(__file__))) + "/output_images/"
NX = 9
NY = 6


# Calculates the camera matrix and distortion coefficients for a set of images.
# Inputs:
#   dir             directory containing a set of checkerboard images
#   nx              number of columns of corners expected per image in the set
#   ny              number of rows of corners expected per image in the set
# Outputs:
#   ret             flag reporting success of the calibration procedure
#   mtx             camera matrix
#   dist            distortion coefficients
def calibrate_camera(directory, nx, ny):
    images = glob.glob(directory + 'calibration*.jpg')

    # Check that inputs are usable for calibration
    if len(images) < 1:
        raise ValueError('No images found to import')

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (7,5,0)
    # Since the chessboard grid is 9x6, columns are 0-8 and rows 0-5
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # this should fill values

    for fname in images:
        # Grab image and convert to grayscale
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    return ret, mtx, dist


# Run script to generate camera matrix and distortion coefficients
#
# Example image to undistort
test_img = mpimg.imread(CAL_DIR + 'calibration1.jpg')

# Generate and save camera matrix, distortion coefficients
ret, mtx, dist = calibrate_camera(CAL_DIR, NX, NY)

print("Camera matrix:")
print(mtx)
with open(OUTPUT_DIR + "camera_matrix.txt", "w+") as f:
    np.savetxt(f, mtx)

print("Distortion coefficients:")
print(dist)
with open(OUTPUT_DIR + "distort_coeffs.txt", "w+") as f:
    np.savetxt(f, dist)

print("Values saved in " + OUTPUT_DIR)

# Apply to undistortion, save image to output folder
undist = cv2.undistort(test_img, mtx, dist, None, mtx)
plt.imsave(OUTPUT_DIR + 'calibration1-undistorted.jpg', undist)
plt.show()