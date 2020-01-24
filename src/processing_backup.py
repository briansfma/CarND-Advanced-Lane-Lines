from os.path import dirname, abspath
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Global constants and parameters
OUTPUT_DIR = dirname(dirname(abspath(__file__))) + "/output_images/"

LANE_REGION_HEIGHT = 0.63       # top boundary (% of ysize)
LANE_REGION_UPPER_WIDTH = 0.05  # upper width (% of ysize, X2 for actual width)
LANE_REGION_LOWER_WIDTH = 0.45  # lower width (% of ysize, X2 for actual width)
LANE_WIDTH_POSTWARP = 0.25  # width of region to use after perspective txfm

R_THRESHOLDS = (200, 255)
S_THRESHOLDS = (90, 255)
SOBEL_THRESHOLDS = (20, 100)
SOBEL_KERNEL = 15

MY = 30 / 720                   # meters per pixel in y dimension
MX = 3.7 / 700                  # meters per pixel in x dimension


def process_image(image, mtx, dist,
                  lheight=LANE_REGION_HEIGHT,
                  lupwidth=LANE_REGION_UPPER_WIDTH,
                  llowwidth=LANE_REGION_LOWER_WIDTH,
                  lwidthwarped=LANE_WIDTH_POSTWARP,
                  left_fit=0, right_fit=0, my=MY, mx=MX,
                  fname=" "):
    # Quick filename clipper to get rid of the .jpg extension
    filename = fname.split('.')[0]

    # Calculate x-y values for region marking
    xsize = image.shape[1]
    ysize = image.shape[0]
    x_ul, x_ur = int(xsize * (0.5 - lupwidth)), int(xsize * (0.5 + lupwidth))
    x_ll, x_lr = int(xsize * (0.5 - llowwidth)), int(xsize * (0.5 + llowwidth))
    y_top = int(ysize * lheight)
    x_wl, x_wr = int(xsize * (0.5 - lwidthwarped)), int(xsize * (0.5 + lwidthwarped))

    # First, undistort the input image according to the calibration coefficients
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    # Check undistorted image
    save_undist = np.copy(undist)
    cv2.line(save_undist, (x_ul, y_top), (x_ur, y_top), color=(255,0,0), thickness=3)
    cv2.line(save_undist, (x_ur, y_top), (x_lr, ysize), color=(255,0,0), thickness=3)
    cv2.line(save_undist, (x_lr, ysize), (x_ll, ysize), color=(255,0,0), thickness=3)
    cv2.line(save_undist, (x_ll, ysize), (x_ul, y_top), color=(255,0,0), thickness=3)
    plt.imsave(OUTPUT_DIR + filename + "_01_undistort.jpg", save_undist)
    plt.imshow(save_undist)
    plt.show()

    # Apply color (R-channel and S-channel) and gradient thresholding
    thresd_img = thresholding(undist)
    # Check thresholded image
    save_thresd_img = np.copy(thresd_img)
    save_thresd_img = cv2.cvtColor(255*save_thresd_img, cv2.COLOR_GRAY2RGB)
    cv2.line(save_thresd_img, (x_ul, y_top), (x_ur, y_top), color=(255,0,0), thickness=3)
    cv2.line(save_thresd_img, (x_ur, y_top), (x_lr, ysize), color=(255,0,0), thickness=3)
    cv2.line(save_thresd_img, (x_lr, ysize), (x_ll, ysize), color=(255,0,0), thickness=3)
    cv2.line(save_thresd_img, (x_ll, ysize), (x_ul, y_top), color=(255,0,0), thickness=3)
    plt.imsave(OUTPUT_DIR + filename + "_02_threshold.jpg", save_thresd_img)
    plt.imshow(save_thresd_img)
    plt.show()

    src_points = np.float32([[x_ul, y_top], [x_ur, y_top], [x_ll, ysize], [x_lr, ysize]])
    dst_points = np.float32([[x_wl, 0], [x_wr, 0], [x_wl, ysize], [x_wr, ysize]])

    warped_image = warp(thresd_img, src_points, dst_points)
    # Check warped image
    save_warped_img = np.copy(warped_image)
    save_warped_img = cv2.cvtColor(255*save_warped_img, cv2.COLOR_GRAY2RGB)
    cv2.line(save_warped_img, (x_wl, 0), (x_wr, 0), color=(255,0,0), thickness=3)
    cv2.line(save_warped_img, (x_wr, 0), (x_wr, ysize), color=(255,0,0), thickness=3)
    cv2.line(save_warped_img, (x_wr, ysize), (x_wl, ysize), color=(255,0,0), thickness=3)
    cv2.line(save_warped_img, (x_wl, ysize), (x_wl, 0), color=(255,0,0), thickness=3)
    plt.imsave(OUTPUT_DIR + filename + "_03_warped.jpg", save_warped_img)
    plt.imshow(save_warped_img)
    plt.show()

    if np.ndim(left_fit) < 1 or np.ndim(right_fit) < 1:
        warped_lane, left_fit, right_fit, ploty = fit_polynomial(warped_image)
        plt.imshow(warped_lane)
        plt.imsave(OUTPUT_DIR + filename + "_05_warpedlane.jpg", warped_lane)
        plt.show()

        lane_overlay = unwarp(warped_lane, src_points, dst_points)
        plt.imshow(lane_overlay)
        plt.imsave(OUTPUT_DIR + filename + "_06_unwarpedlane.jpg", lane_overlay)
        plt.show()

        overlaid_img = weighted_img(lane_overlay, undist)
        plt.imshow(overlaid_img)
        plt.imsave(OUTPUT_DIR + filename + "_07_overlaylane.jpg", overlaid_img)
        plt.show()

        output_img, curverad, offcenter = measure_stats(overlaid_img, ploty,
                                                        left_fit, right_fit, my, mx)
        plt.imshow(output_img)
        plt.imsave(OUTPUT_DIR + filename + "_08_finaloutput.jpg", output_img)
        plt.show()


    return 0


# Helper function: Color and Gradient Thresholding
def thresholding(img, r_thresh=R_THRESHOLDS, s_thresh=S_THRESHOLDS,
                 sx_thresh=SOBEL_THRESHOLDS, sobelk=SOBEL_KERNEL):
    image = np.copy(img)

    # Convert to HLS color space and keep only lightness and saturation channels
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel X
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobelk)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold Sobel X
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold red channel
    r_channel = image[:, :, 0]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    # Combine red channel and saturation channel info
    combined = np.zeros_like(r_binary)
    combined[((r_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1

    return combined


def warp(img, src=0, dst=0):
    # Check for src/dst points - if not inputted, skip processing
    if np.ndim(src) < 2 or np.ndim(dst) < 2:
        raise ValueError('source and/or destination vertices not defined')

    img_size = (img.shape[1], img.shape[0])

    # Compute perspective transform M using given source and destination coordinates
    warp_matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, warp_matrix, img_size, flags=cv2.INTER_LINEAR)


def unwarp(img, src=0, dst=0):
    # Check for src/dst points - if not inputted, skip processing
    if np.ndim(src) < 2 or np.ndim(dst) < 2:
        raise ValueError('source and/or destination vertices not defined')

    img_size = (img.shape[1], img.shape[0])

    # Compute perspective transform M using given source and destination coordinates
    warp_matrix = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(img, warp_matrix, img_size, flags=cv2.INTER_LINEAR)


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    out_img, leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to the pixels we found in each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##

    # Fill space between the polynomial fits
    out_img = fill_lane(out_img, left_fitx, right_fitx, ploty)

    # Color in the left and right lane pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plot the polylines (temporary, only for writeup visualization)
    # vertices_left = np.array([[x, y] for x, y in np.stack((left_fitx, ploty), axis=-1)], np.int32)
    # vertices_right = np.array([[x, y] for x, y in np.stack((right_fitx, ploty), axis=-1)], np.int32)
    # cv2.polylines(out_img, [vertices_left], isClosed=False, color=(255,255,0), thickness=3)
    # cv2.polylines(out_img, [vertices_right], isClosed=False, color=(255,255,0), thickness=3)

    return out_img, left_fit, right_fit, ploty


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = int(leftx_current - margin)
        win_xleft_high = int(leftx_current + margin)
        win_xright_low = int(rightx_current - margin)
        win_xright_high = int(rightx_current + margin)

        # Draw the windows on the visualization image (temporary, only for writeup visualization)
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = [i for i in range(len(nonzerox))
                          if ((win_xleft_low <= nonzerox[i] < win_xleft_high)
                              and (win_y_low <= nonzeroy[i] < win_y_high))]
        good_right_inds = [i for i in range(len(nonzerox))
                           if ((win_xright_low <= nonzerox[i] < win_xright_high)
                               and (win_y_low <= nonzeroy[i] < win_y_high))]

        # Append these indices to the lists
        left_lane_inds.extend(good_left_inds)
        right_lane_inds.extend(good_right_inds)

        # If we found > minpix pixels, recenter next window (leftx_current/rightx_current)
        # on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.mean(nonzerox[good_left_inds])
        if len(good_right_inds) > minpix:
            rightx_current = np.mean(nonzerox[good_right_inds])

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return out_img, leftx, lefty, rightx, righty


def fill_lane(img, left_fitx, right_fitx, ploty):

    vertices_left = [[x, y] for x, y in np.stack((left_fitx, ploty), axis=-1)]
    vertices_right = [[x, y] for x, y in np.stack((right_fitx, ploty), axis=-1)]
    vertices_right.reverse()

    vertices = np.array(vertices_left + vertices_right).astype(int)

    cv2.fillPoly(img, [vertices], color=[0, 128, 0])

    return img


def weighted_img(overlay_img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, overlay_img, β, γ)


# Calculates the curvature of the road, and the distance the camera is off from the lane
# center, in meters, and writes these values onto the image frame.
# Inputs:
#   ploty           array of y-coordinates of all points we fit the curve to
#   left_fit        fit coefficients for the left side curve
#   right_fit       fit coefficients for the right side curve
#   my              pixel-to-meters conversion for y-dimension (after perspective txfm)
#   mx              pixel-to-meters conversion for x-dimension (after perspective txfm)
# Outputs:
#   output_img      image frame with curvature and off-center distance written
#   mean_curverad   mean radius of curvature between left and right lanes
#   off_center      distance of the center of the image from the center of the lane
def measure_stats(img, ploty, left_fit, right_fit, my, mx):
    # Choose the maximum y-value (the bottom of the image) for curvature measurement
    y_eval = np.max(ploty)

    # Using the formula for radius of curvature, we'll need to convert from pixel distances
    # to meters (on the perspective-txfm'd image)
    left_curverad = (1 + (2 * (mx/my/my)*left_fit[0] * y_eval*my + (mx/my)*left_fit[1]) ** 2) ** (3/2) \
                    / abs(2 * (mx/my/my)*left_fit[0])
    right_curverad = (1 + (2 * (mx/my/my)*right_fit[0] * y_eval*my + (mx/my)*right_fit[1]) ** 2) ** (3/2) \
                     / abs(2 * (mx/my/my)*right_fit[0])

    mean_curverad = (left_curverad + right_curverad) // 2

    # Distance off-center can be calculated with just deltas
    left_edge = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_edge = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    off_center = mx * ((left_edge + right_edge)/2 - img.shape[1]/2)

    # Write text to image frame
    rad_message = "Radius of curvature: %d m" % mean_curverad
    if off_center > 0:
        off_message = "Vehicle is %f m left of center" % off_center
    else:
        off_message = "Vehicle is %f m right of center" % -off_center
    cv2.putText(img, rad_message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5, color=(255,255,255), thickness=2, lineType=2)
    cv2.putText(img, off_message, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5, color=(255,255,255), thickness=2, lineType=2)

    return img, mean_curverad, off_center


