from os.path import dirname, abspath
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Global constants and parameters
OUTPUT_DIR = dirname(dirname(abspath(__file__))) + "/output_images/"

LANE_REGION_HEIGHT = 0.63       # top boundary (% of ysize)
LANE_REGION_UPPER_WIDTH = 0.05  # upper width (% of ysize, X2 for actual width)
LANE_REGION_LOWER_WIDTH = 0.45  # lower width (% of ysize, X2 for actual width)
LANE_WIDTH_POSTWARP = 0.25      # width of region to use after perspective txfm

# Test values, used for attempting to clear the "Harder Challenge"
# This doesn't work when the curve of the lane is too sharp, so don't use this
# LANE_REGION_HEIGHT = 0.68       # top boundary (% of ysize)
# LANE_REGION_UPPER_WIDTH = 0.12  # upper width (% of ysize, X2 for actual width)
# LANE_REGION_LOWER_WIDTH = 0.45  # lower width (% of ysize, X2 for actual width)
# LANE_WIDTH_POSTWARP = 0.20      # width of region to use after perspective txfm

L_ADJ_FACTOR = 42               # lightness threshold laxness (higher = more lax)
S_ADJ_FACTOR = 42               # saturation threshold laxness (higher = more lax)
SOBEL_L_THRESHOLD = 20          # gradient (lightness channel, x direction) threshold
SOBEL_S_THRESHOLD = 20          # gradient (saturation channel, x direction) threshold
SOBEL_KERNEL = 15               # kernel size for sobel gradient
LANE_EXTENTS = (600, 679)       # x-values to slice the center of the lane to use in averaging

NWINDOWS = 10                   # For sliding window pixel search, number of slices per image
MARGIN = 40                     # (Margin * 2) is the width of each window slice
MINPIX = 20                     # Number of found pixels needed to force a shift of the next window

MY = 30 / 720                   # meters per pixel in y dimension
MX = 3.7 / 700                  # meters per pixel in x dimension

PLOT = True                    # toggle whether to generate images for intermediate steps


# Finds lane lines on an image, marks them, and displays the radius of curvature
# and the delta between the center of the image and the center of the lane.
# Inputs:
#   image           image to process
#   mtx             camera matrix coefficients
#   dist            distortion coefficients
#   lheight         top boundary of region for perspective transform (% of img height)
#   lupwidth        upper width of region for perspective transform (% of img width, X2)
#   llowwidth       lower width of region for perspective transform (% of img width, X2)
#   lwidthwarped    width of search region post-perspective transform
#   left_fit        prior fit coefficients for left lane line, speeds up pixel search.
#                   if left as default "None", will proceed to pixel search independently.
#   right_fit       prior fit coefficients for left lane line, speeds up pixel search.
#                   if left as default "None", will proceed to pixel search independently.
#   my              conversion factor, meters per pixel in y dimension
#   mx              conversion factor, meters per pixel in x dimension
# Outputs:
#   output_img      processed image with lane lines marked and measurements displayed
#   left_fit        fit coefficients for left lane line, to be used next frame
#   right_fit       fit coefficients for left lane line, to be used next frame
def process_image(image, mtx, dist,
                  lheight=LANE_REGION_HEIGHT,
                  lupwidth=LANE_REGION_UPPER_WIDTH,
                  llowwidth=LANE_REGION_LOWER_WIDTH,
                  lwidthwarped=LANE_WIDTH_POSTWARP,
                  left_fit=None, right_fit=None, my=MY, mx=MX):
    # First, undistort the input image according to the calibration coefficients
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Calculate x-y values for perspective transform
    xsize = image.shape[1]
    ysize = image.shape[0]
    x_ul, x_ur = int(xsize * (0.5 - lupwidth)), int(xsize * (0.5 + lupwidth))
    x_ll, x_lr = int(xsize * (0.5 - llowwidth)), int(xsize * (0.5 + llowwidth))
    y_top = int(ysize * lheight)
    x_wl, x_wr = int(xsize * (0.5 - lwidthwarped)), int(xsize * (0.5 + lwidthwarped))

    # Apply perspective transform to look at the road ahead
    src_points = np.float32([[x_ul, y_top], [x_ur, y_top], [x_ll, ysize], [x_lr, ysize]])
    dst_points = np.float32([[x_wl, 0], [x_wr, 0], [x_wl, ysize], [x_wr, ysize]])
    warped_image = warp(undist, src_points, dst_points)

    # Apply color (R-channel and S-channel) and gradient thresholding
    thresd_warped = thresholding(warped_image)

    if left_fit is None or right_fit is None:
        warped_lane, left_fit, right_fit = find_polynomial(thresd_warped)
    else:
        warped_lane, left_fit, right_fit = search_around_poly(thresd_warped,
                                                                     left_fit, right_fit)

    lane_overlay = warp(warped_lane, dst_points, src_points) # switching src/dst unwarps

    overlaid_img = weighted_img(lane_overlay, undist)

    output_img = measure_stats(overlaid_img, left_fit, right_fit, my, mx)

    ### Visualization ###
    if PLOT:
        # Check undistorted image
        save_undist = np.copy(undist)
        cv2.line(save_undist, (x_ul, y_top), (x_ur, y_top), color=(255, 0, 0), thickness=3)
        cv2.line(save_undist, (x_ur, y_top), (x_lr, ysize), color=(255, 0, 0), thickness=3)
        cv2.line(save_undist, (x_lr, ysize), (x_ll, ysize), color=(255, 0, 0), thickness=3)
        cv2.line(save_undist, (x_ll, ysize), (x_ul, y_top), color=(255, 0, 0), thickness=3)
        plt.imsave(OUTPUT_DIR + "01_undistort.jpg", save_undist)
        plt.imshow(save_undist)
        plt.show()

        # Check warped image
        save_warped_img = np.copy(warped_image)
        cv2.line(save_warped_img, (x_wl, 0), (x_wr, 0), color=(255, 0, 0), thickness=3)
        cv2.line(save_warped_img, (x_wr, 0), (x_wr, ysize), color=(255, 0, 0), thickness=3)
        cv2.line(save_warped_img, (x_wr, ysize), (x_wl, ysize), color=(255, 0, 0), thickness=3)
        cv2.line(save_warped_img, (x_wl, ysize), (x_wl, 0), color=(255, 0, 0), thickness=3)
        plt.imsave(OUTPUT_DIR + "02_warped.jpg", save_warped_img)
        plt.imshow(save_warped_img)
        plt.show()

        # Check thresholded image
        save_thresd_warped = np.copy(thresd_warped)
        save_thresd_warped = cv2.cvtColor(255 * save_thresd_warped, cv2.COLOR_GRAY2RGB)
        cv2.line(save_thresd_warped, (x_wl, 0), (x_wr, 0), color=(255, 0, 0), thickness=3)
        cv2.line(save_thresd_warped, (x_wr, 0), (x_wr, ysize), color=(255, 0, 0), thickness=3)
        cv2.line(save_thresd_warped, (x_wr, ysize), (x_wl, ysize), color=(255, 0, 0), thickness=3)
        cv2.line(save_thresd_warped, (x_wl, ysize), (x_wl, 0), color=(255, 0, 0), thickness=3)
        plt.imsave(OUTPUT_DIR + "03_threshold.jpg", save_thresd_warped)
        plt.imshow(save_thresd_warped)
        plt.show()

        plt.imshow(warped_lane)
        plt.imsave(OUTPUT_DIR + "05_warpedlane.jpg", warped_lane)
        plt.show()

        plt.imshow(lane_overlay)
        plt.imsave(OUTPUT_DIR + "06_unwarpedlane.jpg", lane_overlay)
        plt.show()

        plt.imshow(overlaid_img)
        plt.imsave(OUTPUT_DIR + "07_overlaylane.jpg", overlaid_img)
        plt.show()

    return output_img, left_fit, right_fit


# Performs a perspective transform.
# Inputs:
#   img             image to transform
#   src             source point coordinates (pre-transform)
#   dst             destination points coordinates (post-transform)
# Outputs:
#   img             perspective-transformed image
def warp(img, src=0, dst=0):
    # Check for src/dst points - if not inputted, skip processing
    if np.ndim(src) < 2 or np.ndim(dst) < 2:
        raise ValueError('source and/or destination vertices not defined')

    img_size = (img.shape[1], img.shape[0])

    # Compute perspective transform M using given source and destination coordinates
    warp_matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, warp_matrix, img_size, flags=cv2.INTER_LINEAR)


# Applies color and gradinent thresholding to isolate lane line candidates.
# Inputs:
#   img             image to apply thresholding to
#   l_thresh_comp   lightness threshold laxness (higher = more lax)
#   s_thresh_comp   saturation threshold laxness (higher = more lax)
#   sxl_thresh      gradient (lightness channel, x direction) threshold
#   sxs_thresh      gradient (saturation channel, x direction) threshold
#   sobelk          kernel size for sobel gradient
#   lane_extents    x-values to slice the center of the lane to use in averaging
# Outputs:
#   combined        binary image with culled pixels == 0, passed pixels == 1
def thresholding(img, l_thresh_comp=L_ADJ_FACTOR, s_thresh_comp=S_ADJ_FACTOR,
                 sxl_thresh=SOBEL_L_THRESHOLD, sxs_thresh=SOBEL_S_THRESHOLD,
                 sobelk=SOBEL_KERNEL,
                 lane_extents=LANE_EXTENTS):
    image = np.copy(img)

    # Convert to HLS color space and keep only saturation channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Grab information from the center of the lane to compute what the "road"
    # looks like
    avg_l = np.mean(l_channel[:, lane_extents[0]:lane_extents[1]])  # road lightness
    avg_s = np.mean(s_channel[:, lane_extents[0]:lane_extents[1]])  # road saturation
    # Set lightness and saturation search thresholds
    # The lighter the road, the closer our threshold needs to be to 255
    new_l_thresh = (255 + avg_l) / 2 - l_thresh_comp
    new_s_thresh = (255 + avg_s) / 2 - s_thresh_comp

    # Sobel X on lightness channel
    sobel_xl = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobelk)  # Take the derivative in x
    abs_sobel_xl = np.absolute(sobel_xl)  # Nevermind direction, just want sharp edges
    scaled_sobel_xl = np.uint8(255 * abs_sobel_xl / np.max(abs_sobel_xl))

    # Threshold Sobel X on lightness channel
    sxl_binary = np.zeros_like(scaled_sobel_xl)
    sxl_binary[scaled_sobel_xl >= sxl_thresh] = 1

    # Sobel X on saturation channel
    sobel_xs = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=sobelk)  # Take the derivative in x
    abs_sobel_xs = np.absolute(sobel_xs)  # Nevermind direction, just want sharp edges
    scaled_sobel_xs = np.uint8(255 * abs_sobel_xs / np.max(abs_sobel_xs))

    # Threshold Sobel X on saturation channel
    sxs_binary = np.zeros_like(scaled_sobel_xs)
    sxs_binary[scaled_sobel_xs >= sxs_thresh] = 1

    # Threshold color channel
    r_channel = image[:, :, 0]
    r_binary = np.zeros_like(r_channel)
    r_binary[r_channel > new_l_thresh] = 1

    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[s_channel > new_s_thresh] = 1

    # Use red + saturation channel info for color-only selection
    # Use lightness sobel-x + red for gradient-based selection
    # If neither finds much, use gradients + saturation as a last effort to find pixels
    combined = np.zeros_like(r_binary)
    combined[((r_binary == 1) & (s_binary == 1)) |
             ((sxl_binary == 1) & (r_binary == 1)) |
             ((sxl_binary == 1) & (sxs_binary == 1) & (s_binary == 1))] = 1

    return combined


# Given an appropriate binary image, finds two new lane lines in the current frame.
# Inputs:
#   binary_warped   image frame (binary), perspective transformed already
# Outputs:
#   out_img         image with lane pixels marked, lane filled in green
#   left_fit        fit coefficients for current frame's left lane line
#   right_fit       fit coefficients for current frame's right lane line
def find_polynomial(binary_warped):
    # Find our lane pixels first
    out_img, leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape,
                                                                 leftx, lefty, rightx, righty)

    ## Visualization ##

    # Fill space between the polynomial fits
    out_img = fill_lane(out_img, left_fitx, right_fitx, ploty)

    # Color in the left and right lane pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plot the polylines (temporary, only for writeup visualization)
    # vertices_left = np.array([[x, y] for x, y in np.stack((left_fitx, ploty), axis=-1)], np.int32)
    # vertices_right = np.array([[x, y] for x, y in np.stack((right_fitx, ploty), axis=-1)], np.int32)
    # cv2.polylines(out_img, [vertices_left], isClosed=False, color=(255, 255, 0), thickness=3)
    # cv2.polylines(out_img, [vertices_right], isClosed=False, color=(255, 255, 0), thickness=3)

    return out_img, left_fit, right_fit


# Given an appropriate binary image, finds pixels most likely to belong to
# the left and right lane lines respectively.
# Inputs:
#   binary_warped   image frame (binary), perspective transformed already
#   nwindows        number of horizontal slices to use to try and isolate lane points
#   margin          double margin = width of each horizontal slice
#   minpix          no. of found pixels needed to force a shift for the next window
# Outputs:
#   out_img         image with lane pixels marked, lane filled in green
#   leftx           x-values for the left lane line point cloud
#   lefty           y-values for the left lane line point cloud
#   rightx          x-values for the right lane line point cloud
#   righty          y-values for the right lane line point cloud
def find_lane_pixels(binary_warped, nwindows=NWINDOWS, margin=MARGIN, minpix=MINPIX):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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


# Given two previous 2nd-order polynomials (describing previously found lane lines),
# finds two new lane lines in the current frame.
# Inputs:
#   binary_warped   image frame (binary), perspective transformed already
#   left_fit        fit coefficients for the previous left lane line
#   right_fit       fit coefficients for the previous right lane line
#   margin          margin (left and right) within which to search for lane pixels
# Outputs:
#   out_img         image with lane pixels marked, lane filled in green
#   new_left_fit    fit coefficients for current frame's left lane line
#   new_right_fit   fit coefficients for current frame's right lane line
def search_around_poly(binary_warped, left_fit, right_fit, margin=MARGIN):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on the LAST polynomial fit -- must supply values externally
    left_fit_prev = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2]
    right_fit_prev = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2]

    left_lane_inds = ((nonzerox >= left_fit_prev - margin) &
                      (nonzerox < left_fit_prev + margin))
    right_lane_inds = ((nonzerox >= right_fit_prev - margin) &
                       (nonzerox < right_fit_prev + margin))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    new_left_fit, new_right_fit, left_fitx, right_fitx, ploty \
        = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Fill space between the polynomial fits
    out_img = fill_lane(out_img, left_fitx, right_fitx, ploty)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, new_left_fit, new_right_fit


# Takes two point clouds and fits a 2nd-order polynomial to each.
# Inputs:
#   img_shape       array containing the pixel dimensions of the image
#   leftx           x-values of left-lane-line points
#   lefty           y-values of left-lane-line points
#   rightx          x-values of right-lane-line points
#   righty          y-values of right-lane-line points
# Outputs:
#   left_fit        fit coefficients for the left lane line
#   right_fit       fit coefficients for the right lane line
#   left_fitx       x-values for a set of points describing the left line
#   right_fitx      x-values for a set of points describing the right line
#   ploty           y-values for both sets of x-values
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calculate curve points using ploty, left_fit and right_fit
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return left_fit, right_fit, left_fitx, right_fitx, ploty


# Takes two sets of lane curve points and colors in the space between them.
# Inputs:
#   img             image to draw on
#   left_fitx       x-values of left-lane-line points
#   right_fitx      x-values of right-lane-line points
#   ploty           y-values common to both sets of lane-line points
# Outputs:
#   img             image with lane filled in green
def fill_lane(img, left_fitx, right_fitx, ploty):
    vertices_left = [[x, y] for x, y in np.stack((left_fitx, ploty), axis=-1)]
    vertices_right = [[x, y] for x, y in np.stack((right_fitx, ploty), axis=-1)]
    vertices_right.reverse()

    vertices = np.array(vertices_left + vertices_right).astype(int)

    cv2.fillPoly(img, [vertices], color=[0, 128, 0])

    return img


# Takes two images and overlays the first onto the second.
# Inputs:
#   overlay_img     image to overlay
#   initial_img     image to be overlaid upon
#   α               coefficient for OpenCV weighting function
#   β               coefficient for OpenCV weighting function
#   γ               coefficient for OpenCV weighting function
# Outputs:
#   img             weighted, overlaid image
def weighted_img(overlay_img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, overlay_img, β, γ)


# Calculates the curvature of the road, and the distance the camera is off from the lane
# center, in meters, and writes these values onto the image frame.
# Inputs:
#   img             image frame to write information onto
#   left_fit        fit coefficients for the left side curve
#   right_fit       fit coefficients for the right side curve
#   my              pixel-to-meters conversion for y-dimension (after perspective txfm)
#   mx              pixel-to-meters conversion for x-dimension (after perspective txfm)
# Outputs:
#   img             image frame with curvature and off-center distance written
def measure_stats(img, left_fit, right_fit, my, mx):
    # Choose the maximum y-value (the bottom of the image) for curvature measurement
    y_eval = img.shape[0] - 1

    # Using the formula for radius of curvature, we'll need to convert from pixel distances
    # to meters (on the perspective-txfm'd image)
    # Check that the left/right lines curve the same way - if they don't, the lane is
    # quite straight and any "radius" measurement will be incorrect
    if np.sign(left_fit[0]) == np.sign(right_fit[0]):
        left_curverad = (1 + (2 * (mx/my/my)*left_fit[0] * (y_eval*my) + (mx/my)*left_fit[1]) ** 2) ** (3/2) \
                        / abs(2 * (mx/my/my)*left_fit[0])
        right_curverad = (1 + (2 * (mx/my/my)*right_fit[0] * (y_eval*my) + (mx/my)*right_fit[1]) ** 2) ** (3/2) \
                         / abs(2 * (mx/my/my)*right_fit[0])
        mean_curverad = (left_curverad + right_curverad) // 2
    else:
        mean_curverad = float("inf")

    # Distance off-center can be calculated with just deltas
    left_edge = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_edge = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    off_center = mx * ((left_edge + right_edge) / 2 - img.shape[1] / 2)

    # Write text to image frame
    if mean_curverad == float("inf"):
        rad_message = "Radius of curvature: NA (straight)"
    else:
        rad_message = "Radius of curvature: %d m" % mean_curverad

    if off_center > 0:
        off_message = "Vehicle is %f m left of center" % off_center
    else:
        off_message = "Vehicle is %f m right of center" % -off_center

    cv2.putText(img, rad_message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5, color=(255, 255, 255), thickness=2, lineType=2)
    cv2.putText(img, off_message, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5, color=(255, 255, 255), thickness=2, lineType=2)

    return img
