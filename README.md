**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Chessboard Example"
[image2]: ./output_images/calibration1-undistorted.jpg "Undistorted"

[image3]: ./test_images/test6.jpg "Example Road"
[image4]: ./output_images/00_undistort.jpg "Undistort Road"
[image5]: ./output_images/01_undistort.jpg "Undistort w/ Transform src"
[image6]: ./output_images/02_warped.jpg "Road Transformed w/ dst"
[image7]: ./output_images/03_threshold.jpg "Thresholded Binary"
[image8]: ./output_images/04_foundpoints.jpg "Founds Points, Lines"
[image9]: ./output_images/05_warpedlane.jpg "Generate Warped Lane"
[image10]: ./output_images/06_unwarpedlane.jpg "Unwarp Lane"
[image11]: ./output_images/07_overlaylane.jpg "Final Output"
[image12]: ./output_images/straight_lines1_output.jpg "Straight Example"

[video1]: ./project_processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Python script `calibration.py` located in "./src/".

Most of the setup for calibration was borrowed from the quizzes' boilerplate code. "Object points" and "Image points" containers were prepared to feed accumulated info to `cv2.calibrateCamera()`; `objpoints` contains sets of (x,y,z) points (z = 0 here since this chessboard is only 2D) and `imgpoints` contains sets of corresponding (x,y) pixel information. For each image read, if `cv2.findChessboardCorners()` is able to find all the chessboard corners, we append corner locations to `imgpoints`, and corresponding object point positions to `objpoints`.

The result is a list of 1-to-1 `objpoints` that maps to `imgpoints`, which we feed into `cv2.calibrateCamera()` to generate a camera matrix and distortion coefficients. These are saved into "./output_images/camera_matrix.txt" and "./output_images/distort_coeffs.txt" for other programs to use. With `cv2.undistort()` we can then apply a distortion correction to an image taken by the same camera.

Before:
![alt text][image1]

After:
![alt text][image2]


### Pipeline (single images)

#### 0. Organization

The image processing pipeline and all of its helper functions are contained in the Python module `processing.py` located in "./src/". The `main.py` Python script (also in "./src/") runs the pipeline `process_image()` (lines 58-140 in `processing.py`) after importing camera calibration coefficients and reading in images or videos as requested. All the "levers" a user can pull (input and output directories, parameters, constants, etc.) are located at the top of `main.py` or `processing.py` below the initial imports, for simpler manipulation of pipeline settings.

#### 1. Provide an example of a distortion-corrected image.

We will use "./test_images/test6.jpg" as the example for single-image pipeline processing. First, using the camera matrix and distortion coefficients generated during calibration, we can `cv2.undistort()` the image of the road.

Before:
![alt text][image3]

After:
![alt text][image4]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Slightly deviating from what the project expects, I opted to perform the perspective transform first before thresholding, as it magnifies (in the image frame) the lane in the distance, giving the thresholding function less surrounding noise to deal with.

The perspective transform helper function `warp()` is located in lines 150-159 in `processing.py`. It takes the image (`img`), source and destination points (`src` and `dst` respectively) as inputs and returns the transformed image. The source and destination points are not hardcoded; the trapezoidal window (pre-transform) and rectangular window (post-transform) can be controlled by manipulating lines 9-12 of `processing.py`:

```

LANE_REGION_HEIGHT = 0.63       # top boundary (% of ysize)
LANE_REGION_UPPER_WIDTH = 0.05  # upper width (% of ysize, X2 for actual width)
LANE_REGION_LOWER_WIDTH = 0.45  # lower width (% of ysize, X2 for actual width)
LANE_WIDTH_POSTWARP = 0.25      # width of region to use after perspective txfm

```

The pre-transform `src` and post-transform `dst` points are then calculated in lines 68-77 in `processing.py`:

```

    xsize = image.shape[1]
    ysize = image.shape[0]
    x_ul, x_ur = int(xsize * (0.5 - lupwidth)), int(xsize * (0.5 + lupwidth))
    x_ll, x_lr = int(xsize * (0.5 - llowwidth)), int(xsize * (0.5 + llowwidth))
    y_top = int(ysize * lheight)
    x_wl, x_wr = int(xsize * (0.5 - lwidthwarped)), int(xsize * (0.5 + lwidthwarped))

    src_points = np.float32([[x_ul, y_top], [x_ur, y_top], [x_ll, ysize], [x_lr, ysize]])
    dst_points = np.float32([[x_wl, 0], [x_wr, 0], [x_wl, ysize], [x_wr, ysize]])

```

All the x- and y-values were forced to int() after calculation as it's easily possible for the results to be non-integer values. With the parameters set as they appear currently, `test6.jpg` looks like this after applying `warp()`. The red lines mark the `src` and `dst` points only for visualization, they are not displayed in actual usage.

Before:
![alt text][image5]

After:
![alt text][image6]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The thresholding helper function `thresholding()` is located in lines 173-228 in `processing.py`. It takes the image (`img`), several compensation/thresholding factors (`l_thresh_comp`, `s_thresh_comp`, `sxl_thresh`, `sxs_thresh`, `sobelk`) and x-values of a slice of the lane (`lane_extents`) as inputs and returns a binary image with the thresholding performed. All the levers to control thresholding behavior can be manipulated in lines 21-26 of `processing.py`:

```

L_ADJ_FACTOR = 42               # lightness threshold laxness (higher = more lax)
S_ADJ_FACTOR = 42               # saturation threshold laxness (higher = more lax)
SOBEL_L_THRESHOLD = 20          # gradient (lightness channel, x direction) threshold
SOBEL_S_THRESHOLD = 20          # gradient (saturation channel, x direction) threshold
SOBEL_KERNEL = 15               # kernel size for sobel gradient
LANE_EXTENTS = (600, 679)       # x-values to slice the center of the lane to use in averaging

```

In order to deal with pavement of varying colors, I calculate dynamic thresholds for color and saturation based on an average of the middle of the lane. The average lightness and average saturation of the lane within `lane_extents` is used to determine new thresholds based on this calculation in lines 186-191. The brighter the pavement, the closer we set the new thresholding (assuming, as always, that lanes are brighter than the road) to max brightness; the compensation factors lower the bar a bit so we can find more pixels.

```

    avg_l = np.mean(l_channel[:, lane_extents[0]:lane_extents[1]])  # road lightness
    avg_s = np.mean(s_channel[:, lane_extents[0]:lane_extents[1]])  # road saturation

    new_l_thresh = (255 + avg_l) / 2 - l_thresh_comp
    new_s_thresh = (255 + avg_s) / 2 - s_thresh_comp

```

I found it useful to take the x-direction gradient in both lightness and saturation, as they picked up different information (saturation gradient often picked up road textures along with lane lines if road is bright, but better rejects dark road lines like expansion gaps, tar, etc). So this led to a 3-pronged approach to trying to find lane line pixels:

1) Color-only (simple): if the pixel is bright in red and in saturation, it's likely to be a lane line pixel
2) Lightness gradient + lightness: if the pixel is on a darker-to-brighter edge and bright in red, it's likely to be a lane line pixel
3) Lightness gradient + saturation gradient + saturation: even if the pixel isn't bright in red, if it is on a darker-to-brighter edge and a less-to-more-saturated edge, and high enough in saturation, we will include it

The code to apply these thresholds was largely borrowed from quiz code, but I stopped using an upper limit for the threshold as there realistically are few objects sharper and brighter than lane markers on the pavement. As an example, the code for applying thresholds to the Sobel X-derivative on the lightness channel (lines 193-200):

```

    # Sobel X on lightness channel
    sobel_xl = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobelk)  # Take the derivative in x
    abs_sobel_xl = np.absolute(sobel_xl)  # Nevermind direction, just want sharp edges
    scaled_sobel_xl = np.uint8(255 * abs_sobel_xl / np.max(abs_sobel_xl))

    # Threshold Sobel X on lightness channel
    sxl_binary = np.zeros_like(scaled_sobel_xl)
    sxl_binary[scaled_sobel_xl >= sxl_thresh] = 1

```

And likewise for simple brightness thresholding (lines 211-214):

```

    # Threshold color channel
    r_channel = image[:, :, 0]
    r_binary = np.zeros_like(r_channel)
    r_binary[r_channel > new_l_thresh] = 1

```

The three approaches to finding pixels were implemented via bitwise logic (lines 223-226), following class examples.

```

    combined = np.zeros_like(r_binary)
    combined[((r_binary == 1) & (s_binary == 1)) |
             ((sxl_binary == 1) & (r_binary == 1)) |
             ((sxl_binary == 1) & (sxs_binary == 1) & (s_binary == 1))] = 1

```

Applied to the image, often times approach 3) could "see" the farthest, identifying two edges per lane line nearly to the horizon. Approach 2) was best at identifying small dots (like cats' eye markers) in a dotted line. Approach 1) "filled in" the near field pixels found, making it easier to find the start of a lane line.

The combined output of these three pixel finding approaches looks like this on `test6.jpg`:

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The polynomial line-finding helper function `find_polynomial()` is located in lines 238-260 in `processing.py`. It itself requires two helper functions, a lane-pixel identifier `find_lane_pixels()` and a polynomial fitting function `fit_poly()` located at lines 276-342 and lines 408-424 respectively. `find_polynomial()` takes the binary image output from `thresholding()` and returns a color image with the lane, left line, and right line pixels filled in (`out_img`), and fit coefficients for the left and right lines (`left_fit` and `right_fit`). All the levers to control pixel-finding can be manipulated in lines 28-30 of `processing.py`:

```

NWINDOWS = 10                   # For sliding window pixel search, number of slices per image
MARGIN = 40                     # (Margin * 2) is the width of each window slice
MINPIX = 20                     # Number of found pixels needed to force a shift of the next window

```

The first step to finding a polynomial fit is to find all the correct pixels to fit. In `find_lane_pixels()`, following the sliding window method taught in class, I first take a histogram on the bottom half of the warped image to find approximately where the lane starts near the car. Using as narrow of a window as possible, the search slides up the image until it finds all the likely lane pixels. Most of this code is borrowed from class quizzes, I only tried to reduce some looping to attempt to improve processing speed.

The line pixels found are fed into `fit_poly()` (again, borrowed from quiz code, but it's a straightforward function) which generates the left line and right line fit coefficients to return. For visualization only, I used class quiz code to draw search windows onto an example image and `cv2.polylines()` to draw a polyline approximation of the fitted curves for `test6.jpg`.

![alt text][image8]

In actual use, the lines and windows are not drawn, instead the lane between the lines is filled, and then the left line and right line pixels are colored in on top of the lane.

![alt text][image9]


Note: for video usage, `find_polynomial()` is only used on frame 1, afterwards the last found polynomial fits are passed back to `process_image()` such that the function will call another helper function `search_around_poly()` from frame 2 onwards. The toggle is a simple "None or not" check (lines 83-87 in `processing.py`):

```

    if left_fit is None or right_fit is None:
        warped_lane, left_fit, right_fit = find_polynomial(thresd_warped)
    else:
        warped_lane, left_fit, right_fit = search_around_poly(thresd_warped, left_fit, right_fit)

```

`search_around_poly()` (lines 356-392) also borrows from quiz code, a straightforward function that sets a margin (40px with current parameters) around the last found polynomial lines and logs which points are both thresholded (white) and within the margin. Once those points are found, the rest of the functionality is the same as `find_polynomial()` where we fill in the lane area and color the left line and right line pixels. `search_around_poly()`, logically, returns the same quantities as `find_polynomial()`, and the fit coefficients for left and right lines (`new_left_fit` and `new_right_fit`) get used for processing the next frame.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The road stats helper function `measure_stats()` is located in lines 470-508 in `processing.py`. It takes the image (`img`), the left and right line fit coefficients (`left_fit` and `right_fit`) and pixels-to-meters conversion factors for x and y (`mx` and `my`) and returns an image with the curvature and off-center distance drawn on top of the frame. As I was unsure whether the `mx` and `my` values provided were accurate at all after modifying the perspective transform window, they are left modifiable in lines 32 and 33:

```

MY = 30 / 720                   # meters per pixel in y dimension
MX = 3.7 / 700                  # meters per pixel in x dimension

```

I used the hint from class regarding converting pixels to meters to implement the curvature formula as suggested:

```

left_curverad = (1 + (2 * (mx/my/my)*left_fit[0] * (y_eval*my) + (mx/my)*left_fit[1]) ** 2) ** (3/2) \
                / abs(2 * (mx/my/my)*left_fit[0])
right_curverad = (1 + (2 * (mx/my/my)*right_fit[0] * (y_eval*my) + (mx/my)*right_fit[1]) ** 2) ** (3/2) \
                 / abs(2 * (mx/my/my)*right_fit[0])

```

But taking the mean curvature between these two becomes a problem if 1) the radius is so large as to be difficult to comprehend/measure in reality, or 2) the road is actually perfectly straight, and the curvature of the two line fits is due only to image transforms. So I opted to include a condition where, if the two lines have opposite concavity, the function assumes the road is actually straight and marks the mean curvature as infinite. The resulting code is in lines 478-485:

```

    if np.sign(left_fit[0]) == np.sign(right_fit[0]):
        left_curverad = (1 + (2 * (mx/my/my)*left_fit[0] * (y_eval*my) + (mx/my)*left_fit[1]) ** 2) ** (3/2) \
                        / abs(2 * (mx/my/my)*left_fit[0])
        right_curverad = (1 + (2 * (mx/my/my)*right_fit[0] * (y_eval*my) + (mx/my)*right_fit[1]) ** 2) ** (3/2) \
                         / abs(2 * (mx/my/my)*right_fit[0])
        mean_curverad = (left_curverad + right_curverad) // 2
    else:
        mean_curverad = float("inf")

```

The infinite value is not immediately useful to this assignment since we are only drawing the text on the frame, so any marker will do but it's nice to be physically correct (as a road straightens out, the mean curvature will increase towards infinite anyways).

The distance off-center of the vehicle is very straightforward to calculate (lines 488-490):

```

    left_edge = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_edge = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    off_center = mx * ((left_edge + right_edge) / 2 - img.shape[1] / 2)

```

This returns a positive value if the car is left of the center of the lane, negative if on the right.

These two values gettin written to the final image frame after the lane is overlaid, which we'll discuss in the next section.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Continuing from section 4), once the lane has been marked, we now need to reverse the perspective transform to have a lane we can overlay back on the original colored road image. This final processing is handled from lines 89-93 in `processing.py`, 

```

    lane_overlay = warp(warped_lane, dst_points, src_points) # switching src/dst unwarps

    overlaid_img = weighted_img(lane_overlay, undist)

    output_img = measure_stats(overlaid_img, left_fit, right_fit, my, mx)

```

First `warp()` with reversed `src` and `dst` unwarps the lane:

![alt text][image10]

Then `weighted_img()` (helper function lifted directly from quiz code) handles the overlay work. Finally, the mean curvature and off-center  distance are calculated and drawn onto the frame in `measure_stats()`.

![alt text][image11]

`measure_stats()` uses `cv2.putText()` to write the curvature and off-center distance onto the frame, the message just gets tailored a bit depending on the values to write. If the road is straight (aka curvature "infinite"), we adjust the message, for example:

![alt text][image12]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_processed.mp4)

After fine-tuning parameters a bit, I was able to get the Challenge video to mostly work as well. There is a wobble near the car's position as it passes under the bridge (I looked at R/G/B/H/S/L channel data to hunt for clues, there just doesn't seem to be much information at such low exposure) but the lines stay steady without catastrophic failure. Here's a [link to this video](./challenge_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest pitfall to this pipeline is the perspective transform: since (pixel height):(distance in reality) ramps up as your viewing angle gets more horizontal, the upper boundary of the `src` points is hypercritical. I would need to implement some kind of horizon finding in order to make the whole pipeline more robust when dealing with inclines and such. Imagine if you're at the top of a hill looking down, the perspective transform could allow sky pixels in and that could throw off the entire thresholding/histogram/polyline operation.

The perspective transform also fails when the road curvature becomes sharp enough: since the functions assume that a curve fit will extend from the bottom of the warped image to the top, when the road curves sharply enough the lane line points run off the side of the image, breaking the ability to search in the next frame for new lane line pixels higher up. This is painfully obvious two turns in for the Harder Challenge video [link to my attempt](./harder_challenge_Attempt1.mp4) Adjusting the perspective transform window (post-warp) can alleviate this somewhat, but as we see more and more of the lane, we find that tight, curvy roads aren't easily modeled by a 2nd-order polynomial fit.

Several times when attempting the Harder Challenge I exported and looked at individual frames and found that they basically look like S-curves after applying the perspective transform. Not only does that make it hard to fit a 2nd-order polynomial, the resulting poor fit sets the pipeline up for failure when searching for lane pixels in the next frame. It's probably possible to fit a 3rd-order polynomial, but that does not help issue #2 at all if the lane line physically runs off the side of the screen.

That leads to the last issue, where in the Harder Challenge video, the white lane actually disappears off the screen as the car approaches a blind right hairpin turn. Since our pipeline has no memory besides the last fit coefficients, we can't "remember" where the white line was and all the pipeline can find is a yellow line in the center, leaving no lane to draw without a right line. 

The best attempt to solve these challenges, without adding variables/containers or features, is probably to keep tabs on which lane line (left or right) is "more successfully found" for the current frame. This could be done by tracking total number of pixels found, or by the height (across the warped lane image) that found lane pixels span. Since lane lines in reality are supposed to be pretty close to parallel, I could imagine using the polynomial fit of the left lane line and using it as a search window for both lane lines if the left line is stronger than the right, or vice versa.