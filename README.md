<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. Camera Calibration</a></li>
<li><a href="#sec-2">2. Pipeline test images</a></li>
<li><a href="#sec-3">3. Pipeline video</a></li>
<li><a href="#sec-4">4. Discussion</a></li>
</ul>
</div>
</div>

# Camera Calibration<a id="sec-1" name="sec-1"></a>

**1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.**

The code for calibrating the camera and undistorting images are shown in `lane_lines.Chessboard` and `lane_lines.Calibration`. An example which runs the code is shown in `test_lane_lines.CalibrationTest.test_calibrate_camera_from_filename()`.

The calibration method works by analysing images of chessboards taken from different angles and distances (these were provided and are in `camera_cal`). The OpenCV function `findChessboardCorners` then detects the edges between the chess board squares. The sets of these detected corners in multiple images are then fed into the OpenCV `calibrateCamera` function which, using the fact that in reality the corners are equally spaced, calculates and returns the calibration and distortion coefficients.

![img](./writeup_images/calibration_undistorted.png)

# Pipeline test images<a id="sec-2" name="sec-2"></a>

**1. Provide an example of a distortion-corrected image**

Below is an image taken using the dashcam, before and after undistorting.

![img](./writeup_images/dashcam_undistorted.png)

**2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.**

![img](../debug_video_stills/0730.jpg)

I used several different values for thresholds and kernel sizes for the different sobel transformations, with the best combination being shown in the debug plot. The is code is in `lane_line_finding.Sobel`.

I also investigated the HLS colour space. The S (saturation) channel was very effective at identifying the yellow lane lines as can be seen in the debug video still. The code for converting to HLS, extracting S channel and applying a threshold is in `lane_line_finding.hls_select`.

The sobel outputs and the S colour space were combined with an and, so that if either method identified a pixel as possibly being part of a lane line, they were included in the output.

**3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.**

The code for calculating and applying the perspective transform warp is in `lane_line_finding.PerspectiveTransform`. 

![img](./writeup_images/perspective_transform.png)

**4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?**

The functions used to calculate the second order polynomial fit for the left and right lanes are `udacity.calculate_first_frame` and `udacity.calculate_subsequent_frame`. 

The first of these functions is used for the first frame and does this by creating a histogram of the bottom half of the perspective transform. It then uses the 2 maxima as the starting point of the left and right lanes. It then moves a sliding window in steps up the image to follow the lane line, it finally calculates the second order polynomial using the NumPy function `polyfit`. The output of this step is calculated for each debug frame (panel "Output of fitting from scratch (not used)"), though after the first frame the next method is used.

The subsequent frames are processed using `udacity.calculate_subsequent_frame` and the output is shown on the "Fitting based on previous fit" panel. This uses the previous fit as a starting point and so is much faster to compute.

**5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.**

The radius of curvature and position of the vehicle were calculated by the `udacity.calculate_curvature_and_position` function. The polynomial fits previously detected were converted into meters. 

\(R_{curve} = \frac{(1+(2Ay + B)^2)^{3/2}}{\abs{2A}}\)
$R_{curve} = \frac{(1+(2Ay + B)^2)^{3/2}}{\abs{2A}}$

**6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.**

# Pipeline video<a id="sec-3" name="sec-3"></a>

# Discussion<a id="sec-4" name="sec-4"></a>
