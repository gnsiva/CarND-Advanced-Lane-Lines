import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import lane_line_finding


def calculate_first_frame(binary_warped, margin=100, plot=False, ax=None):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = margin
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        if plot:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    if plot:
        if not ax:
            ax = plt
        # draw left lane in red and right lane in blue
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')

    return left_fitx, right_fitx, ploty, left_fit, right_fit#, leftx, rightx


def calculate_subsequent_frame(binary_warped, left_fit, right_fit, margin=100):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit#, leftx, rightx


def plot_subsequent_frame(out_img, left_fitx, right_fitx, ploty, margin=100, ax=None):
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    window_img = np.zeros_like(out_img)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ax.imshow(result)
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)


def calculate_curvature_and_position(ploty, leftx, rightx, transformed):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    # Now our radius of curvature is in meters
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # calculate vehicle position
    # find average x and y position of the transformed lanes (in meters)
    left_position_av = (leftx * xm_per_pix).mean()
    right_position_av = (rightx * xm_per_pix).mean()

    # find the midpoint of the two lanes and the center of the image
    lanes_midpoint = left_position_av + ((right_position_av - left_position_av) / 2)
    image_midpoint = transformed.shape[1] / 2 * xm_per_pix
    # the difference is the vehicle position
    position = image_midpoint - lanes_midpoint

    return (left_curverad, right_curverad), position


def unwarp_draw_line(binary_warped, left_fitx, right_fitx, ploty, M, img, ax=None, curvature=None, position=None):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(M), (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    x = 30
    y = 0
    font_size = 0.8
    colour = (255, 255, 255)
    thickness = 2
    if curvature is not None:
        y += 40
        message = "Radius of curvature = {:06.0f} m".format(curvature)
        cv2.putText(result, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, thickness)
    if position is not None:
        y += 40
        message = "Vehicle position = {:+0.3f} m".format(position)
        cv2.putText(result, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, thickness)

    if not ax:
        ax = plt
    ax.imshow(result)
    return result


def project_plot(original_img, transformed, margin, M, last_fit=None):
    import udacity
    # calculate fit
    if last_fit is not None:
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = \
            udacity.calculate_subsequent_frame(transformed, last_fit[0], last_fit[1], margin=margin)
    else:
        left_fitx, right_fitx, ploty, left_fit, right_fit = \
            udacity.calculate_first_frame(transformed, margin, plot=False)

    # determine curvatures
    curvature, position = calculate_curvature_and_position(
        ploty, leftx=left_fitx, rightx=right_fitx, transformed=transformed)
    av_curvature = (curvature[0] + curvature[1]) / 2

    # plot output image
    output_img = udacity.unwarp_draw_line(
        transformed, left_fitx, right_fitx, ploty, M, original_img, curvature=av_curvature, position=position)
    fit = [left_fit, right_fit]
    return output_img, fit


def debug_plot(original_img, pipeline_stages, transformed, margin, M, last_fit=None, prev_curvatures=None):
    import udacity
    grid_shape = (6, 3)
    label_loc = 4
    axes = []

    # Sobel plots
    ax = plt.subplot2grid(grid_shape, (0, 0))
    key = "sobel_abs_x"
    ax.imshow(pipeline_stages[key], cmap='gray')
    ax.add_artist(AnchoredText("Sobel absolute x", loc=label_loc))
    axes.append(ax)

    ax = plt.subplot2grid(grid_shape, (0, 1))
    key = "sobel_abs_y"
    ax.imshow(pipeline_stages[key], cmap='gray')
    ax.add_artist(AnchoredText("Sobel absolute y", loc=label_loc))
    axes.append(ax)

    ax = plt.subplot2grid(grid_shape, (0, 2))
    key = "sobel_abs_x_abs_y"
    ax.imshow(pipeline_stages[key], cmap='gray')
    ax.add_artist(AnchoredText("Sobel absolute x and y combined", loc=label_loc))
    axes.append(ax)

    ax = plt.subplot2grid(grid_shape, (1, 0))
    key = "sobel_mag"
    ax.imshow(pipeline_stages[key], cmap='gray')
    ax.add_artist(AnchoredText("Sobel magnitude", loc=label_loc))
    axes.append(ax)

    ax = plt.subplot2grid(grid_shape, (1, 1))
    key = "sobel_direction"
    ax.imshow(pipeline_stages[key], cmap='gray')
    ax.add_artist(AnchoredText("Sobel direction", loc=label_loc))
    axes.append(ax)

    ax = plt.subplot2grid(grid_shape, (1, 2))
    key = "sobel_mag_direction"
    ax.imshow(pipeline_stages[key], cmap='gray')
    ax.add_artist(AnchoredText("Sobel magnitude and direction", loc=label_loc))
    axes.append(ax)

    # combined sobel and plot
    ax = plt.subplot2grid(grid_shape, (2, 0), rowspan=2, colspan=2)
    key = "sobel_hls_combined"
    ax.imshow(pipeline_stages[key], cmap='gray')
    ax.add_artist(AnchoredText("All sobel and HLS S channel combined", loc=label_loc))
    axes.append(ax)

    # HLS plot
    ax = plt.subplot2grid(grid_shape, (2, 2))
    key = "hls"
    ax.imshow(pipeline_stages[key], cmap='gray')
    ax.add_artist(AnchoredText("S channel from HLS", loc=label_loc))
    axes.append(ax)

    # plot perspective transform images
    # sliding window positions
    ax = plt.subplot2grid(grid_shape, (3, 2))
    left_fitx, right_fitx, ploty, left_fit, right_fit = \
        udacity.calculate_first_frame(transformed, margin, plot=True, ax=ax)
    ax.add_artist(AnchoredText("Output of fitting from scratch (not used)", loc=label_loc))
    axes.append(ax)

    # calculate fit
    if last_fit is not None:
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = \
            udacity.calculate_subsequent_frame(transformed, last_fit[0], last_fit[1], margin=margin)
    else:
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = \
            udacity.calculate_subsequent_frame(transformed, left_fit, right_fit, margin=margin)

    # curvatures plot
    current_curvatures, position = calculate_curvature_and_position(
        ploty, leftx=left_fitx, rightx=right_fitx, transformed=transformed)
    if prev_curvatures:
        prev_curvatures.append(current_curvatures)
    else:
        prev_curvatures = [current_curvatures]
    l = [x[0] for x in prev_curvatures]
    r = [x[1] for x in prev_curvatures]
    av = [(x[0]+x[1])/2 for x in prev_curvatures]

    ax = plt.subplot2grid(grid_shape, (5, 2))
    xs = (np.arange(len(l))*-1)[::-1]
    ax.plot(xs, l, label="left curvature", alpha=0.5)
    ax.plot(xs, r, label="right curvature", alpha=0.5)
    ax.plot(xs, av, label="average curvature", color='k')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Curvature")
    ax.set_ylim([0, 2000])
    ax.legend(loc=1)

    ax = plt.subplot2grid(grid_shape, (4, 2))
    udacity.plot_subsequent_frame(out_img, left_fitx, right_fitx, ploty, margin=margin, ax=ax)
    ax.add_artist(AnchoredText("Fitting based on previous fit", loc=label_loc))
    axes.append(ax)

    # plot final image
    ax = plt.subplot2grid(grid_shape, (4, 0), rowspan=2, colspan=2)
    udacity.unwarp_draw_line(
        transformed, left_fitx, right_fitx, ploty, M, original_img, ax=ax, curvature=av[-1], position=position)
    ax.add_artist(AnchoredText("Project output stream", loc=label_loc))
    axes.append(ax)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    last_fit = [left_fit, right_fit]
    return last_fit, prev_curvatures
