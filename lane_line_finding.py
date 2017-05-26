import cv2
import numpy as np
import matplotlib.image as mpimg
from os.path import join
import matplotlib.pyplot as plt
import skimage.io
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor as Executor


class Chessboard:
    def __init__(self, img_fn, dirname=None, nx=9, ny=6, imread=cv2.imread):
        if dirname:
            img_fn = join(dirname, img_fn)

        self.img = imread(img_fn)
        self.nx = nx
        self.ny = ny

        self.corners_detected = None
        self.corners = None

    def get_chessboard_corners(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray_img, (self.nx, self.ny), None)
        self.corners_detected = success
        self.corners = corners
        if success:
            return corners

    def plot_chessboard_corners(self):
        img_copy = self.img.copy()
        cv2.drawChessboardCorners(img_copy, (self.nx, self.ny), self.corners, self.corners_detected)
        plt.imshow(img_copy)

    def get_outer_corners(self, offset=100):

        # get indices of corner corners
        top_left = 0
        top_right = self.nx - 1
        bottom_left = self.nx * self.ny - self.nx
        bottom_right = self.nx * self.ny - 1

        # dest coords (you might not want this to be the corners of the image)
        dst_top_left = (offset, offset)
        dst_top_right = (self.img.shape[1] - offset, offset)
        dst_bottom_left = (offset, self.img.shape[0] - offset)
        dst_bottom_right = (self.img.shape[1] - offset, self.img.shape[0] - offset)
        dst_coords = np.array(
            [
                dst_top_right,
                dst_bottom_right,
                dst_bottom_left,
                dst_top_left
            ],
            dtype=np.float32
        )

        src_coords = []
        print(dst_coords.shape)
        print(self.corners.shape)
        print([top_right, bottom_right, bottom_left, top_left])

        for n in [top_right, bottom_right, bottom_left, top_left]:
            src_coords.append((self.corners[:, :, 0][n], self.corners[:, :, 1][n]))

        return np.array(src_coords, dtype=np.float32)[:, :, 0], dst_coords


class Calibration:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

        # result of calibration
        self.mtx = None
        self.dist = None

    def calibrate_camera(self, chessboards):
        objp = np.zeros((self.nx*self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        imgpoints = [cb.get_chessboard_corners() for cb in chessboards]
        objpoints = [objp for c in imgpoints]

        shape = chessboards[0].img.shape[:2][::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        self.dist = dist
        self.mtx = mtx

    def calibrate_camera_from_filenames(self, cb_fns, dirname=None):
        cbs = [Chessboard(fn, dirname, self.nx, self.ny) for fn in cb_fns]
        self.calibrate_camera(cbs)

    def undistort_image(self, img):
        if any([x for x in [self.dist, self.mtx] if x is None]):
            raise LookupError("calibrate_camera has not been run yet, or the calibration parameters are None")

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def undistort_image_from_filename(self, img_fn, dirname=None, imread=mpimg.imread):
        if dirname:
            img_fn = join(dirname, img_fn)

        return self.undistort_image(imread(img_fn))


class PerspectiveTransform:
    def __init__(self, src_coords, dst_coords):
        self.src_coords = self._check_fix_coords(src_coords)
        self.dst_coords = self._check_fix_coords(dst_coords)

        self.M = cv2.getPerspectiveTransform(self.src_coords, self.dst_coords)

    def transform_image(self, img):
        return cv2.warpPerspective(
            img, self.M, img.shape[:2][::-1], flags=cv2.INTER_LINEAR)

    def plot_coords(self, img=None, ax=None):
        if not ax:
            ax = plt
        if isinstance(img, np.ndarray):
            ax.imshow(img)
            
        ax.scatter(self.src_coords[:, 0], self.src_coords[:, 1], label="src_coords")
        ax.scatter(self.dst_coords[:, 0], self.dst_coords[:, 1], label="dst_coords")

    def plot_boundary(self, img, ax=None):
        mask = np.zeros_like(img, dtype=np.int8)
        vertices = np.array([self.dst_coords], dtype=np.int32)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = img.copy()
        masked_image[mask == 0] = 0
        if ax is None:
            ax = plt
        plt.imshow(masked_image)

    @staticmethod
    def determine_dst_coords(img, l_offset, r_offset, v_offset):
        dst_top_left = (l_offset, v_offset)
        dst_top_right = (img.shape[1] - r_offset, v_offset)
        dst_bottom_left = (l_offset, img.shape[0] - v_offset)
        dst_bottom_right = (img.shape[1] - r_offset, img.shape[0] - v_offset)
        dst_coords = np.array(
            [dst_top_right, dst_bottom_right, dst_bottom_left, dst_top_left],
            dtype=np.float32
        )
        return dst_coords
        
        
    @staticmethod
    def _check_fix_coords(coords):
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords, np.float32)
        elif not isinstance(coords.item(0), np.float32):
            coords = coords.astype(np.float32)
        return coords


class Sobel:
    def __init__(self, imread=skimage.io.imread):
        self.imread = imread

    @staticmethod
    def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255), plot=True):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/abs_sobel.max())

        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        if plot:
            plt.imshow(sbinary, cmap='gray')

        return sbinary

    @staticmethod
    def mag_threshold(img, sobel_kernel=3, thresh=(0, 255), plot=True):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobelxy = np.absolute(np.sqrt(sobelx**2 + sobely**2))

        scaled_sobel = np.uint8(255*abs_sobelxy/abs_sobelxy.max())

        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        if plot:
            plt.imshow(sbinary, cmap='gray')

        return sbinary

    @staticmethod
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2), plot=True):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_angle = np.abs(np.arctan(sobely/sobelx))

        sabinary = np.zeros_like(sobel_angle)
        sabinary[(sobel_angle >= thresh[0]) & (sobel_angle <= thresh[1])] = 1

        if plot:
            plt.imshow(sabinary, cmap='gray')

        return sabinary


    @staticmethod
    def calculate_masks(abs_x, abs_y, mag, direction, only_final=False):
        abs_x_abs_y = np.zeros_like(abs_x)
        abs_x_abs_y[(abs_x == 1) & (abs_y == 1)] = 1

        mag_direction = np.zeros_like(abs_x)
        mag_direction[(mag == 1) & (direction == 1)] = 1

        combined = np.zeros_like(abs_x)
        combined[((abs_x == 1) & (abs_y == 1)) | ((mag == 1) & (direction == 1))] = 1

        if only_final:
            return combined
        else:
            return abs_x_abs_y, mag_direction, combined

    @staticmethod
    def plot_all(abs_x, abs_y, mag, direction):
        # Calculate combined masks
        abs_x_abs_y, mag_direction, combined = Sobel.calculate_masks(
            abs_x, abs_y, mag, direction, only_final=False)

        # ================
        # Do the plotting
        axes = []
        plt.figure(figsize=(16, 16))

        # first row
        ax = plt.subplot2grid((4, 3), (0, 0))
        ax.imshow(abs_x, cmap='gray')
        ax.set_title("X gradient")
        axes.append(ax)

        ax = plt.subplot2grid((4, 3), (0, 1))
        ax.imshow(abs_y, cmap='gray')
        ax.set_title("Y gradient")
        axes.append(ax)

        ax = plt.subplot2grid((4, 3), (0, 2))
        ax.imshow(abs_x_abs_y, cmap='gray')
        ax.set_title("X and Y gradients combined")
        axes.append(ax)

        # second row
        ax = plt.subplot2grid((4, 3), (1, 0))
        ax.imshow(mag, cmap='gray')
        ax.set_title("Gradient magnitude")
        axes.append(ax)

        ax = plt.subplot2grid((4, 3), (1, 1))
        ax.imshow(direction, cmap='gray')
        ax.set_title("Gradient direction")
        axes.append(ax)

        ax = plt.subplot2grid((4, 3), (1, 2))
        ax.imshow(mag_direction, cmap='gray')
        ax.set_title("Magnitude and direction combined")
        axes.append(ax)

        # final figure
        ax = plt.subplot2grid((4, 3), (2, 0), colspan=3, rowspan=2)
        ax.imshow(combined, cmap='gray')
        ax.set_title("Combined")
        axes.append(ax)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        return combined


def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls_img[:, :, 2]
    s_thresholded = np.zeros_like(s_channel)
    s_thresholded[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return s_thresholded


def apply_mask(img, left_bottom=[133, 720], left_top=[577, 444], right_top=[717, 444], right_bottom=[1219, 720]):

    mask = np.zeros_like(img, dtype=np.int8)
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = img.copy()
    masked_image[mask == 0] = 0
    return masked_image


def multithreaded_pipeline(img, pt, cal=None, plot_stages=False, already_undistorted=False, return_arrays=False):
    """
    :type img: np.ndarray
    :type pt: PerspectiveTransform
    :type cal: Calibration
    :type plot_stages: bool
    :type already_undistorted: bool
    :param return_arrays: completely transformed array first, then a dictionary of in between stages.
    :type return_arrays: (np.ndarray, dict)
    :return: np.ndarray - transformed unless return_arrays is true
    """
    if not already_undistorted:
        img = cal.undistort_image(img)

    # masked_img = apply_mask(img)
    masked_img = img.copy()

    # Sobel
    with Executor() as executor:
        future_abs_x = executor.submit(
            lambda img, orient: Sobel().abs_sobel_threshold(
                img, orient=orient, sobel_kernel=9, thresh=(20, 220), plot=False), img.copy(), 'x')
        future_abs_y = executor.submit(
            lambda img, orient: Sobel().abs_sobel_threshold(
                img, orient=orient, sobel_kernel=9, thresh=(20, 220), plot=False), img.copy(), 'y')
        future_mag = executor.submit(
            lambda img: Sobel().mag_threshold(
                img, sobel_kernel=11, thresh=(40, 200), plot=False), img.copy())
        future_direction = executor.submit(
            lambda img: Sobel().dir_threshold(
                img, sobel_kernel=11, thresh=(0.7, 1.3), plot=False), img.copy())

        abs_x = future_abs_x.result()
        abs_y = future_abs_y.result()
        mag = future_mag.result()
        direction = future_direction.result()
    sobel_binary = Sobel().calculate_masks(abs_x, abs_y, mag, direction, only_final=True)

    # HLS
    hls_binary = hls_select(masked_img, thresh=(100, 255))

    # Combine them
    combined = np.zeros_like(sobel_binary)
    combined[(sobel_binary == 1) | (hls_binary == 1)] = 1

    # Perspective transform
    # transformed = pt.transform_image(apply_mask(combined))
    transformed = pt.transform_image(combined)

    if plot_stages:
        f, axes = plt.subplots(4, 1, figsize=[16, 20])
        axes[0].imshow(sobel_binary, cmap='gray')
        axes[0].set_title("sobel_binary")

        axes[1].imshow(hls_binary, cmap='gray')
        axes[1].set_title("hls_binary")

        axes[2].imshow(combined, cmap='gray')
        axes[2].set_title("sobel and hls combined")

        axes[3].imshow(transformed, cmap='gray')
        axes[3].set_title("perspective transformed")
        plt.tight_layout()

    if return_arrays:
        abs_x_abs_y, mag_direction, _ = Sobel().calculate_masks(abs_x, abs_y, mag, direction, only_final=False)
        return transformed, OrderedDict([
            ('sobel', sobel_binary),
            ('hls', hls_binary),
            ('sobel_abs_x', abs_x),
            ('sobel_abs_y', abs_y),
            ('sobel_abs_x_abs_y', abs_x_abs_y),
            ('sobel_mag', mag),
            ('sobel_direction', direction),
            ('sobel_mag_direction', mag_direction),
            ('sobel_hls_combined', combined)
        ])

    return transformed


def pipeline(img, pt, cal=None, plot_stages=False, already_undistorted=False, return_arrays=False):
    """
    :type img: np.ndarray
    :type pt: PerspectiveTransform
    :type cal: Calibration
    :type plot_stages: bool
    :type already_undistorted: bool
    :param return_arrays: completely transformed array first, then a dictionary of in between stages.
    :type return_arrays: (np.ndarray, dict)
    :return: np.ndarray - transformed unless return_arrays is true
    """
    if not already_undistorted:
        img = cal.undistort_image(img)

    # masked_img = apply_mask(img)
    masked_img = img.copy()

    # Sobel
    sobel = Sobel()
    abs_x = sobel.abs_sobel_threshold(masked_img, orient='x', sobel_kernel=9, thresh=(20, 200), plot=False)
    abs_y = sobel.abs_sobel_threshold(masked_img, orient='y', sobel_kernel=9, thresh=(20, 200), plot=False)
    mag = sobel.mag_threshold(masked_img, sobel_kernel=11, thresh=(40, 200), plot=False)
    direction = sobel.dir_threshold(masked_img, sobel_kernel=11, thresh=(0.7, 1.3), plot=False)
    sobel_binary = sobel.calculate_masks(abs_x, abs_y, mag, direction, only_final=True)

    # HLS
    hls_binary = hls_select(masked_img, thresh=(100, 255))

    # Combine them
    combined = np.zeros_like(sobel_binary)
    combined[(sobel_binary == 1) | (hls_binary == 1)] = 1

    # Perspective transform
    # transformed = pt.transform_image(apply_mask(combined))
    transformed = pt.transform_image(combined)

    if plot_stages:
        f, axes = plt.subplots(4, 1, figsize=[16, 20])
        axes[0].imshow(sobel_binary, cmap='gray')
        axes[0].set_title("sobel_binary")

        axes[1].imshow(hls_binary, cmap='gray')
        axes[1].set_title("hls_binary")

        axes[2].imshow(combined, cmap='gray')
        axes[2].set_title("sobel and hls combined")

        axes[3].imshow(transformed, cmap='gray')
        axes[3].set_title("perspective transformed")
        plt.tight_layout()

    if return_arrays:
        abs_x_abs_y, mag_direction, _ = sobel.calculate_masks(abs_x, abs_y, mag, direction, only_final=False)
        return transformed, OrderedDict([
            ('sobel', sobel_binary),
            ('hls', hls_binary),
            ('sobel_abs_x', abs_x),
            ('sobel_abs_y', abs_y),
            ('sobel_abs_x_abs_y', abs_x_abs_y),
            ('sobel_mag', mag),
            ('sobel_direction', direction),
            ('sobel_mag_direction', mag_direction),
            ('sobel_hls_combined', combined)
        ])

    return transformed

