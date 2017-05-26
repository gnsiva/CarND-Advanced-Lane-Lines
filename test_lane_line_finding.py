import unittest
import lane_line_finding
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import matplotlib.image as mpimg
from os.path import join

RUN_PLOTTING_TESTS = True
proj_dir = "/home/gns/repos/sdc/CarND-Advanced-Lane-Lines/"


class ChessboardTest(unittest.TestCase):
    def test_init(self):
        fn = "camera_cal/calibration7.jpg"

        cb = lane_line_finding.Chessboard(fn, proj_dir, 9, 6)
        self.assertTrue(isinstance(cb.img, np.ndarray))

    def test_get_chessboard_corners_doesnt_find_them(self):
        fn = "camera_cal/calibration4.jpg"

        cb = lane_line_finding.Chessboard(fn, proj_dir, 9, 9)
        corners = cb.get_chessboard_corners()
        self.assertEqual(corners, None)

    def test_get_chessboard_corners_does_find_them(self):
        fn = "camera_cal/calibration7.jpg"

        cb = lane_line_finding.Chessboard(fn, proj_dir, 9, 6)
        corners = cb.get_chessboard_corners()
        self.assertTrue(isinstance(corners, np.ndarray))

    def test_plot_chessboard_corners(self):
        
        fn = "camera_cal/calibration7.jpg"

        cb = lane_line_finding.Chessboard(fn, proj_dir, 9, 6)
        cb.get_chessboard_corners()
        cb.plot_chessboard_corners()
        
        if RUN_PLOTTING_TESTS:
            plt.show()

    def test_get_outer_corners(self):
        fn = "camera_cal/calibration7.jpg"

        cb = lane_line_finding.Chessboard(fn, proj_dir, 9, 6)
        cb.get_chessboard_corners()
        src_coords, dst_coords = cb.get_outer_corners()

        plt.imshow(cb.img)
        plt.scatter([xy[0] for xy in src_coords], [xy[1] for xy in src_coords])
        plt.scatter([xy[0] for xy in dst_coords], [xy[1] for xy in dst_coords], color='r')
        
        if RUN_PLOTTING_TESTS:
            plt.show()


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.cal_fns = glob.glob("camera_cal/calibration??.jpg")

    def test_calibrate_camera_from_filenames(self):
        cal = lane_line_finding.Calibration(9, 6)
        cal.calibrate_camera_from_filenames(self.cal_fns, proj_dir)

        self.assertIsNotNone(cal.dist)
        self.assertIsNotNone(cal.mtx)

    def test_undistort_image_from_filename(self):
        fn = "camera_cal/calibration4.jpg"

        cal = lane_line_finding.Calibration(9, 6)
        cal.calibrate_camera_from_filenames(self.cal_fns, proj_dir)

        img = cal.undistort_image_from_filename(fn, proj_dir)

        if RUN_PLOTTING_TESTS:
            f, axes = plt.subplots(1, 2, figsize=[16, 6])
            axes[0].imshow(cv2.imread(fn))
            axes[0].set_title("Original image")

            axes[1].imshow(img)
            axes[1].set_title("Undistorted image")
            plt.tight_layout()
            plt.show()

    def test_undistort_image_from_filename_ProperImage(self):
        fn = "quiz_img/signs_vehicles_xygrad.png"
        cal = lane_line_finding.Calibration(9, 6)
        cal.calibrate_camera_from_filenames(self.cal_fns, proj_dir)

        img = cal.undistort_image_from_filename(fn, proj_dir)

        if RUN_PLOTTING_TESTS:
            f, axes = plt.subplots(1, 2, figsize=[16, 6])
            axes[0].imshow(mpimg.imread(fn))
            axes[0].set_title("Original image")

            axes[1].imshow(img)
            axes[1].set_title("Undistorted image")
            plt.tight_layout()
            plt.show()


class PerspectiveTransformTest(unittest.TestCase):

    def test_transform_image(self):
        # calibrate the camera
        cal_fns = glob.glob("camera_cal/calibration??.jpg")
        cal = lane_line_finding.Calibration(9, 6)
        cal.calibrate_camera_from_filenames(cal_fns, proj_dir)

        # open a chessboard image
        fn = "camera_cal/calibration10.jpg"
        cb = lane_line_finding.Chessboard(fn, proj_dir, 9, 6)

        # undistort image
        cb.img = cal.undistort_image(cb.img)

        # find outer corners of chessboard
        cb.get_chessboard_corners()
        src_coords, dst_coords = cb.get_outer_corners()

        # do a perspective transform
        pt = lane_line_finding.PerspectiveTransform(src_coords, dst_coords)
        trans_img = pt.transform_image(cb.img)

        if RUN_PLOTTING_TESTS:
            f, axes = plt.subplots(1, 2, figsize=[15, 5])
            # plot original (undistorted) image
            axes[0].imshow(cb.img)
            axes[0].scatter(src_coords[:, 0], src_coords[:, 1], label='src_coords')
            axes[0].scatter(dst_coords[:, 0], dst_coords[:, 1], label='dst_coords')
            axes[0].legend()
            axes[0].set_title("Original image (undistorted)")

            axes[1].imshow(trans_img)
            axes[1].set_title("Transformed image")
            axes[1].imshow(trans_img)

            plt.tight_layout()
            plt.show()

    def test_transform_image_on_road_image(self):
        # Get image
        img = mpimg.imread("test_images/straight_lines2.jpg")
        # img = mpimg.imread("test_images/test2.jpg")

        # Done before undistorting camera stupidly
        left_bottom = (307.0, 649.1)
        right_bottom = (1013.8, 649.1)
        left_top = (615.7, 434.3)
        right_top = (668.3, 434.3)

        # Do transformation
        src_coords = np.array([right_top, right_bottom, left_bottom, left_top])

        l_offset = 200
        r_offset = 100
        v_offset = 0
        dst_coords = lane_line_finding.PerspectiveTransform \
            .determine_dst_coords(img, l_offset, r_offset, v_offset)

        pt = lane_line_finding.PerspectiveTransform(src_coords, dst_coords)
        trans_img = pt.transform_image(img)

        if RUN_PLOTTING_TESTS:
            f, axes = plt.subplots(1, 2, figsize=[15, 5])
            # plot original (undistorted) image
            pt.plot_coords(img=img, ax=axes[0])
            axes[0].legend()
            axes[0].set_title("Original image (undistorted)")

            axes[1].imshow(trans_img)
            axes[1].set_title("Transformed image")
            axes[1].imshow(trans_img)
            
            plt.tight_layout()
            plt.show()


class SobelTest(unittest.TestCase):
    def test_abs_sobel_threshold(self):
        img = mpimg.imread(join(proj_dir, "quiz_img/signs_vehicles_xygrad.png"))
        sobel = lane_line_finding.Sobel()
        for orient in ['x', 'y']:
            ast_img = sobel.abs_sobel_threshold(
                img, orient=orient, sobel_kernel=3, thresh=(0, 255), plot=False)
            self.assertAlmostEqual(ast_img.mean(), 1)

            ast_img = sobel.abs_sobel_threshold(
                img, orient=orient, sobel_kernel=3, thresh=(20, 200), plot=False)
            self.assertLess(ast_img.mean(), 1)
            self.assertAlmostEqual(np.median(ast_img), 0)

        if RUN_PLOTTING_TESTS:
            ast_img = sobel.abs_sobel_threshold(
                img, orient=orient, sobel_kernel=3, thresh=(20, 200), plot=True)
            plt.show()

    def test_mag_threshold(self):
        img = mpimg.imread(join(proj_dir, "quiz_img/signs_vehicles_xygrad.png"))
        sobel = lane_line_finding.Sobel()

        ast_img = sobel.abs_sobel_threshold(
            img, sobel_kernel=3, thresh=(0, 255), plot=False)
        self.assertAlmostEqual(ast_img.mean(), 1)

        ast_img = sobel.mag_threshold(
            img, sobel_kernel=3, thresh=(20, 200), plot=False)
        self.assertLess(ast_img.mean(), 1)
        self.assertAlmostEqual(np.median(ast_img), 0)

        if RUN_PLOTTING_TESTS:
            ast_img = sobel.mag_threshold(
                img, sobel_kernel=3, thresh=(20, 200), plot=True)
            plt.show()

    def test_dir_threshold(self):
        # TODO - this test needs asserts
        img = mpimg.imread(join(proj_dir, "quiz_img/signs_vehicles_xygrad.png"))
        sobel = lane_line_finding.Sobel()

        sobel.dir_threshold(img, thresh=(0.7, 1.3))
        if RUN_PLOTTING_TESTS:
            plt.show()

    def test_plot_all(self):
        # calibrate image
        raw_img = mpimg.imread("quiz_img/signs_vehicles_xygrad.png")
        cal_fns = glob.glob("camera_cal/calibration??.jpg")
        
        cal = lane_line_finding.Calibration(nx=9, ny=6)
        cal.calibrate_camera_from_filenames(cal_fns)
        img = cal.undistort_image(raw_img)
        
        sobel = lane_line_finding.Sobel()
        abs_x = sobel.abs_sobel_threshold(img, orient='x', sobel_kernel=9, thresh=(20, 200), plot=False)
        abs_y = sobel.abs_sobel_threshold(img, orient='y', sobel_kernel=9, thresh=(20, 200), plot=False)
        mag = sobel.mag_threshold(img, sobel_kernel=11, thresh=(40, 200), plot=False)
        direction = sobel.dir_threshold(img, sobel_kernel=11, thresh=(0.7, 1.3), plot=False)

        combined = sobel.plot_all(abs_x, abs_y, mag, direction)
        if RUN_PLOTTING_TESTS:
            plt.show()


if __name__ == "__main__":
    unittest.main()
