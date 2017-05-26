from process_image import ProcessImage
import lane_line_finding
import glob
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


def create_calibration():
    cal_fns = glob.glob("camera_cal/calibration??.jpg")
    cal = lane_line_finding.Calibration(nx=9, ny=6)
    cal.calibrate_camera_from_filenames(cal_fns)
    return cal


def create_perspective_transform():
    # Do transformation
    src_coords = np.float32([(686.7, 445.6), [1100, 720], [207, 720], (601.0, 445.6)])

    l_offset = 150
    r_offset = 300
    v_offset = 0

    img = mpimg.imread("test_images/straight_lines2.jpg")
    dst_coords = lane_line_finding.PerspectiveTransform \
        .determine_dst_coords(img, l_offset, r_offset, v_offset)

    pt = lane_line_finding.PerspectiveTransform(src_coords, dst_coords)
    return pt


if __name__ == "__main__":
    cal = create_calibration()
    pt = create_perspective_transform()

    # video_name = "project_video.mp4"
    video_name = "02_crop_project_video.mp4"

    pi = ProcessImage(pt, cal, smooth_window=5)
    project_output = '170520_0.2.0-project-output_{}'.format(video_name)

    clip = VideoFileClip("../" + video_name)
    project_clip = clip.fl_image(pi.run)
    project_clip.write_videofile(project_output, audio=False)
    print("Output video: {}".format(project_output))
