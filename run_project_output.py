from process_image import ProcessImage
import lane_line_finding
from moviepy.editor import VideoFileClip


if __name__ == "__main__":
    cal = lane_line_finding.Calibration.create_calibration()
    pt = lane_line_finding.PerspectiveTransform.create_perspective_transform(0, 0, 0)

    video_name = "project_video.mp4"
    # video_name = "00_crop_project_video.mp4"

    pi = ProcessImage(pt, cal, smooth_window=5)
    project_output = '170520_0.5.0-project-output_{}'.format(video_name)

    clip = VideoFileClip("../" + video_name)
    project_clip = clip.fl_image(pi.run)
    project_clip.write_videofile(project_output, audio=False)
    print("Output video: {}".format(project_output))
