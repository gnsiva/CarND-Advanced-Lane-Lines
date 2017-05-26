import lane_line_finding
import udacity
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ProcessImage:
    def __init__(self, pt, cal, margin=100, smooth_window=5, n_curvatures=100):
        """
        
        :param pt: 
        :param cal: 
        :param margin: 
        :param smooth_window: Over how many frames to smooth the fit, if None no smoothing is done
        :param n_curvatures: Maximum number of previous curvatures to show in the debug plot  
        """
        self.left_fit = []
        self.right_fit = []
        self.last_left_fit = None
        self.last_right_fit = None
        self.curvatures = None

        self.margin = margin
        self.M = pt.M
        self.pt = pt
        self.cal = cal
        self.smooth_window = smooth_window
        self.max_curvatures = n_curvatures

    def is_anomaly(self, left_fit, right_fit, limit=0.25):
        """limit is between 0 and 1, 0 is the same, 1 is infinitely different"""
        def f(new_value, current, limit):
            difference = np.abs((new_value - current) / (new_value + current)).mean()
            if difference > limit:
                return True

        # current here means the last value, could also be the average
        if f(left_fit, self.left_fit[-1], limit):
            return True
        elif f(right_fit, self.right_fit[-1], limit):
            return True
        else:
            return False

    @staticmethod
    def convert_to_image_dumb():
        fn = "del_temp_file.jpeg"
        plt.savefig(fn)
        return mpimg.imread(fn)

    @staticmethod
    def convert_to_image_smart():
        figure = plt.gcf()

        # remove anti aliasing, might not be necessary
        matplotlib.rcParams['text.antialiased'] = False
        for ax in figure.axes:
            plt.setp(
                [ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()],
                antialiased=False)

        # draw the figure
        figure.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        return data.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    def update_left_right_fit_smoothing_reset(self, left_fit, right_fit, anomaly_limit=0.75):
        if len(self.left_fit) == 0:
            self.last_left_fit = left_fit
            self.last_right_fit = right_fit

            self.left_fit.append(left_fit)
            self.right_fit.append(right_fit)
        else:
            if not self.is_anomaly(left_fit, right_fit, limit=anomaly_limit):
                self.left_fit.append(left_fit)
                self.right_fit.append(right_fit)

                if len(self.left_fit) > self.smooth_window:
                    self.left_fit = self.left_fit[1:]
                    self.right_fit = self.right_fit[1:]

                    weights = np.linspace(0, 1, len(self.left_fit)) + 0.5
                    self.last_left_fit, self.last_right_fit = \
                        [np.average(x, axis=0, weights=weights) for x in [self.left_fit, self.right_fit]]
                else:
                    self.last_left_fit = left_fit
                    self.last_right_fit = right_fit

    def run(self, img):
        transformed, pipeline_stages = lane_line_finding\
            .multithreaded_pipeline(img, self.pt, cal=self.cal, return_arrays=True)

        last_fit = None
        if self.last_left_fit is not None:
            last_fit = [self.last_left_fit, self.last_right_fit]

        output_img, fit = udacity.project_plot(img, transformed, margin=100, M=self.M, last_fit=last_fit)
        self.last_left_fit = fit[0]
        self.last_right_fit = fit[1]
        return output_img

    def run_debug(self, img):
        # detect lane lines and transform
        transformed, pipeline_stages = lane_line_finding\
            .multithreaded_pipeline(img, self.pt, cal=self.cal, return_arrays=True)

        figure = plt.figure(figsize=[16, 18])

        last_fit = None
        if self.last_left_fit is not None:
            last_fit = [self.last_left_fit, self.last_right_fit]

        last_fit, prev_curvatures = udacity.debug_plot(
            img, pipeline_stages, transformed, margin=100, M=self.M, last_fit=last_fit, prev_curvatures=self.curvatures)

        self.curvatures = prev_curvatures[-(self.max_curvatures+1):]

        # figimg = self.convert_to_image_dumb()
        figimg = self.convert_to_image_smart()

        plt.close(figure)

        # fitting the lane lines
        # self.update_left_right_fit_smoothing_reset(*last_fit)  # this would add smoothing and reset on dodgy fit
        self.last_left_fit = last_fit[0]
        self.last_right_fit = last_fit[1]

        return figimg


if __name__ == "__main__":
    from moviepy.editor import VideoFileClip
    import pickle

    with open("../170520_pt_cal2.p", "rb") as ifile:
        pt, cal = pickle.load(ifile)

    video_name = "project_video.mp4"
    # video_name = "02_crop_project_video.mp4"

    pi = ProcessImage(pt, cal, smooth_window=5)
    project_output = '170520_0.3.4-no-smoothing_{}'.format(video_name)
    clip = VideoFileClip("../" + video_name)
    project_clip = clip.fl_image(pi.run_debug)
    project_clip.write_videofile(project_output, audio=False)
    print("Output video: {}".format(project_output))
