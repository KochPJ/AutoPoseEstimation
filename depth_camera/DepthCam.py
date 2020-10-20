import pyrealsense2 as rs
import numpy as np
import cv2
import time

class DepthCam:
    def __init__(self, fps=6, height=480, width=640):
        self.height = height
        self.width = width
        self.fps = fps


        self.pipe = None
        self.config = None
        self.profile = None
        self.align = None
        self.colorizer = None
        self.depth_sensor = None
        self.init_pipeline()
        self.repairing = False

    def init_pipeline(self):
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.profile = self.pipe.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        self.depth_sensor = self.profile.get_device().first_depth_sensor()

        self.color_sensor = self.profile.get_device().query_sensors()[1]
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.color_sensor.set_option(rs.option.enable_auto_exposure, False)
        self.color_sensor.set_option(rs.option.exposure, 200.0)
        self.color_sensor.set_option(rs.option.white_balance, 3200.0)

    def stream(self, fps = 30, show_color=True, show_depth=False, show_depth_color=False, show_added=False):
        while True:

            return_depth_colorized = False
            if show_depth_color or show_added:
                return_depth_colorized = True
            out = self.get_frames(return_depth_colorized=return_depth_colorized)
            image = out['image']
            depth = out['depth']

            if show_color:
                show = image
            if show_depth:

                show = np.array(depth/2000*255, dtype=np.uint8)
            if show_depth_color:
                show = out['depth_colorized']
            if show_added:
                depth_colorized = out['depth_colorized']
                show = cv2.addWeighted(image, 0.7, depth_colorized, 0.3, 0)

                cv2.imshow('stream', cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == 27:
                break

            time.sleep(1/fps)

        self.pipe.stop()

    def get_frames(self, return_intrinsics=False,
                   return_depth_colorized=False,
                   with_repair=True,
                   return_first_try=False,
                   return_first=False,
                   secure_image=False,
                   check_state=False):

        first_try = True
        while True:

            if secure_image:
                # check if the current state is not broken and only a old image is in the stack
                t = time.time()+1
                while time.time()<t:
                    success, frames = self.pipe.try_wait_for_frames()
                    if not success:
                        break
            else:
                success, frames = self.pipe.try_wait_for_frames()

            if success:
                try:

                    # checks if the next images are ready
                    if check_state:
                        check = True
                        for _ in range(10):
                            check, f = self.pipe.try_wait_for_frames()
                            if not check:
                                return None, False


                    frames = self.align.process(frames)
                    out = {'frames': frames}


                    if return_intrinsics:
                        intr = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
                        out['intr'] = intr
                        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                        out['depth_scale'] = depth_scale


                    image = np.array(frames.get_color_frame().get_data())
                    depth = np.array(frames.get_depth_frame().get_data())
                    out['image'] = image
                    out['depth'] = depth

                    if return_depth_colorized:
                        depth_colorized = np.array(self.colorizer.colorize(frames.get_depth_frame()).get_data())
                        out['depth_colorized'] = depth_colorized

                    if with_repair:
                        self.repairing = False

                    if return_first_try:
                        return out, first_try
                    else:
                        return out

                except:
                    success = False

            if not success:
                print('failed to get images')
                first_try = False

                if return_first:
                    return None, first_try

                if with_repair:
                    self.repairing = True
                    while True:
                        try:
                            self.init_pipeline()
                            break
                        except:
                            print('init pipelie failed, trying again')
                            continue

                else:
                    while self.repairing:
                        time.sleep(1)

                time.sleep(1)

    def get_intrinsics(self):
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        return intr
    def get_depth_scale(self):
        depth_scale = self.depth_sensor.get_depth_scale()
        return depth_scale



if __name__ == '__main__':
    print('create depth cam')
    #DC = DepthCam()
    DC = DepthCam(fps=15, height=720, width=1280)
    intr = DC.get_intrinsics()
    print(intr)
    print(DC.get_depth_scale()*1000)
    DC.stream(show_color=True, show_added=True, show_depth=True)
