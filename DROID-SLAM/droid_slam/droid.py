import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()

        self.args = args
        self.disable_vis = args.disable_vis
        self.visualizer = None

        self.load_weights(args.weights)

        # store images, depth, poses, intrinsics
        self.video = DepthVideo(
            args.image_size,
            args.buffer,
            stereo=args.stereo
        )

        # filter incoming frames so that there is enough motion
        # IMU rotation prior weight 추가
        imu_rotation_weight = getattr(args, "imu_rotation_weight", 1.0)

        self.filterx = MotionFilter(
            self.net,
            self.video,
            thresh=args.filter_thresh,
            imu_rotation_weight=imu_rotation_weight
        )

        # frontend process
        self.frontend = DroidFrontend(
            self.net,
            self.video,
            self.args
        )

        # backend process
        self.backend = DroidBackend(
            self.net,
            self.video,
            self.args
        )

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(
                target=visualization_fn,
                args=(self.video, None)
            )
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            self.net,
            self.video
        )

    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)

        self.net = DroidNet()

        state_dict = OrderedDict([
            (k.replace("module.", ""), v)
            for (k, v) in torch.load(weights).items()
        ])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None, imu_prior=None):
        """
        main thread - update map

        변경점:
        - demo.py에서 전달받은 imu_prior를 MotionFilter.track()으로 넘긴다.
        - DROID-SLAM 원본 frontend/backend 실행 흐름은 유지한다.
        """

        with torch.no_grad():
            self.filterx.track(
                tstamp,
                image,
                depth,
                intrinsics,
                imu_prior=imu_prior
            )

            counter = self.video.counter.value
            warmup = getattr(self.args, "warmup", 8)

            # local bundle adjustment
            if counter == warmup:
                self.frontend()

            # global bundle adjustment
            elif counter > warmup:
                self.frontend()
                self.backend(2)

            if self.visualizer is not None:
                self.visualizer()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()