import json
import os

import cv2
import imageio
import numpy as np
import torch.utils.data as data
from lib.config import cfg
from lib.utils import base_utils
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from PIL import Image
from plyfile import PlyData


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots["cams"]
        # load camera intrinsic and extrinsic parameters for each view

        num_cams = len(self.cams["K"])
        # K denotes the camera intrinsic, len() return the number of cameras
        test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        # get test_view camera index
        view = cfg.training_view if split == "train" else test_view
        # view store the train view, in the yaml config for CoreView_313 sequence, the train_view is
        #  [0, 6, 12, 18]
        if len(view) == 0:
            view = [0]
        # if the number of training views is zero, then set the #0 view for training ???

        # prepare input images
        i = 0
        i = i + cfg.begin_i
        i_intv = cfg.i_intv
        ni = cfg.ni
        if cfg.test_novel_pose:
            i = (i + cfg.ni) * i_intv
            ni = cfg.novel_pose_ni
            if self.human == "CoreView_390":
                i = 0
        # sample the training frames by setting the frame interval (i_intv = 6)
        # and number of frames (ni = 60) with the time offset (begin_i)
        # for the training set 313, if sample the frame start from #0
        # the training data should be

        #           view#0, view#6, view#12, view#18
        # frameno.  0001    0001,   0001,    0001
        #           0007    0007,   0007,    0007
        #           0013    0013,   0013,    0013
        # ........................................
        #           0355,   0355,   0355,    0355
        # this process shows how to extract sets of images from multi-view settings from a video
        self.ims = np.array(
            [
                np.array(ims_data["ims"])[view]
                for ims_data in annots["ims"][i : i + ni * i_intv][::i_intv]
            ]
        ).ravel()  # flatten the string array to a 1-d array
        self.cam_inds = np.array(
            [
                np.arange(len(ims_data["ims"]))[view]
                for ims_data in annots["ims"][i : i + ni * i_intv][::i_intv]
            ]
        ).ravel()  # flatten the view index to a 1-d array
        self.num_cams = len(view)
        # number of cameras for traininig
        # number of camera rays for 313 sequence, it is 1024
        self.nrays = cfg.N_rand

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, "mask", self.ims[index])[:-4] + ".png"
        msk = imageio.imread(msk_path)
        msk = (msk != 0).astype(np.uint8)

        msk_path = (
            os.path.join(self.data_root, "mask_cihp", self.ims[index])[:-4] + ".png"
        )
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)

        msk = (msk | msk_cihp).astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, cfg.vertices, "{}.npy".format(i))
        xyz = np.load(vertices_path).astype(np.float32)  # (6890, 3)
        nxyz = np.zeros_like(xyz).astype(np.float32)  # (6890, 3) all zeros

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)  # (1, 3)
        max_xyz = np.max(xyz, axis=0)  # (1, 3)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)  # (2, 3)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params, "{}.npy".format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params["Rh"]  # (1, 3)
        # Converts a rotation matrix to a rotation vector or vice versa.
        # here convert rotation vector to a rotation matrix.
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)  # (3, 3)
        Th = params["Th"].astype(np.float32)  # (1, 3)
        xyz = np.dot(xyz - Th, R)  # (6890, 3)
        # transform the smpl vertices from the world coordinate to the smple coordinate

        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)
        #
        # obtain the bounds for coord construction
        # obtain the bounding box after the transformation augmentation
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)  # (6890,3)
        nxyz = nxyz.astype(np.float32)  # (6890,3) all zeros
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)  # (6890, 6)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)
        # voxelized smpl model coordinates

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans

    def __getitem__(self, index):
        # get one image from the ims array
        img_path = os.path.join(self.data_root, self.ims[index])
        # normalize the values to 0~1
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        # resize the image to 1024, 1024
        img = cv2.resize(img, (1024, 1024))
        # get the image mask according to the index
        msk = self.get_mask(index)

        # get the camera index of this image
        cam_ind = self.cam_inds[index]
        # get the camera intrinsic parameter K (focal length and optical center)
        K = np.array(self.cams["K"][cam_ind])
        # get the camera intrinsic parameter D
        D = np.array(self.cams["D"][cam_ind])

        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = np.array(self.cams["R"][cam_ind])
        T = np.array(self.cams["T"][cam_ind]) / 1000.0

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K[:2] = K[:2] * cfg.ratio

        if self.human in ["CoreView_313", "CoreView_315"]:
            i = int(os.path.basename(img_path).split("_")[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i
        (
            feature,  # (6890, 6)
            coord,  #
            out_sh,
            can_bounds,
            bounds,
            Rh,
            Th,
            center,
            rot,
            trans,
        ) = self.prepare_input(i)

        (
            rgb,
            ray_o,
            ray_d,
            near,
            far,
            coord_,
            mask_at_box,
        ) = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, can_bounds, self.nrays, self.split
        )
        acc = if_nerf_dutils.get_acc(coord_, msk)

        ret = {
            "feature": feature,
            "coord": coord,
            "out_sh": out_sh,
            "rgb": rgb,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "near": near,
            "far": far,
            "acc": acc,
            "mask_at_box": mask_at_box,
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        i = index // self.num_cams
        if cfg.test_novel_pose:
            i = 0
        meta = {
            "bounds": bounds,
            "R": R,
            "Th": Th,
            "center": center,
            "rot": rot,
            "trans": trans,
            "i": i,
            "frame_index": frame_index,
            "cam_ind": cam_ind,
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
