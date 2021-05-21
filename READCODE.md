# Neural-Body Code Reading
This repository is forked from [[zju3dv/neuralbody](https://github.com/zju3dv/neuralbody)] and aims to record some comments in the code-reading.
## Pre-defined or pre-computed parameters for transformation 
There are two sets of transformation parameters.
The first set of transformation parameters is stored in the `dataset/params/*.npy` for each frame. It contains the rotation vector `Rh`, which is a 1x3 angle-axis rotation vector, and the translation transformation parameter `Th`, which is a 1x3 vector, this transformation parameters are utilized to transform vertices on a smpl model from the world coordinate to the smpl coordinate. 

The second set of transformation parameters is stored in the `dataset/annots.npy` data file. For each frame on each view, there will be a set of camera intrinsic and extrinsic parameters stored in variables `K`, `D`, `R` and `T`. Specifically, `K` denotes the camera intrinsic matrix (3-by-3), including focal length and optical center, `D` should be the distortion coefficients, (see the [`cv2.undistort()`](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d) function, which is used to transform an image to compensate for lens distortion.). `R` is the camera pose with rotation transformation paramters, and `T` denotes the translation transformation paramters, which indicate the offset or location of the camera  (if we treat the camera as a point, then `T` should be the start point of each camera ray). `K` and `D` parameters are firstly used to transform the original image. `K`, `R`, `T`, compensated image and mask will used to get rays, sampling 3D points, and GT rgb values.

Right now, the vertices coordinates on a human smpl model is in the smpl coordinates, not in the world coordinates, but the camera rays are in the world coordinates, how to connect these two kinds of data in 3D space? Actually, we can transform the smpl coordinates back to the world coordinates. Not really clear, why we need get the smpl coordinate to form the feature vector for each point. Maybe the smpl coordinate is invariant to the world coordinates, will not be changed along with the world coordinate system. 


## The computational logic to sample rays
According to the camera intrinsic and extrinsic, for each pixel position (i,j), we can get the 3D start point and end point in the space by shoting a camera ray thought that point, then according to the semantic segmention mask (24 parts for each human body), we can sampling rays on each pixel position with respect to different human body part, for example, for human body part part, we can locate pixel locations where `msk == 1`, then sample some pixel locations according to some preset sampling ratio, so as to the face part, and other parts in the projected 2D bounding box mask. 
 
Specifically, the process to sample rays is defined in the function `sample_ray_h36m()` in `if_nerf_data_utils.py` file. Firstly, a `get_rays()` function is called to get the start (`ray_o`) and end points (`ray_d`) on each pixel location (i,j) according to the size of image, camera intrinsic and extrinsic parameters. According to the bounding box of the smpl model (axis-aligned bodunig box, AABB) in the world coordinates, `get_bound_2d_mask()` function will return a projected bounding box mask in 2D plane with the same size of image. Generally, the bounding box mask shows a hexagon (polygon with six edges maybe). It indicates the region of valid rays, which can hit the particles of the 3D model within the 3D bounding box. 

The process of sampling rays can be described as follows:
(1) get the index in according to the semantic mean on the mask, (2) randomly select some locations on the mask, (3) index the start (`ray_o`) and end points (`ray_d`) of the camera rays, (4) index the RGB values according to the location of the selected positions as ground-truth RGB values.

The process of getting the `near` and `far` points for each ray. **Note that: in the original NeRF model, near and far values for each ray is pre-defined as (for example, in the training process for `lego` dataset, the `near` and `far` parameters are set to be 2.0 and 6.0)**. However, in `neural body`, the `near` and `far` for each ray are calcuate according to the 3D bounding box of the human body. Concretely, the computational process is defined in the `get_near_far()` function.

> TODO list: Right now, it's still unclear for me to understand the whole process of how to get the near and far parameters, but I think it is more related or similar to the algorithm in [`Ray intersection with Box`] [https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection], I think the code is implemented the process in a more efficient way by consider all the point once. But the goal of this function is clear, that is, to get the near and far boundary for each ray.

After sample rays on some specific human body region, as well the projected 3D bound box mask, the `__getitem__()` function defined in the `dataloader` will return two main sets of data stored in dictionaries `ret` and `meta`, respectively. (Actually, the two sets of data are merged together by using the `dict.update()` function). More specifically,

```python
ret = {
    "feature": feature,      # 3D point cloud coordintes (6890, 6) in the smpl coordinates
    "coord": coord,          # voxel coodinates by voxelization the 3D mesh, more exactly, the 3D point cloud of a human body
    "out_sh": out_sh,        # not clear yet, the voxel shape with how many voxel in heigh, width and length dimension. three interger.
    "rgb": rgb,              # rgb values for each sampled ray, [N_rays, 3]
    "ray_o": ray_o,          # start point for each sampled ray, [N_rays, 3]
    "ray_d": ray_d,          # end point for each sampled ray, [N_rays, 3]
    "near": near,            # near point for each sampled ray, [N_rays, 1]
    "far": far,              # far point for each sampled ray, [N_rays, 1]
    "acc": acc,              # Not clear about this parameter
    "mask_at_box": mask_at_box, # ray-box intersection mask
}

meta = {
    "bounds": bounds,        # bounding box of smpl coordinates 
    "R": R,                  # Rotation matrix of smpl model (1,3) vector, converted to rotation matrix by using cv2.Rodrigues() function
    "Th": Th,                # Translation matrix of smpl model (1,3) vector
    "center": center,        # center of smpl model after  transformation augmentation
    "rot": rot,              # rotation parameters of sml model after transformation augmentation
    "trans": trans,          # translation parameters of sml model after transformation augmentation
    "i": i,                  # index of camera by calculating the mode(num_camera), since the frames are flatten together, this is the way to index the camera view of this frame
    "frame_index": frame_index, # frame index
    "cam_ind": cam_ind,      # camera_index, cam_inds is also flatten the camera index for each image, in general, i and cam_ind should be with equal number.
}
```

The whole process of dataloader can be described as follows:
- extract subset of images to construct the training image set (obtain frames with a fixed interval with respect to different views.).
- extract the parameters of smpl model for each frame (one set of smpl parameters for the frames with the same frame number with respect to different views.)
- randomly select one image, sampling camera rays on the image according to the camera intrinsic and extrinsic.
  - sampling positions on the 2D image plane to index the camera rays.
  - filtering out rays according to specific part of human body (e.g., human body, face.)
  - filtering out rays accroding to bounding box mask
  - filtering out rays according to near and far bounding box mask
  - concatenate the sampling points until the pre-set number of sampling rays is reached.
- return the feature (coordinates in smpl model) and camera rays information (including start and end points of each ray, near, far points, etc.).
- return the 3D bounding box, paramters of smpl model of this frame.
  
> TODO: (1) there is no samping 3D points process in the dataloader?
> (2) why the smpl model parameters are returned, why the camera intrinsic and extrinic should not be returned? is it because the camera rays informtion are already transfered to world coordinates?


# model construction
## Overview of the whole computational graph
class NetworkWrapper(nn.Module):
The code in the `train_net.py` show two stage of model construction. The first part of model is defined in `network = make_network()` which aims to output the raw values of RGB and alpha value according to the 3D points in the smpl coordinates. (The idea is seems like the neural texture feature encoding proposed in the Deferred Neural Rendering, but not exactly the same). The second part of network is defined in `make_trainer()` function in the begining of `train()` function. A more specficial execution pipeline is `make_trainer()`->`_wrapper_factory()`->`if_nerf_clight.NetworkWrapper()`. The `network` will fed into `if_clight_renderer.Renderer` class, which is defined in `lib.networks.render.if_clight_render.Render`. The `Render` class is similar with `NeRF` model structure. The `render()` function in the `Render` class also contains 
- The `get_sampling_points` on the ray according to the start point(`ray_o`), end point (`ray_d`), `near`, `far` and number of sampling points (`N_samples` defined in the `yaml` file).
- Positional encoding function `embedder.xyz_embedder` for 3D points sampled on each ray, view direction encoding `embedder.view_embedder` on the normalized (`ray_d`) values. 

But, instead of directly feed these light points and direction embeddings into a MLP to regress the raw color and alpha value for each point (original `NeRF` model). The neural-body model introduced three extra functions, including `pts_to_can_pts()`, `prepare_sp_input()` and `get_grid_coords()` to get another two varibles, i.e., `sp_input` and `grid_coords` as input to fed into the `Network` module, the network model will regress the `rgb` and `alpha` values.


## Data pre-processing 
- `pts_to_can_pts()` function will transform `pts` with `(x, y,z)` from the world coordinate to the smpl coordinate. The input pts `(bs, n_rays, n_samples, 3)` in the world coordinates, the output pts should be also with the size `(bs, n_rays, n_samples, 3)`

- `prepare_sp_input()` function will update the variables including `feature`, voxel `coordinates`, voxel `shape` and `batch_size` and `i` to a new dictionary `sp_input`
  - `feature` with size `(bs, 6890, 6)` will be reshaped to `(bs*6890, 6)`
  - `coord` which stored the voxel coordinates in the smpl coordinate system with size `(bsx6890, 3)` will be update to `(bsx6890, 4)` the first value is the index within a batch data.
  - `out_sh` is updated to the maximum voxel shape with the maximum number of voxel in height, width and length-axis.
  - keep the `batch size`, and `i` as input, `i` should be the camera index of each image in the batch.

- `get_grid_coords()` function will conver the xyz coordinates `pts` of the smpl coordinates to the voxel coordinates. This processs is actually the `Voxelization of a Point Cloud`. The sampled points constructed a point cloud, and the voxel bound, or the voxel shape defined the 3D space which will be voxlized.

After the above process, we actuall obtained two sets of coordinates, the first set of data is actually the vertices coordinates in the smpl model coordinates, the second set of data are sampled points with the voxel coordinates. In addition, `light_pts` and `viewdir`, which are embedings of samplied points and view direction in the 3D world coordinates, are also fed into the network module and used to regress the raw `rgb` and `alpha` values. 

## Regress the raw rgb and alpha values for each sampled point with respect to voxel coordinates
This part or network structure is still unclear for me. The model introduced the `spconv.SparseConvTensor` layer, which is defined in the [[Spatially Sparse Convolution Implementation](https://github.com/traveller59/spconv)]. It seems firstly we construct the voxel code of the smpl model and the latent representation, then we perform sparse convolution on those human body voxels. and then we propogate the information of the human body voxel reprentation to the sampled voxel in the space, which denotes a particle in the space. In the impelementation, it concatenate a multi-scale-like feature together, to get the encoded sampling points feature. 

Then the view direction and light points positional embeddings are concatenated togather, and fed into a fully connected layer to get the raw `rgb` value, a shallow-layer will output the raw `alpha` values for each sampling point. 

## raw2ouptus network part
The `raw2outputs()` is acctually the same a the original `NeRF` paper, which applied the volumetric rendering equations to gather all the information on the sampling rays, and calcuate the final rgb map, and depth map.

Specifically, `raw2outputs()` function transforms the raw RGB and alpha values of sampled 3D points on a camera ray to a RGB and alpha values.
- The input of the `raw2outputs()` function is:
  - `raw`: the predicted rgb and alpha values for the sampled 3D points along each camera ray, the size of raw tensor `[N_rays, N_samples, 4]`.
  - `z_val`: the accumulated ray marching distance for each sample points within `near` and `far` bound.
  - `rays_d`: view direction of each ray.
- Forward passing process of the `raw2outputs()` function.
  - Get the bin distance on the ray marching process. (TODO: Padding the last distance to a large value, which denotes infinite?)
  - Get the norm of the ray direction
  - `dists`, Distance of each bin on each ray are re-weighted by the norm of each ray direction, return `dists`.
  - apply `sigmoid` function on the first three values in the raw output to get the `rgb` values (range from 0~1, each value are independent)
  - apply a `raw2alpha()` function on the fourth value (alpha prediction + noise value) and `dists` value on each sampled point. the `raw2alpha` is a lambda function which is defined by `1.0 - exp(-relu(raw) * dists)`. It can be observed, if the distance is infinite, the alpha values should be 1.0 accroding to the equation of this function.
  - According the Equation (5) in the ECCV'20 paper of NeRF, the next step aims to accumuate the `1.0-alpha` values to the previous sampling position.
    - here, `torch.cumprod` operation will return the cumulative production of elements by following equations:
    - y_1 = x_1
    - y_2 = x_1 * x_2
    - y_i = x_1 * x_2 * ... * x_i
    - y_Nsample = x_1 * x_2 * ... * x_nsample
  - Then the alpha will multipled by the accumuated `1-alpha` values to get the `weights` as discribed in the second part of Equation (5)
  - re-weight the `rgb` values in the each sampling points, and aggreated them to a single set of RGB values for each camera ray. (See the first part of the Equation (5).)
  - re-weight the accumulated ray marching distance `z_val` and sum the values together to get a depth value for each camera ray.

## Cacluating loss
The calcuation of loss is defined in `lib.train.if_nerf_clight.NetworkWrapper.forward`. Give the RGB prediction of sampled rays and its corresponding ground-truth RGB values (can be indexed by the rays sampling position mask), calcuating the MSE loss on each pixel position. updating the loss in this iteration.

## Summary
The relationship between code execution and functions can be represented by the following hierarchy.

```shell
if __name__ == '__main__':
├── main()
|   ├── dist_setting()
|   ├── make_network()
|   ├── train() # train process if needed
|       ├── make_trainer()
|       ├── make_optimizer()
|       ├── make_lr_scheduler()
|       ├── make_recorder()
|       ├── make_evaluator()        #for evaluation
|       ├── load_model()
|       ├── set_lr_scheduler()
|       ├── make_train_data_loader()
|       ├── make_test_data_loader()
|       ├── training_loop()           # controled by number of epochs 
|           ├── trainer.train()           # train model with one epoch
|               ├── iteration_loop()
|                   ├── data_to_device()     # batch data to cuda devices
|                   ├── network_forward()    # core part defined in the forward function in the model
|                   ├── optimizer.zero_grad() 
|                   ├── loss.mean()
|                   ├── loss.backward()
|                   ├── clip_grad_value()    # 
|                   ├── optimizer.step()
|                   ├── update_recorder()    # update the logger and print the log info if needed.
|           ├── lr_scheduler.step()       # update lr after this training epoch 
|           ├── save_model()              # if needed
|           └── evaluate_model()          # if needed
|   └── test()  # if needed
```


## TODO list:
- Read the Spatial Sparse Convolution Layer.
- Ref: [[spconv - code](https://github.com/traveller59/spconv)]
- Ref: Benjamin Graham,et al. 3d semantic segmentation with submanifold sparse convolutional networks. In CVPR, 2018. [[Project](https://github.com/facebookresearch/SparseConvNet)]

## Unclear part
- How to get the mesh from NeRF (Marching Cube algorithm)?