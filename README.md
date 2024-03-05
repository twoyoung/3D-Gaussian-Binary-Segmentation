# 3DGS-binary-segmentation

This is the first part of a project that aims to count the number of fruits on trees based on 3D Gaussian model. It achieves the function of binary segmentation of the target objects from the surrounding enviornment point cloud. The underlying logic is explained in this [paper](https://huggingface.co/datasets/twoyoung/3DGS-binary-segmentation/resolve/main/paper_2D_guided_3D_gaussian-based_binary_segmentation.pdf)

Segmentation effect on a Christmas tree:

![demo video](https://huggingface.co/datasets/twoyoung/christmasTree/resolve/main/demo.gif)

The paper is [here](https://huggingface.co/datasets/twoyoung/christmasTree/resolve/main/paper_2D_guided_3D_gaussian-based_binary_segmentation.pdf).

## Run the demo code
Run the demo with prepared dataset:

    /colab/3DGS_binary_segmentation.ipnyb

## Preparation
To bianry segment the target object, 3 things need to be prepared in advance:

1. camera information (https://huggingface.co/datasets/twoyoung/christmasTree/resolve/main/christmasTree.zip):

    - use colmap to get the sparse point cloud from original images:
      
          /colab/use_colmap_to_generage_point_cloud.ipynb

2. pre-trained model with added p and rgb attributes (https://huggingface.co/datasets/twoyoung/christmasTree/resolve/main/pre-trained_3d_gaussian_splatting_model_with_p_and_rgb_added.zip):

    - pre-train the model from the sparse points:
  
          /colab/3D_gaussian_splatting.ipynb

    - add attribute p and rgb to the pre-trained model:
      
          /colab/modify_ply_file_to_add_attribute_p_and_rgb.ipynb

3. masks (https://huggingface.co/datasets/twoyoung/christmasTree/resolve/main/masks.zip):

    - use grounded-SAM to get masks:
      
          /colab/grounded_SAM_for_single_image.ipnyb (This one can be used to try different prompts)
          /colab/grounded_SAM_for_multiple_images.ipnyb (After trying out the best prompts, then generate the masks in a batch)

## Train the model
### Train locally
- clone the repo:
  ```
  git clone --recursive https://github.com/twoyoung/3D-gaussian-binary-segmentation.git
  ```
- create the environment:
  ```
  conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
  conda activate <Drive>/<env_path>/gaussian_splatting
  ```
- install the submodules:
  ```
  cd /3D-gaussian-binary-segmentation
  pip install ./submodules/diff-gaussian-rasterization
  pip install ./submodules/simple-knn
  ```
- train the pre-trained model to get each point's p value
  ```
  python train.py -s /path-to-christmasTree_sparse_point_generated_by_colmap -m /path-to-pre-trained_3D_gaussian_model_with_attribute_p_and_rgb_added/c95e51cf-3/point_cloud/iteration_10000 --kmask_path /path-to-christmasTree_masks
  ```
### Train on Colab

      /colab/3DGS_binary_segmentation.ipnyb

## View the result
- view the trained model as point cloud in open3d and filter the points by p:
  ```
  python ./scripts/filter_point_cloud_by_p.py
  ```
- generate new .ply file with the filtered points and view it in a viewer:
  ```
  python ./scripts/create_ply_file_with_only_filtered_points.py
  ```
  - You can view it locally with SIBR (pre-built binaries for windows: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip)
      ```
      ./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <path to trained model>
      ```
  - Or view it with other tools such as: https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php
  
## Evaluation
- train the model with --eval:
  ```
  python train.py --eval -s /path-to-christmasTree_sparse_point_generated_by_colmap -m /path-to-pre-trained_3D_gaussian_model_with_attribute_p_and_rgb_added/c95e51cf-3/point_cloud/iteration_10000 --kmask_path /path-to-christmasTree_masks
  ```
- render the tain & test image:
  ```
  python render.py -m <path to trained model>
  ```
- calculate mBIoU:
  ```
  python  mBIoU_calculation.py --model_path \path-to-trained-model\
  ```
  
