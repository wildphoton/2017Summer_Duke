# utils
This folder contains useful scripts for segmentation tasks.  
Important code:

* `CORAL.py` domain adaptation method in [Return of Frustratingly Easy Domain Adaptation](https://arxiv.org/abs/1511.05547). The orginal matlab source code and code of related papers are [here](https://github.com/VisionLearningGroup/CORAL)
* `patchify.py` functions to generate small image patches from images of large sizes and stitch small patches back into large images
* `patchify_building_images.py` script to patchify building images 
* `patchify_solar_panel_images.py` script to patchify solar panel images 
* `unify_image_resolution.py` script to resize images so that they have the same physical resolution (length/pixel)
* `make_all_but_one_file_list.py` script to generate patch file list that comes from multiple cities (all cities but one) and write into text file. This is for experiments training model on data of all-but-one cities.
