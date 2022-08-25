
Project - v4 2022-08-23 9:22am
==============================

This dataset was exported via roboflow.com on August 23, 2022 at 6:27 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 192 images.
Kimlik are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random rotation of between -29 and +29 degrees
* Random brigthness adjustment of between -19 and +19 percent
* Random Gaussian blur of between 0 and 0.25 pixels


