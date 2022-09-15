
meteors - v1 2022-05-27 11:39pm
==============================

This dataset was exported via roboflow.ai on May 28, 2022 at 4:47 AM GMT

It includes 20509 images.
Meteor are annotated in Tensorflow TFRecord (raccoon) format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 512x512 (Fit (reflect edges))

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -45 and +45 degrees


