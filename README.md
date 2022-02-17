# Stereo reconstruction
Working repository for code related stereo reconstruction using classical solutions

Much of this code is based on lectures given by Professor Cyrill Stachniss where a full photogrammetry playlist can be found [here](https://www.youtube.com/watch?v=SyB7Wg1e62A&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y).

An example of stereo reconstruction using this repo is shown here:
<p align="center">
<img src="./readme_images/stereocam_back.jpg" alt="Basic reconstruction" style="width:250px;"/>   <img src="./readme_images/stereocam_front.jpg" alt="Basic reconstruction" style="width:250px;">
</p>
<p align="center">
<img src="./readme_images/andrew.jpg" alt="Basic reconstruction" style="width:500px;"/>
</p>
<p align="center">
<img src="./readme_images/animation.gif" alt="Basic reconstruction" style="width:750px;"/>
</p>
## Currently implemented
* Zhang's method to compute the camera calibration matrix of a pinhole camera.
    * src/stereo_reconstruction/intrinsics.py
* Working on cleaning the code to compute the fundamental matrix.
* Working on cleaning the code to compute disparities.