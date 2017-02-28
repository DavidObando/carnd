**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration1]: ./output_images/camera_calibration.jpg "Calibration"
[calibration2]: ./test_images/extra_image.jpg "Image"
[calibration3]: ./output_images/1.undistorted-extra_image.jpg "Undistorted image"
[binary1]: ./test_images/extra_image.jpg "Image"
[binary2]: ./output_images/2.binary-extra_image.jpg "Binary thresholded image"
[video1]: ./processed_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this section is contained in lines 15 through 59 of the file called `p4.py`. I'm saving the values I compute to a pickle file for easy access given the iterative nature of this project. If the pickle file doesn't exist then I compute the camera calibration values. The steps I follow to compute this:
  1. Convert the image to grayscale
  2. Try to find the chessboard corners using cv2.findChessboardCorners
  3. If found, I add them to the collections tracking the object points and image points

These collections of points are what allows us to calibrate images. Object points is simply a set of points in the x- and y-axis of the picture. We don't calibrate on the z-axis as this axis is not spatial data but color channel data. We simply create a flat distribution of points in a `9 x 6` grid and call it the object points. Image points are what we find from calling cv2.findChessboardCorners.

These points will be used to calibrate pictures by calling cv2.calibrateCamera with the object points and image points as reference to what type of distortions the image presents.

### Pipeline (test images)

#### Provide an example of a distortion-corrected image.
This is one of the camera calibration pictures corrected using the data obtained in the previous step.

![Calibration][calibration1]

The code for this section is contained in lines 62 through 66 of the file called `p4.py`.

Here's an action snap from the road:
![Calibration][calibration2]

And now, calibrated:
![Calibration][calibration3]


#### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

The code for this section is contained in lines 69 through 96 of the file called `p4.py`. I created 2 functions, so I'll detail them both:

##### Function `thresholded_binary`
The function expects an RGB image and converts it to HSV color space. It then runs cv2.sobel on L channel to obtain the gradient with respect to the X coordinate. The result of sobel is averaged and scaled to enable comparison against an 8-bit threshold. We then compare the sobel to see if it lands within the specified gradient threshold (with values `[20, 100]` by default).

Then, the function detects which entries in the S channel fall within a threshold (with values `[170, 255]` by default). This is so the S channel tells us where the color saturation is high regardless of the color of the are we look at. This is useful as sometimes it's hard to see lines when they are under tree shades or on clear concrete.

Both the x-gradient on the L channel and the value on the S channel are taken into account to compute a bitmap of 1s and 0z that tell us where the areas of interest are, where the colors change on the x-axis, and where the saturation of the colors is most visible.

##### Function `double_thresholded_binary`
The function also expects an RGB image as it forwards the call to `thresholded_binary`. However, after the thresholded binary image is obtained, it creates a black-and-white RGB image based on it and sends it a second time to function `thresholded_binary`. This turned out to provide amazing effects when dealing with shadows and other color changes on the road.

Here's the result of the double thresholded binary function:
![Binary][binary2]

#### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
