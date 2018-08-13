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
[perspective1]: ./test_images/extra_image.jpg "Image"
[perspective2]: ./output_images/3.trimmed-extra_image.jpg "Perspective, trimmed"
[perspective3]: ./output_images/3.warped-extra_image.jpg "Perspective, warped"
[poly1]: ./test_images/extra_image.jpg "Image"
[poly2]: ./output_images/5.lanes_polynomial-extra_image.jpg "Lanes, polynomial"
[plot1]: ./test_images/extra_image.jpg "Image"
[plot2]: ./output_images/6.plotted-extra_image.jpg "Lanes, polynomial"
[video1]: ./processed_project_video.mp4 "Video"

## How to run this project
Make sure you've set up your environment as indicated by Udacity in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit). Then, from a command prompt run:
```
python p4.py
```

This script will open file `project_video.mp4` and generate `processed_project_video.mp4` by applying the processing pipeline described in this document.

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
The code for this section is contained in lines 99 through 132 of the file called `p4.py`. The first step I'm taking is cropping the image to only include the area of interest by creating a mask with cv2.fillPoly, and applying it to the image by calling cv2.bitwise_and. This allows me to go from the original:
![Perspective][perspective1]

To a trimmed binary of the original:
![Perspective][perspective2]

After this, I apply a perspective transform by calling cv2.warpPerspective, and obtain a birds's eye view of the road.

```python
    src = np.float32([
            (585, 455),
            (695, 455),
            (1080, 700),
            (200, 700)
        ])
    dst = np.float32([
            (450, 0),
            (830, 0),
            (830, 700),
            (450, 700)
        ])
```

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 455      | 450, 0        |
| 695, 455      | 830, 0        |
| 1080, 700     | 830, 700      |
| 200, 700      | 450, 700      |

Finally, I apply a second data cropping just to remove excess false data from the edges of the image. Note that here I've artificially added green lines to mark what straight lines look like, as a guide:
![Perspective][perspective3]

#### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The code for this section is contained in lines 135 through 266 of the file called `p4.py`. This was the most challenging and fun part of the project.

This function receives an image and an optional state object. If the state is null, then we calculate the lane polynomials by creating a histogram of the bottom half of the image and finding the peak of the left and right halves of the histogram, taking these as the starting points for the left and right lines. I then defined my search based on 25 sliding windows, giving a margin of 20 pixels between windows, and expecting to find a minimum of 100 points per window in order to consider it valid data. This is basically the code we learned at Udacity with very small modifications to account for the binary data I'm giving it, which has been double filtered through the threshold binary processor.

If there's a state data passed to the function, the histogram and sliding windows search is avoided in favor of a simple index search calculation, also as recommended in the tips section of this assignment in Udacity.

After obtaining the lane point indexes I extract the pixel positions and fit a second-order polynomial to them. I then calculate the curvature in world-space and the (x,y) points that will comprise the fit (the lane information). I then use the curvature radii data in order to perform basic sanity check:
1. Does the left-lane curvature is within 4x of the right-lane curvature?
2. Do the directions of the left-lane curvature and the right-lane curvature match?

If either of the two questions above is negative, I then discard the lane point data for the lane that has the highest curvature, and make the lane take the same values as the "more correct" lane but with a translation in position so as to match the original set of data points.

This seems like a trivial correction, and performance-wise isn't so bad, but it made a lot of difference in the outcome of the video, as it helped reduce jitterness and other undesired behaviors derived from not having enough information in the picture frame.

Finally, I plot the lane data, and the polynomial fit to a blank canvas and return it to the caller. I also return the "state", comprised of the left-fit and right-fit data, the left-lane and right-lane curvature in world-space, and the offset from the center in world-space.

This is an example image:
![Polynomial][poly1]

This is the result I observe, with the curvature and offset data added for guidance:
![Polynomial][poly2]

#### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
This is also done in the same function where I identify the lane pixels. The radius of curvature is found in lines 206 to 216 of the file called `p4.py`, while the offset from the center is calculated in line 263 of the same file.

For the curvature we're assuming that 700 pixels in the y-axis are equivalent to 30 meters, while 720 pixels in the x-axis are equivalent to 3.7 meters. This does appear to be very accurate.

#### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
The code for this section is contained in lines 269 through 284 of the file called `p4.py`. This takes two images: 
  1. The undistorted version of the camera input, and
  2. The polynomial fit lane image

The second image is transformed back to the original perspective using cv2.warpPerspective with a transformation matrix that's inverted from what we had in the creation of the birds-eye view. Then, this warped image is overlayed on top of the undistorted image with a call to cv2.addWeighted.

```python
    src = np.float32([
            (450, 0),
            (830, 0),
            (830, 700),
            (450, 700)
        ])
    dst = np.float32([
            (585, 455),
            (695, 455),
            (1080, 700),
            (200, 700)
        ])
```

| Source        | Destination   |
|:-------------:|:-------------:|
| 450, 0        | 585, 455      |
| 830, 0        | 695, 455      |
| 830, 700      | 1080, 700     |
| 450, 700      | 200, 700      |

This is an example image:
![Plot][plot1]

This is the result I observe after plotting:
![Plot][plot2]

### Pipeline (video)

#### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

Here's a video of the outcome of this code (link to [youtube](https://www.youtube.com/watch?v=S0oM7TWZ3wQ)):

  [![Video](https://img.youtube.com/vi/S0oM7TWZ3wQ/0.jpg)](https://www.youtube.com/watch?v=S0oM7TWZ3wQ)

Also available for download from my [Github repository](https://github.com/DavidObando/carnd/tree/master/Term1/Project4). Look for file `processed_project_video.mp4`.

### Discussion

It's clear that the project can consume a large amount of time to truly fine tune. I'm turning in a version of the code that isn't finished to my standards, but that meets the bar for a project submition. That said, there are several issues with it:
  - When the road is white, or clear-colored, the yellow lane markings are very hard to recognize and cause some jittering in the green area
  - This is only meant to target this very specific type of video conditions. I haven't had time to apply this pipeline to other videos with elements such as snow.
  - The performance of the code is abhominable. My computer is barely processing 1.5 frames a second, making this 50-second video take over 14 minutes to process.

I was lucky to have been able to overcome issues with the tree shadow in the later part of the video by using a double thresholded binary processing that essentially eliminates most of the tree shadow data from my input. This, combined with the error correction done based on the perceived curvature allowed me to get a very high-quality result without too much complication.

After this project is submitted I'm planning to keep improving this code. I have a deadline to meet so I must send this as it is, but the topic of computer vision just got a lot more interesting to me after the experience given by project 4.
