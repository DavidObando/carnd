**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[calibration1]: ./output_images/camera_calibration.jpg "Calibration"

## How to run this project
Make sure you've set up your environment as indicated by Udacity in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit). Then, from a command prompt run:
```
python p5.py
```

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Histogram of Oriented Gradients (HOG)

#### Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

The code for this section is contained in lines 19 through 141 of the file called [`p5.py`](./p5.p5). In specific, the following routines are declared:
  * Line 20, `get_hog_features`: Extracts the HOG features and returns them. Optionally also returns a visualization of the HOG features.
  * Line 31, `bin_spatial`: Takes a 3-channel image in the shape of a 3-D array, and resizes the 3 channels to the specified size. The channels are then flattened and concatenated linearly.
  * Line 42, `color_hist`: Takes a 3-channel image in the shape of a 3-D array, and produces a histogram for each channel using the specified number of bins. The resulting histograms are concatenated linearly.
  * Line 55: `single_img_features`: Receives an image and a set of parameters indicating:
      1) whether or not to do HOG feature extraction, and on which channels the HOG features will be extracted, the number of orientations, the pixels per cell, and the cells per block to use.
      2) whether or not to do color historgram feature extraction, and how many bins to use for the histograms.
      3) whether or not to do spatial feature extraction, and what size the image should be scaled to before creating the feature vector.
    The routine also allows the caller to specify the color space to use before feature extraction. The input image is assumed to be 'RGB' and a color space transformation will be done as required.
  * Line 110, `extract_features`: Iterates on each of the image file paths specified in the `imgs` list, reads the image file and extracts its features. The resulting feature set is returned.

All of these routines are familiar to the class as they've been defined in multiple forms and multiple places in the class contents leading up to this project. Following the declaration of those routines we declare the main parameters for this application, namely those that will drive feature extraction, color space to use, among others. Lines 132 to 141:
```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (64, 64) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

The values seen above were the ones I ended up using for my final rendering. I tried all the color spaces that were available, as well as a combination of different values for `spatial_size`, `hist_bins`, `pix_per_cell` and `cell_per_block`. A 64 by 64 spatial binning, with 16 bins for color histogram made my code give the best results. I went on a trial-and-error quest which took me to spatial binning values as low as 8 and as high as 128. I'm assuming that since the original training data was 64 by 64, that might influence the linear SVC I'm using and thus a 64 by 64 arrangement works best. As or color histogram bin values, I noticed that they go hand-in-hand with which color space you select. From all the color space and histogram bin size combinations I tested, using `'YCrCb'` and 16 bins yielded the least false positives in my rendering.

I also tested turning off the colleciton of spatial features, histogram features, and HOG features, but in all cases having the three feature sets turned on yielded the best results.

#### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this section is contained in lines 143 through 190 of the file called [`p5.py`](./p5.p5). For practical reasons, the code first checks whether there's a pickle file it can use to load the classifier and the scaler, and if there isn't then we proceed to create a new classifier and train it.

Training of the classifier begins by obtaining the data that will be used. You can obtain the vehicle data from [vehicles.zip](./data/vehicles.zip) and the non-vehicle data from [non-vehicles.zip](./data/non-vehicles.zip). The zip files are expected to be extracted into a `./data` directory.

The code proceeds to "glob" the data into 2 separate sets:
```python
    car_image_list = glob.glob('./data/vehicles/**/*.png')
    non_car_image_list = glob.glob('./data/non-vehicles/**/*.png')
```

Each of these sets is processed with a call to `extract_features`, with the resulting feature sets containing the features of both sets of images. We then proceed to generate the feature set and the label set that will be fed to the classifier for training.
```python
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_image_list)), np.zeros(len(non_car_image_list))))
```

The car features and notcar features are vertically stacked, and then scaled using a standard scaler. The labels are generated with a horizontal stacking of ones for the car images and zeroes for the notcar images.


We then split the feature and labels into a training set and a testing set, and use the training data to "fit" or train the linear SVC classifier:
```python
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)

    svc = LinearSVC()
    svc.fit(X_train, y_train)
```

After creating and training the classifier, we obtain its accuracy by calling `svc.score` with the testing data, and save the resulting classifier and scaler to a pickle file for future use.

### Sliding Window Search

#### Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The code for this section is contained in lines 191 through 346 of the file called [`p5.py`](./p5.p5). In specific, the following routines are declared:
  * Line 191, `slide_window`: Takes an image, boundary positions on x and y, a window size, and an how much to overlap between windows on the x and y axis. Returns a set of all the windows, each window being a tuple of 2 coordinates: top-left and bottom-right.
  * Line 233, `search_windows`: akes an image, and a set of windows to focus our analysis on. For each image focused by a window, we'll extract the features (based on the same parameters used to train the classifier) and then we ask the classifier for its prediction.

#### Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?

### Video Implementation

#### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

#### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

