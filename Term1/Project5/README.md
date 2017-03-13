**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[window1]: ./report_images/1.window32x32.png "Window definition 1"
[window2]: ./report_images/2.window64x64.png "Window definition 2"
[window3]: ./report_images/3.window128x128.png "Window definition 3"
[detection1]: ./report_images/4.detection.png "Car detection"
[detection2]: ./report_images/5.detectionwithheatmap.png "Car detection with heatmap"
[test1]: ./report_images/6.test1.png "Test image"
[test2]: ./report_images/7.test2.png "Test image"
[test3]: ./report_images/8.test3.png "Test image"
[test4]: ./report_images/9.test4.png "Test image"
[test5]: ./report_images/10.test5.png "Test image"
[test6]: ./report_images/11.test6.png "Test image"
[optimization1]: ./report_images/12.optimization1.png "Optimization"

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

The code for this section is contained in lines 191 through 407 of the file called [`p5.py`](./p5.p5). In specific, the following routines are declared:
  * Line 191, `slide_window`: Takes an image, boundary positions on x and y, a window size, and an how much to overlap between windows on the x and y axis. Returns a set of all the windows, each window being a tuple of 2 coordinates: top-left and bottom-right.
  * Line 233, `search_windows`: akes an image, and a set of windows to focus our analysis on. For each image focused by a window, we'll extract the features (based on the same parameters used to train the classifier) and then we ask the classifier for its prediction.
  * Line 262, `draw_boxes`: Draws squares as specified by the bounding box set.
  * Line 352, `add_heat`: Increases the value of a region in a heatmap when a box touches the region.
  * Line 361, `apply_threshold`: Zeroes out pixels below the specified threshold.
  * Line 368, `draw_labeled_bboxes`: Given a set of labeled boxes and an image, we'll draw a box around each set of minimum-maximum values for a label region.

All of these routines are familiar to the class as they've been defined in the class contents leading up to this project.

I then proceed to generate the sliding windows parameter set. That is, which parameters will be used for different window sizes. The first parameter set I define comes in line 282. It defines a 32x32 window on a span centered around the horizon viewport:

```python
xy_window=(32, 32)
x_start_stop = [np.int(draw_image.shape[1]*0.33), np.int(draw_image.shape[1]*0.66)]
y_start_stop = [380, 450]
```
![Windows][window1]

The second parameter set is in line 296, being a 64x64 window on a slightly larger area:

```python
xy_window=(64, 64)
x_start_stop = [np.int(draw_image.shape[1]*0.1), np.int(draw_image.shape[1]*0.1*9)]
y_start_stop = [380, 540]
```
![Windows][window2]

The third and final parameter set is in line 310, with a 128x128 window on the largest area yet:

```python
xy_window=(128, 128)
x_start_stop = [None, None]
y_start_stop = [400, 600]
```
![Windows][window3]

The images above show the grid as it would appear with zero overlap between the windows, but in the actual window search code I'm using a 75% overlap between windows in both the x and y axis. We can then detect a good number of features we're looking for, and some we aren't. From line 328:

```python
for window_parameter_set in sliding_windows_parameter_set:
    i_xy_window = window_parameter_set[0]
    i_x_start_stop = window_parameter_set[1]
    i_y_start_stop = window_parameter_set[2]
    windows = slide_window(image, x_start_stop=i_x_start_stop, y_start_stop=i_y_start_stop,
                    xy_window=i_xy_window, xy_overlap=(0.75, 0.75))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    if len(hot_windows) > 0:
        hot_window_list.append(hot_windows)
```

![Detection][detection1]

After this, we proceed to create a heatmap to aid in identifying properly which of these overlapping regions are more likely true positives. For that, we call functions `add_heat` with the result of the hot windows found above, and then `apply_threshold` in order to drop some of the false positives.

```python
heat = np.zeros_like(image[:,:,0]).astype(np.float)
...
# Add heat to each box in box list
heat = add_heat(heat, np.concatenate(hot_window_list))
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,2)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

heatmap = np.uint8(heatmap / heatmap.max() * 255)
heatmap = np.dstack((heatmap,heatmap,heatmap))
draw_img[0:288, 0:512, :] = cv2.resize(heatmap, dsize=(512,288))
```

![Detection][detection2]

#### Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?

Here's some test images run under the processing pipeline:

![Test][test1]
![Test][test2]
![Test][test3]
![Test][test4]
![Test][test5]
![Test][test6]

As you can see, the detection algorithms still show some issues. They are less relevant in the video than in the stills above given that the pipeline for the video takes into account information from previous frames in order to help inform the decision where the cars are.

### Video Implementation

#### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a video of the outcome of this code:
  - Youtube [video](https://youtu.be/FufPayIMWHc)

Also available for download from my [Github repository](https://github.com/DavidObando/carnd/tree/master/Term1/Project5). Look for file `processed_project_video.mp4`.

#### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The main optimization done to increase the quality of the data was the fine-tunning of the parameters. Nothing else I did in this project has the same degree of impact as playing with these parameters, and with the window overlap parameter:
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

That said, a second optimization done was keeping a state variable for the video pipeline, in which previous frames inform the current frame on where the heatmap might be the brightest. Give that previous frames can and will affect current frames we also must account for this when setting a value to the threshold:

```python
def pipeline(img, state, with_heat_map=False):
    image = img.astype(np.float32)/255.
    hot_window_list = []
    for window_parameter_set in sliding_windows_parameter_set:
        i_xy_window = window_parameter_set[0]
        i_x_start_stop = window_parameter_set[1]
        i_y_start_stop = window_parameter_set[2]
        windows = slide_window(image, x_start_stop=i_x_start_stop, y_start_stop=i_y_start_stop,
                        xy_window=i_xy_window, xy_overlap=(0.75, 0.75))

        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        if len(hot_windows) > 0:
            hot_window_list.append(hot_windows)

    if len(hot_window_list) > 0:
        hot_window_list = np.concatenate(hot_window_list)
    if state is None:
        state = [hot_window_list]
    else:
        state.append(hot_window_list)
    time_window = 5
    if len(state) > time_window:
        state = state[1:(time_window+1)]
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat_score = 1
    for frame_windows in state:
        heat = add_heat(heat, frame_windows, heat_score)
        #heat_score += 1
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, len(state)*1.5)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    if with_heat_map:
        heatmap = np.uint8(heatmap / heatmap.max() * 255)
        heatmap = np.dstack((heatmap,heatmap,heatmap))
        draw_img[0:288, 0:512, :] = cv2.resize(heatmap, dsize=(512,288))
    
    return draw_img, state
```

Notice how the call to `apply_threshold` computes the threshold based on the length of the state (which is capped to be maximum 5 frames) times 1.5. This allows the previous 4 frames to add to the body of knowledge about the scene. I played with a larger time frame (up to 10), as well as with modifying the "heat score" each frame added to the heat map (older frames adding less heat) but this didn't seem to move the needle too much so I reduced the code to its basic form.

One more example lays in function `draw_labeled_bboxes`, lines 381 through 385 of the file called [`p5.py`](./p5.p5). Here, if the width or the height of the box is less than 32 pixels then we omit it:
```python
        # Skip skinny boxes
        if (bbox[1][0] - bbox[0][0]) < 32:
            continue
        if (bbox[1][1] - bbox[0][1]) < 32:
            continue
```

That optimization makes boxes that aren't really there go away, for example compare the top-left heatmap to the actual picture which correclty skips these boxes:

![Optimization][optimization1]

The "skip skinny boxes" values were tuned by hand, but they seem to be intutively agreeable given they measure half of the pixels used by the spatial binning feature on each dimension it works on.

Another more explicit optimization was the use of smaller areas on which the windows will operate, such as:

![Windows][window1]

This ensures we don't go looking for cars in regions of the image where none will appear. The change between having these regions and not having them, or even only bounding on the y axis versus also bounding on the x axis, are very steep. Given that my model uses an overlap of 75% we get massive performance increases simply by doing less work.

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The code is far from perfect, but does achieve a workable solution on car detection. I worked very hard to reduce both the false positives and the wobbliness of the box once a car ws found, but I think in order to achieve better results I should move beyond a linar SVC. This project involved a lot of experimentation, over 20 hours were spent just in parameter tunning given that the results would vary wildly if I were to simply use a different color space and not modify the color histogram bin size. The effort was worth it, I feel like I have gotten a good grasp of how these variables interact and affect each other, and I will be able to improve this technique for image detection in the future.

My pipeline still lacks robustness around false positives that sprout out of the concrete. I wasn't able to fully eliminate them without also stop detecting true positives. It appears to me that either the training data I have isn't enough, or the linear SVC model I used isn't powerful enough to linearly separate the data at hand with the configuration I've created in this project. One thing that came to mind was using more than one color space in the feature data in order to have more features to look at. Additionally, I'm thinking that replacing the linear support vector machine by a convolutional neural network might prove like a fun post-submit challenge.

Finally, the performance I obtained was less than ideal. Despite some of my efforts to reduce complexity or total amount of computations performed, each frame was taking over 1 second to process, which makes this code likely unsuitable for near-real-time applications where the video feed can inform a machine on how to drive a car. I'm looking forward to learning more about how to address this issue in term 2 of the Self-driving car nanodegree program.

Thank you for reading and reviewing this!

David.
