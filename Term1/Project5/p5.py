import os
import pickle
import numpy as np
import glob
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


# Perform a histogram of oriented gradients (HOG) feature extraction on a labeled training set of images

# Feature extraction helper functions
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Extracts the HOG features and returns them. Optionally also returns a visualization of
    the HOG features
    """
    return hog(img, orientations=orient, 
          pixels_per_cell=(pix_per_cell, pix_per_cell),
          cells_per_block=(cell_per_block, cell_per_block), 
          transform_sqrt=False, 
          visualise=vis, feature_vector=feature_vec)

def bin_spatial(img, size=(32, 32)):
    """
    Takes a 3-channel image in the shape of a 3-D array, and resizes the 3
    channels to the specified size. The channels are then flattened and
    concatenated linearly.
    """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):
    """
    Takes a 3-channel image in the shape of a 3-D array, and produces
    a histogram for each channel using the specified number of bins. The
    resulting histograms are concatenated linearly.
    """
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Receives an image and a set of parameters indicating:
      1) whether or not to do HOG feature extraction, and on which channels
          the HOG features will be extracted, the number of orientations,
          the pixels per cell, and the cells per block to use.
      2) whether or not to do color historgram feature extraction, and how
          many bins to use for the histograms.
      3) whether or not to do spatial feature extraction, and what size the
          image should be scaled to before creating the feature vector.
    The routine also allows the caller to specify the color space to use
    before feature extraction. The input image is assumed to be 'RGB' and
    a color space transformation will be done as required.
    """
    img_features = []
    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    # Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    # Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    # Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = np.array([])
            for channel in range(feature_image.shape[2]):
                np.append(hog_features, get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)
    return np.concatenate(img_features)


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Iterates on each of the image file paths specified in the `imgs` list,
    reads the image file and extracts its features. The resulting feature set
    is returned.
    """
    features = []
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        features.append(single_img_features(img=image, color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           hog_channel=hog_channel,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat))
    return features

# Load training data

color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (64, 64) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

svc_pickle_file = "./svc.pickle"
if os.path.exists(svc_pickle_file):
    with open(svc_pickle_file, 'rb') as f:
        pdata = pickle.load(f)
        svc = pdata['svc']
        X_scaler = pdata['X_scaler']
        del pdata
else:
    car_image_list = glob.glob('./data/vehicles/**/*.png')
    non_car_image_list = glob.glob('./data/non-vehicles/**/*.png')

    car_features = extract_features(car_image_list, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(non_car_image_list, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_image_list)), np.zeros(len(non_car_image_list))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)

    svc = LinearSVC()
    svc.fit(X_train, y_train)

    fit_accuracy = svc.score(X_test, y_test)
    print("Accuracy:", fit_accuracy)

    with open(svc_pickle_file, 'wb') as f:
        pickle.dump({
                    "svc": svc,
                    "X_scaler": X_scaler
                },
                f, pickle.DEFAULT_PROTOCOL)

# Implement a sliding-window technique and use your trained classifier to search for vehicles in images

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Takes an image, boundary positions on x and y, a window size, and an how much to overlap
    between windows on the x and y axis.
    Returns a set of all the windows, each window being a tuple of 2 coordinates: top-left
    and bottom-right.
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = math.ceil((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = math.ceil((yspan-ny_buffer)/ny_pix_per_step)

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            left = xs*nx_pix_per_step + x_start_stop[0]
            right = left + xy_window[0]
            top = ys*ny_pix_per_step + y_start_stop[0]
            bottom = top + xy_window[1]
            window_list.append(((left, top), (right, bottom)))
    return window_list


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    """
    Takes an image, and a set of windows to focus our analysis on. For each
    image focused by a window, we'll extract the features (based on the same
    parameters used to train the classifier) and then we ask the classifier
    for its prediction.
    """
    on_windows = []
    for window in windows:
        test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws squares as specified by the bounding box set
    """
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

# Let's generate the sliding windows parameter set

image = mpimg.imread('test_images/test1.jpg')
draw_image = np.copy(image)
image = image.astype(np.float32)/255.

sliding_windows_parameter_set=[]

# Determine the scale and windows to use
# 1. Small window

xy_window=(32, 32)
x_start_stop = [np.int(draw_image.shape[1]*0.33), np.int(draw_image.shape[1]*0.66)]
y_start_stop = [380, 450]

sliding_windows_parameter_set.append([xy_window,x_start_stop,y_start_stop])

windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=xy_window, xy_overlap=(0.0, 0.0))                      

window_img = draw_boxes(draw_image, windows, color=(255, 0, 0), thick=3)
#plt.imshow(window_img)

# 2. Medium small window

xy_window=(64, 64)
x_start_stop = [np.int(draw_image.shape[1]*0.1), np.int(draw_image.shape[1]*0.1*9)]
y_start_stop = [380, 540]

sliding_windows_parameter_set.append([xy_window,x_start_stop,y_start_stop])

windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=xy_window, xy_overlap=(0.0, 0.0))                      

window_img = draw_boxes(draw_image, windows, color=(255, 0, 0), thick=3)
#plt.imshow(window_img)

# 3. Medium large window

xy_window=(128, 128)
x_start_stop = [None, None]
y_start_stop = [400, 600]

sliding_windows_parameter_set.append([xy_window,x_start_stop,y_start_stop])

windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=xy_window, xy_overlap=(0.0, 0.0))                      

window_img = draw_boxes(draw_image, windows, color=(255, 0, 0), thick=3)
#plt.imshow(window_img)

# Let's analyze the test image using the defined sliding windows parameter set

window_img = np.copy(draw_image)

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

    window_img = draw_boxes(window_img, hot_windows, color=(0, 0, 255), thick=6)

#plt.imshow(window_img)

# Now, let's create a heatmap and see where the windows converge

heat = np.zeros_like(image[:,:,0]).astype(np.float)

def add_heat(heatmap, bbox_list, heat_score=1):
    """
    Increases the value of a region in a heatmap when a box touches the region
    """
    for box in bbox_list:
        # Assuming each box takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += heat_score
    return heatmap
    
def apply_threshold(heatmap, threshold):
    """
    Zeroes out pixels below the specified threshold
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    """
    Given a set of labeled boxes and an image, we'll draw a box
    around each set of minimum-maximum values for a label region
    """
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Skip skinny boxes
        if (bbox[1][0] - bbox[0][0]) < 32:
            continue
        if (bbox[1][1] - bbox[0][1]) < 32:
            continue
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 5)
    return img

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

#plt.imshow(draw_img)

# Video processing pipeline
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


processing_state = None # global state for the processing pipeline
def process_image_with_heatmap(image):
    global processing_state 
    result, processing_state = pipeline(image, processing_state, True)
    return result

# Now, let's process the test video!
print("Begin video processing")
processed_output = 'processed_test_video.mp4'
original_clip = VideoFileClip("test_video.mp4")
processing_state = None
processed_clip = original_clip.fl_image(process_image_with_heatmap)
processed_clip.write_videofile(processed_output, audio=False)
print("Done processing")


processing_state = None # global state for the processing pipeline
def process_image(image):
    global processing_state 
    result, processing_state = pipeline(image, processing_state)
    return result

# Now, let's process the project video!
print("Begin video processing")
processed_output = 'processed_project_video.mp4'
original_clip = VideoFileClip("project_video.mp4")
processing_state = None
processed_clip = original_clip.fl_image(process_image)
processed_clip.write_videofile(processed_output, audio=False)
print("Done processing")

