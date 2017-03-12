import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Video processing pipeline
def pipeline(img, previously_known_polynomials):
    pass

processing_state = None # global state for the processing pipeline
def process_image(image):
    global processing_state 
    result = pipeline(image, processing_state)
    return result

# Now, let's process the video!
print("Begin video processing")
processed_output = 'processed_project_video.mp4'
original_clip = VideoFileClip("project_video.mp4")
processing_state = None
processed_clip = original_clip.fl_image(process_image)
processed_clip.write_videofile(processed_output, audio=False)
print("Done processing")
