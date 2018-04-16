PYTHON2 = ~/.pyenv/versions/2.7.12/bin/python
PYTHON2G = env LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ~/.pyenv/versions/2.7.12/bin/python
PYTHON3 = ~/.pyenv/versions/3.6.4/bin/python


all: gt

gt:
	$(PYTHON2) gt_format.py

extract_images:
	$(PYTHON2) bag_to_images.py bag/just_traffic_light.bag ssd/images /current_pose
