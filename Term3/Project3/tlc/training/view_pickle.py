import numpy as np
import pickle

if __name__ == '__main__':

    # load pickle data set annotation
    #with open('VOC2007.pkl2', 'rb') as f:
    #with open('TLD201803.pkl', 'rb') as f:
    #with open('prior_boxes_ssd300.pkl', 'rb') as f:
    with open('TLD201803-4.p', 'rb') as f:
        #u = pickle._Unpickler(f)
        #u.encoding = 'latin1'
        #data = u.load()
        data = pickle.load(f)
        print(data)
