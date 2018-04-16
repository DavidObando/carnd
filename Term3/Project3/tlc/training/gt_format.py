import numpy as np
import csv

class txt2pkl(object):

    WIDTH = 1368
    HEIGHT = 1096

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 4  # 20
        self.data = dict()
        self._process()

    def _process(self):
        with open(self.path_prefix, 'rb') as f:
            spamreader = csv.reader(f, delimiter=' ')  # , quotechar='|')

            print spamreader
            for row in spamreader:
                # print row[0], row[1], row[3], row[4], row[5], row[6], row
                xmin = float(row[3]) / self.WIDTH
                ymin = float(row[4]) / self.HEIGHT
                #xmax = float(row[5]) / self.WIDTH
                #ymax = float(row[6]) / self.HEIGHT
                xmax = (float(row[3]) + float(row[5])) / self.WIDTH
                ymax = (float(row[4]) + float(row[6])) / self.HEIGHT
                # print row[0], row[1], xmin, ymin, xmax, ymax

                bounding_boxes = []
                bounding_box = [xmin, ymin, xmax, ymax]
                bounding_boxes.append(bounding_box)

                one_hot_classes = []
                one_hot_class = self._to_one_hot(int(row[1]))
                one_hot_classes.append(one_hot_class)

                image_name = row[0]
                bounding_boxes = np.asarray(bounding_boxes)
                one_hot_classes = np.asarray(one_hot_classes)
                image_data = np.hstack((bounding_boxes, one_hot_classes))
                self.data[image_name] = image_data

    def _to_one_hot(self, classno):
        """
        uint8 UNKNOWN=4 -> delete
        uint8 GREEN=2
        uint8 YELLOW=1
        uint8 RED=0

        loc = obj[:4]
            ratio / pixel 
        label = np.argmax(obj[4:])

        """
        one_hot_vector = [int(0)] * self.num_classes
        if classno == 0:
            one_hot_vector[0] = 1
        elif classno == 1:
            one_hot_vector[1] = 1
        elif classno == 2:
            one_hot_vector[2] = 1
        #elif classno == 4:
        #    one_hot_vector[3] = 1
        else:
            print('unknown classno: %d' % classno)
        return one_hot_vector


data = txt2pkl("anno_color_train_3.txt").data
#print(data)
import pickle
pickle.dump(data, open('TLD201803-3.p','wb'), protocol=2)
