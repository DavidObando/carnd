import keras
from keras.preprocessing import image
from scipy.misc import imread

import matplotlib
import matplotlib.pyplot as plt

from tl_classifier import TLClassifier

tld_classes = ['RED', 'YELLOW', 'GREEN', 'unknown']
tld_color = ['red', 'yellow', 'green', 'grey']


if __name__ == '__main__':
    light_classifier = TLClassifier()

    #fig, ax = plt.subplots(1, 1)
    plt.ion()
    ax = plt.subplot()

    for i in range(0, 710, 2):
        inputs = []
        images = []
        img_path = 'images/frame000{:03d}.png'.format(i)
        img = image.load_img(img_path)
        img = image.img_to_array(img) # to 3D numpy.array

        #print(img.shape)
        ## test (1096, 1368, 3)
        ## impl (1096, 1368, 3)

        label, score, top_xmin, top_ymin, top_xmax, top_ymax = light_classifier.get_classification(img)
    
        display_txt = '{:0.2f}, {}'.format(score, label)
        print(label, display_txt)

        ax.cla()
        plt.imshow(img / 255.)
        currentAxis = plt.gca()
    
        xmin = int(round(top_xmin * img.shape[1]))
        ymin = int(round(top_ymin * img.shape[0]))
        xmax = int(round(top_xmax * img.shape[1]))
        ymax = int(round(top_ymax * img.shape[0]))

        if label != 4:
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            display_txt = '{:0.2f}, {}:{}'.format(score, label, tld_classes[label])
            color = tld_color[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

            #plt.show()
            plt.draw()
            plt.pause(0.01)

