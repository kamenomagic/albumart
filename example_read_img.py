from PIL import Image
from pymongo import MongoClient
import base64
from matplotlib import pyplot as plt
import cStringIO
import numpy as np


def main():
    mongo = MongoClient()
    db = mongo.albart
    albums = db.albums

    srch_dict = {}
    search = albums.find(srch_dict)
    dataset = []
    color = [0, 0, 255]
    for album in search:
        img = cStringIO.StringIO(base64.b64decode(album["image"]))
        img = np.asarray(Image.open(img))
        test = np.mean(img, axis=(0, 1))
        if np.sum(np.absolute(test - color)) < 260:
            dataset.append(img)
    print(len(dataset))
    """
    for entry in dataset:
        plt.figure(1)
        plt.imshow(entry)
        plt.show()
    """

if __name__ == '__main__':
    main()