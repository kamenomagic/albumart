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

    srch_dict = {'artist': 'AC/DC'}
    test = albums.find(srch_dict)
    for album in test:
        img = cStringIO.StringIO(base64.b64decode(album["image"]))
        img = np.asarray(Image.open(img))
        plt.figure(1)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()