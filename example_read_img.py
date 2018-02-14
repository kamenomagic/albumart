from PIL import Image
from pymongo import MongoClient
import base64
from matplotlib import pyplot as plt
import cStringIO

def main():
    mongo = MongoClient()
    db = mongo.albart
    albums = db.albums

    test = albums.find()
    img = cStringIO.StringIO(base64.b64decode(test[1]["image"]))
    img = Image.open(img)
    plt.figure()
    plt.imshow(img)
    plt.show()
    print img








if __name__ == '__main__':
    main()