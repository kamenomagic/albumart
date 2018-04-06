import tensorflow as tf
from pymongo import MongoClient
import matplotlib.pyplot as plt
import cgan
import image_finder

class Collager:

    def __init__(self, sess, model_dir="./cgan_model"):
        self.sess = sess
        self.track_db = MongoClient().albart.tracks
        self.cgan_model_dir = model_dir
        self.cgan = cgan.CGAN(self.sess, n_batches=150)
        self.cgan.init_model()

    def create_collage(self):
        backgrounds = self.cgan.get_gen_imgs()
        for background in backgrounds:
            plt.imshow(background)
            plt.show()



def main():
    with tf.Session() as sess:
        collager = Collager(sess)
        collager.create_collage()
        # config=tf.ConfigProto(device_count={'GPU': 0})


if __name__ == '__main__':
    main()


