import tensorflow as tf
import numpy as np
from pymongo import MongoClient
import cStringIO
import base64
from PIL import Image
from tqdm import tqdm, trange

class ImageAutoEncoder:
    epochs = 25
    input_size = [1, 300, 300, 3]
    compressed_size = 256

    def _init_(self):
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, self.input_size, name='x')
        self.encoded_x = tf.placeholder(tf.float32, [1, self.compressed_size], name='encoded_x')

        #Convolution layers
        conv0 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu, name='conv0')
        conv1 = tf.layers.conv2d(inputs=conv0, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[5,5], padding="same", activation=tf.nn.relu, name='conv2')

        # Dense layers
        flat = tf.reshape(conv2, [-1, 300*300*128])
        dense0 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu, name='dense0')
        dense1 = tf.layers.dense(inputs=dense0, units=512, activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu, name='dense2')
        self.encoder = dense2
        self.training_decoder = self.build_decoder(self.encoder, self.input_size, reuse=False)
        self.decoder = self.build_decoder(self.encoded_x, self.input_size)
        self.loss = tf.reduce_sum(tf.abs(self.x - self.training_decoder), name='loss')
        self.train = tf.train.AdamOptimizer().minimize(self.loss, name='train')

    @staticmethod
    def build_decoder(x, out_size, reuse=True):
        decode2 = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu, name='decode2')
        decode1 = tf.layers.dense(inputs=decode2, units=512, activation=tf.nn.relu, name='decode1')
        decode0 = tf.layers.dense(inputs=decode1.encoding, units=1024, activation=tf.nn.relu, name='decode0')

        # Deconvolution layers
        blowout = tf.reshape(decode0, out_size)
        deconv2 = tf.layers.conv2d_transpose(inputs=blowout, filters=128, kernel_size=[5, 5], padding="same",
                                             activation=tf.nn.relu, name='deconv2')
        deconv1 = tf.layers.conv2d_transpose(inputs=deconv2, filters=64, kernel_size=[5, 5], padding="same",
                                             activation=tf.nn.relu, name='deconv1')
        deconv0 = tf.layers.conv2d_transpose(inputs=deconv1, filters=32, kernel_size=[5, 5], padding="same",
                                             activation=tf.nn.relu, name='deconv0')
        return deconv0

    def start_training(self):
        self.sess.run(tf.global_variables_initializer())
        epoch_iterator = trange(self.epochs)
        for e in epoch_iterator:
            current_sum = 0
            count = 0
            try:
                for album in MongoClient().albart.albums.find():
                    try:
                        img = cStringIO.StringIO(base64.b64decode(album[1]["image"]))
                        img = np.asarray(Image.open(img))
                        # Probably still need to do something to the image here...
                        feed_dict = {self.x: [img]}
                        loss, _ = self.sess.run([self.loss, self.train], feed_dict=feed_dict)
                        count += 1
                        current_sum += loss
                    except Exception:
                        continue
            except Exception:
                continue
            finally:
                epoch_iterator.set_description('Epoch {} average training loss: {}'.format(e, current_sum / count))


def main():
    auto_encoder = ImageAutoEncoder()
    auto_encoder.start_train()

if __name__ == '__main__':
    main()
