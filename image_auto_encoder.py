import tensorflow as tf
import numpy as np
import os
from pymongo import MongoClient
import cStringIO
import base64
from PIL import Image
from tqdm import tqdm, trange

class ImageAutoEncoder:
    epochs = 25
    input_size = [1, 300, 300, 3]
    compressed_size = 256
    model_directory = './image_encoder_trained_model'

    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
        tf.add_to_collection('x', self.x)
        tf.add_to_collection('encoded_x', self.encoded_x)
        tf.add_to_collection('encoder', self.encoder)
        tf.add_to_collection('decoder', self.decoder)
        tf.add_to_collection('training_decoder', self.training_decoder)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('train', self.train)
        self.saver = tf.train.Saver()

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

    def start_training(self, load_existing_model=True):
        self.sess.run(tf.global_variables_initializer())
        if load_existing_model and os.path.exists(self.model_directory):
            self.load_trained_model()
        else:
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
                self.save_trained_model()
                epoch_iterator.set_description('Epoch {} average training loss: {}'.format(e, current_sum / count))

    def save_trained_model(self):
        self.saver.save(self.sess, os.path.join(self.model_directory, 'model'))

    def load_trained_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_directory))

    def encode(self, image):
        return self.sess.run(self.encoder, feed_dict={self.x: image})

    def decode(self, encoded_image):
        return self.sess.run(self.encoder, feed_dict={self.encoded_x: encoded_image})

def main():
    auto_encoder = ImageAutoEncoder()
    auto_encoder.start_training()

if __name__ == '__main__':
    main()
