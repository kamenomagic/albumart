import tensorflow as tf
import numpy as np
import os
from pymongo import MongoClient
import cStringIO
# import io
import base64
from PIL import Image
from tqdm import tqdm, trange


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


class ImageAutoEncoder:
    epochs = 10000
    input_size = [None, 128, 128, 3]
    compressed_size = [None, 16, 16, 32]
    model_directory = '.\image_encoder_trained_model'

    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.x = tf.placeholder(tf.float32, self.input_size, name='x')
        self.encoded_x = tf.placeholder(tf.float32, self.compressed_size, name='encoded_x')

        #Convolution layers
        conv0 = tf.layers.conv2d(self.x, 64, 3, padding="same", activation=lrelu, name='conv0')
        pool0 = tf.layers.max_pooling2d(conv0, 2, 2, padding="same", name='pool0')
        conv1 = tf.layers.conv2d(pool0, 48, 3, padding="same", activation=lrelu, name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding="same", name='pool1')
        conv2 = tf.layers.conv2d(pool1, 32, 3, padding="same", activation=lrelu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding="same", name='pool2')

        # Dense layers
        # flat = tf.reshape(pool2, [-1, 16*16*32])
        # dense0 = tf.layers.dense(flat, units=1024, activation=lrelu, name='dense0')
        # dense1 = tf.layers.dense(dense0, units=512, activation=lrelu, name='dense1')
        self.encoder = pool2
        self.training_decoder = self.build_decoder(self.encoder, reuse=False)
        self.decoder = self.build_decoder(self.encoded_x)
        # self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.training_decoder, labels=self.x)
        self.loss = tf.reduce_sum(tf.abs(self.x - self.training_decoder), name='loss')
        self.cost = tf.reduce_mean(self.loss)
        self.train = tf.train.AdamOptimizer().minimize(self.cost, name='train')
        tf.add_to_collection('x', self.x)
        tf.add_to_collection('encoded_x', self.encoded_x)
        tf.add_to_collection('encoder', self.encoder)
        tf.add_to_collection('decoder', self.decoder)
        tf.add_to_collection('training_decoder', self.training_decoder)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('cost', self.cost)
        tf.add_to_collection('train', self.train)
        self.saver = tf.train.Saver()

    @staticmethod
    def build_decoder(x, reuse=True):
        # decode1 = tf.layers.dense(x, units=1024, activation=lrelu, name='decode1', reuse=reuse)
        # decode0 = tf.layers.dense(decode1, units=16*16*32, activation=lrelu, name='decode0', reuse=reuse)

        # Deconvolution layers
        # blowout = tf.reshape(decode0, [-1, 16, 16, 32])
        deconv2 = tf.layers.conv2d(x, 32, 3, padding="same", activation=lrelu, name='deconv2', reuse=reuse)
        upsamp2 = tf.layers.conv2d_transpose(deconv2, 32, 3, 2, padding="same", name="upsamp2", reuse=reuse)
        upsamp1 = tf.layers.conv2d_transpose(upsamp2, 48, 3, 2, padding="same", name='upsamp1', reuse=reuse)
        upsamp0 = tf.layers.conv2d_transpose(upsamp1, 64, 3, 2, padding="same", name='upsamp0', reuse=reuse)
        decoded = tf.layers.conv2d(upsamp0, 3, 3, padding="same", name='decoded', reuse=reuse)
        # return tf.sigmoid(decoded, name='sigged')
        return decoded

    def start_training(self, load_existing_model=True):
        self.sess.run(tf.global_variables_initializer())
        if load_existing_model and os.path.exists(self.model_directory):
            self.load_trained_model()
        else:
            self.sess.run(tf.global_variables_initializer())
        epoch_iterator = trange(self.epochs)
        for epoch in epoch_iterator:
            current_sum = 0
            count = 0
            batch_size = 10
            batch_count = 0
            batch = []
            try:
                for album in tqdm(MongoClient().albart.albums.find(), total=MongoClient().albart.albums.count()):
                    if batch_count >= batch_size:
                        try:
                            feed_dict = {self.x: batch}
                            loss, _ = self.sess.run([self.cost, self.train], feed_dict=feed_dict)
                            count += 1
                            current_sum += loss
                            batch_count = 0
                            batch = []
                        except Exception as fail:
                            continue
                    else:
                        img = cStringIO.StringIO(base64.b64decode(album["image"]))
                        img = Image.open(img)
                        # buff = io.BytesIO(base64.b64decode(album["image"]))
                        # img = Image.open(buff)
                        # Need to resize image once we have the image selector
                        img.thumbnail((128, 128), Image.ANTIALIAS)
                        resized_img = np.asarray(img)
                        resized_img = resized_img / 255.0
                        batch.append(resized_img)
                        batch_count += 1
                if len(batch) > 0:
                    try:
                        feed_dict = {self.x: batch}
                        loss, _ = self.sess.run([self.loss, self.train], feed_dict=feed_dict)
                        count += 1
                        current_sum += loss
                    except Exception as fail:
                        continue
            except Exception as fail:
                continue
            finally:
                self.save_trained_model()
                epoch_iterator.set_description('Epoch {} average training loss: {}'.format(epoch, (current_sum / count) / batch_size))

    def save_trained_model(self):
        self.saver.save(self.sess, os.path.join(self.model_directory, 'model'))

    def load_trained_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_directory))

    def encode(self, image):
        return self.sess.run(self.encoder, feed_dict={self.x: image})

    def decode(self, encoded_image):
        return self.sess.run(self.decoder, feed_dict={self.encoded_x: encoded_image})

def main():
    auto_encoder = ImageAutoEncoder()
    auto_encoder.start_training()


if __name__ == '__main__':
    main()
