import numpy as np
import tensorflow as tf
import time
import spotify_auto_encoder
import image_auto_encoder
from pymongo import MongoClient
import cStringIO
import base64
from PIL import Image
import random
import os


class CGAN:
    def __init__(self, sess, dim_img=128, n_batches=10000, batch_size=8, model_dir="./cgan_model"):
        self.sess = sess
        self.dim_img = dim_img
        self.model_dir = model_dir

        self.dim_clas = 8
        self.dim_enc_img = 192
        self.dim_rand = 128
        self.dim_z = self.dim_rand  # + self.dim_enc_img + self.dim_clas  # Dimension of latent z space (size of inputs to generator)
        self.n_filters = 256

        self.LR = 0.001  # 0.005
        self.lr_d = 0.005  # 0.01
        self.n_batches = n_batches
        self.batch_size = batch_size  # 4
        self.dis_iters = 5  # discriminator iterations for every generator iteration
        self.lmbda = 10  # gradient penalty multiplier
        # Adam Optimizer parameters
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.n_channels = 3
        #self.song_auto_encoder = spotify_auto_encoder.SpotifyAutoEncoder()
        #self.song_auto_encoder.load_trained_model()
        self.img_auto_encoder = image_auto_encoder.ImageAutoEncoder()
        #self.img_auto_encoder.load_trained_model()
        self.mongodb = MongoClient().albart

        with tf.variable_scope("gan_vars") as scope:
            self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.dim_img, self.dim_img, self.n_channels])
            self.noise = tf.placeholder(tf.float32, [self.batch_size, self.dim_z], name='noise')
            self.real_clas = tf.placeholder(tf.float32, [self.batch_size, self.dim_clas], name='real_class')
            self.clas = tf.placeholder(tf.float32, [self.batch_size, self.dim_clas], name='clas')
            #self.noise_class = tf.concat([self.noise, self.clas], 1)
            self.fake_data = self.generator(noise=self.noise)

            dis_real = self.discriminator(self.real_data, self.real_clas)
            scope.reuse_variables()
            dis_fake = self.discriminator(self.fake_data, self.clas)

            self.gen_cost = -tf.reduce_mean(dis_fake)
            self.dis_cost = tf.reduce_mean(dis_fake) - tf.reduce_mean(dis_real)

            epsilon = tf.random_uniform(
                shape=[self.batch_size],
                minval=0,
                maxval=1.
            )

            x_hat = epsilon * tf.transpose(self.real_data) + (1 - epsilon) * tf.transpose(self.fake_data)
            x_hat = tf.transpose(x_hat)
            x_hat_clas = epsilon * tf.transpose(self.real_clas) + (1 - epsilon) * tf.transpose(self.clas)
            x_hat_clas = tf.transpose(x_hat_clas)
            gradients = tf.gradients(self.discriminator(x_hat, x_hat_clas), [x_hat])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.dis_cost += self.lmbda * gradient_penalty

        self.dis_var_list = self.get_trainable_vars_like('dis_', 'gan_vars')
        self.gen_var_list = self.get_trainable_vars_like('gen_', 'gan_vars')

        with tf.variable_scope("gan_vars") as scope:
            self.dis_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_d, beta1=self.beta1, beta2=self.beta2, name='d_adam').minimize(
                self.dis_cost, var_list=self.dis_var_list)
            self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.LR, beta1=self.beta1, beta2=self.beta2, name='g_adam').minimize(
                self.gen_cost, var_list=self.gen_var_list)

        #  self.create_vars_collection()
        self.saver = tf.train.Saver(self.dis_var_list + self.gen_var_list)

    def init_model(self):
        if os.path.exists(self.model_dir) and os.listdir(self.model_dir) != []:
            self.load_model()
            print "loaded model"
        else:
            self.train()

    def create_vars_collection(self):
        for var in self.dis_var_list:
            tf.add_to_collection(var.name, var)
        for var in self.gen_var_list:
            tf.add_to_collection(var.name, var)

    def load_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model'))

    def whiten_data(self, data, mean, std_dev):
        if mean is None:
            mean = np.mean(data)
        if std_dev is None:
            std_dev = np.std(data)

        data = data - mean
        data = data / std_dev
        return data, mean, std_dev

    def conv(self, x, filter_size=4, stride=2, num_filters=64, is_output=False, use_bn=True, name="conv"):
        with tf.name_scope(name) as scope:
            x_shape = x.get_shape().as_list()
            w = tf.get_variable(name + "_filter", shape=[filter_size, filter_size, x_shape[3], num_filters],
                                initializer=tf.random_normal_initializer)
            b = tf.get_variable(name + "_bias", shape=[num_filters],
                                initializer=tf.random_normal_initializer)
            do_conv = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding="SAME")
            add_bias = tf.add(do_conv, b)
            if not is_output:
                if use_bn:
                    relu = tf.nn.relu(add_bias)
                    bn = self.batch_norm(relu, name)
                    return bn
                else:
                    relu = tf.nn.relu(add_bias)
                    return relu
            else:
                return add_bias

    def conv_transpose(self, x, filter_size=4, stride=2, num_filters=None, output_shape=None, is_output=False, name="conv_t"):
        if output_shape is None:
            output_shape = [int(x.shape[0]), int(x.shape[1]) * 2, int(x.shape[2]) * 2, int(x.shape[3]) // 2]
        if num_filters is None:
            num_filters = int(x.shape[3]) // 2

        with tf.name_scope(name) as scope:
            x_shape = x.get_shape().as_list()
            w = tf.get_variable(name + "_filter", shape=[filter_size, filter_size, num_filters, x_shape[3]],
                                initializer=tf.random_normal_initializer)
            b = tf.get_variable(name + "_bias", shape=[num_filters],
                                initializer=tf.random_normal_initializer)
            do_conv = tf.nn.conv2d_transpose(x, w, output_shape, [1, stride, stride, 1])
            add_bias = tf.add(do_conv, b)

            if not is_output:
                relu = tf.nn.relu(add_bias)
                bn = self.batch_norm(relu, name)
                return bn
            return add_bias

    def fc(self, x, dim_out, use_relu=False, name="fc"):
        with tf.name_scope(name) as scope:
            # flatten all but the batch dimension
            shape = x.get_shape().as_list()
            dim = np.prod(shape[1:])
            x = tf.reshape(x, [-1, dim])
            w = tf.get_variable(name + "_weights", shape=[dim, dim_out],
                                initializer=tf.random_normal_initializer)
            b = tf.get_variable(name + "_bias", shape=[dim_out],
                                initializer=tf.random_normal_initializer)
            net = tf.add(tf.matmul(x, w), b)
            if use_relu:
                relu = tf.nn.relu(net)
                # bn = batch_norm(relu, name, global_norm=False)
                return relu
            return net

    def batch_norm(self, x, name, global_norm=True):
        dims = [0]
        if global_norm:
            dims = [0, 1, 2]
        moments = tf.nn.moments(x, dims, name=name + "_bn_moments", keep_dims=True)
        bn = tf.nn.batch_normalization(x, moments[0], moments[1], None, None, 1e-6, name=name + "_bn")
        return bn

    def discriminator(self, input_data, clas):
        conv0 = self.conv(input_data, num_filters=self.n_filters / 8, use_bn=False, name="dis_conv0")
        conv1 = self.conv(conv0, num_filters=self.n_filters / 4, name="dis_conv1")
        conv2 = self.conv(conv1, num_filters=self.n_filters / 2, name="dis_conv2")
        conv3 = self.conv(conv2, num_filters=self.n_filters, name="dis_conv3")
        fc0 = self.fc(conv3, 1, name="dis_fc0")
        #fc1 = self.fc(tf.concat([fc0, clas], 1), 1, name="dis_fc1")
        return fc0

    def generator(self, noise=None):
        if noise is None:
            raise Exception()
        new_dim = self.dim_img / 16

        fc0 = self.fc(noise, new_dim * new_dim * self.n_filters, use_relu=False, name="gen_fc")
        fc0_reshape = tf.reshape(fc0, [int(fc0.shape[0]), new_dim, new_dim, self.n_filters], name="gen_fc_rs")
        conv_t0 = self.conv_transpose(fc0_reshape, name="gen_conv_t0")
        conv_t1 = self.conv_transpose(conv_t0, name="gen_conv_t1")
        conv_t2 = self.conv_transpose(conv_t1, name="gen_conv_t2")
        conv_t3 = self.conv_transpose(conv_t2, num_filters=self.n_channels,
                                 output_shape=[int(conv_t2.shape[0]), int(conv_t2.shape[1]) * 2,
                                               int(conv_t2.shape[2]) * 2,
                                               self.n_channels],
                                 is_output=True,
                                 name="gen_conv_t3")
        tanh = tf.nn.tanh(conv_t3)
        return tanh

    def get_rand_imgs(self):
        images = np.zeros((self.batch_size, self.dim_img, self.dim_img, self.n_channels))
        encoded_tracks = np.zeros((self.batch_size, self.dim_clas))

        album_cursor = self.sample_albums(self.batch_size)
        index = 0
        for album in album_cursor:
            img = cStringIO.StringIO(base64.b64decode(album["image"]))
            img = Image.open(img)
            img = img.resize((self.dim_img, self.dim_img))
            img = np.array(img)
            images[index] = img
            #features = self.song_auto_encoder.json_to_spotify_feature(track)
            #encoded_tracks[index] = self.song_auto_encoder.encode(features.reshape((1, features.shape[0])))
            index += 1
            if index == self.batch_size:
                break

        return images, encoded_tracks

    def one_hot_encode(self, class_info, n_classes):
        new_info = np.zeros((class_info.shape[0], n_classes))
        for i in range(class_info.shape[0]):
            cl = class_info[i]
            new_info[i][int(cl)] = 1
        return new_info

    @staticmethod
    def get_trainable_vars_like(like_str, scope_name):
        like_str = scope_name + "/" + like_str
        vars_like = []
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name):
            if i.name.startswith(like_str):
                vars_like.append(i)
        return vars_like

    @staticmethod
    def fake_encode_img(img):
        #img = Image.open(img)
        img = img.resize((8, 8))
        img = np.asarray(img)
        return img.flatten()

    def get_seed_imgs(self):
        # Use NLP to get related images here.
        imgs = np.zeros((self.batch_size, self.dim_enc_img))
        for i in range(self.batch_size):
            index = random.randint(0, 4)
            img = Image.open("test_imgs/" + str(index) + ".jpeg")
            img = img.resize((self.dim_img, self.dim_img))
            #img = np.array(img)
            imgs[i] = self.fake_encode_img(img)
        return imgs

    def call_generator(self, batch_size, dim_z, summarize=False):
        noise = tf.get_default_graph().get_tensor_by_name('gan_vars/noise:0')
        my_class = tf.get_default_graph().get_tensor_by_name('gan_vars/clas:0')
        #noise_class = tf.concat([noise, my_class], 1)
        gen = self.generator(noise=noise)
        gen = ((gen / 2) + 0.5) * 255
        if summarize:
            summary = tf.summary.image("gen_img_", tf.reshape(gen[0], [1, self.dim_img, self.dim_img, self.n_channels]))
            return summary
        return gen

    def get_gen_imgs(self):
        with tf.variable_scope("gan_vars", reuse=True) as scope:
            _, noize, enc_tracks = self.gen_dis_inputs()
            imgs = self.sess.run(self.call_generator(self.batch_size, self.dim_z, summarize=False),
                             {self.noise: noize, self.clas: enc_tracks})
        return imgs

    def sample_albums(self, n):
        # Mongo "$sample" is erroring out with a small sample size, so sample more and just take what we need
        return self.mongodb.albums.aggregate([{"$match": {"genres": "rap"}}, {"$sample": {"size": 20 + n}}])

    def gen_dis_inputs(self):
        imgs, enc_tracks = self.get_rand_imgs()
        enc_seed_imgs = self.get_seed_imgs()
        gen_class = np.append(enc_seed_imgs, enc_tracks, axis=1)
        noize = np.random.uniform(-1.0, 1.0, (self.batch_size, self.dim_rand)).astype(np.float32)
        #noize = np.append(gen_class, noize, axis=1)
        imgs = 2 * ((imgs.astype(np.float32) / 255) - 0.5)
        return imgs, noize, enc_tracks

    def train(self):
        time_str = time.strftime("%d%b%Y-%H:%M:%S", time.gmtime())
        with tf.variable_scope("gan_vars", reuse=True) as scope:
            writer = tf.summary.FileWriter('./logs/' + time_str)
            self.sess.run(tf.global_variables_initializer())

            for i in range(self.n_batches):
                print(str(i) + " of " + str(self.n_batches))
                self.train_batch()
                # Used to track progress of generator, write a generated image to Tensorboard summary
                if i % 100 == 0:
                    self.save_model()
                    _, noize, enc_tracks = self.gen_dis_inputs()
                    summary = self.sess.run(self.call_generator(self.batch_size, self.dim_z, summarize=True),
                                    {self.noise: noize, self.clas: enc_tracks})
                    writer.add_summary(summary, i)

    def train_batch(self):
        # Run "dis_iters" discriminator iterations for each 1 generator iteration
        for j in range(self.dis_iters):
            imgs, noize, enc_tracks = self.gen_dis_inputs()
            d_cost, _ = self.sess.run([self.dis_cost, self.dis_train_op],
                                 feed_dict={self.real_data: imgs, self.noise: noize,
                                            self.clas: enc_tracks, self.real_clas: enc_tracks})
            print("d_cost:" + str(d_cost))
            if j + 1 % self.dis_iters == 0:
                dis_loss_vs_time = d_cost

        _, noize, enc_tracks = self.gen_dis_inputs()
        g_cost, _ = self.sess.run([self.gen_cost, self.gen_train_op], feed_dict={self.noise: noize,
                                                                            self.clas: enc_tracks})
        print("g_cost:" + str(g_cost))
        #gen_loss_vs_time[i] = g_cost




def main():
    with tf.Session() as sess:
        cgan = CGAN(sess)
        cgan.train()
        # config=tf.ConfigProto(device_count={'GPU': 0})


if __name__ == '__main__':
    main()
