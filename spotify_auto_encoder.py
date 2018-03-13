import tensorflow as tf
import numpy as np
import os
import sys
import pymongo
from pymongo import MongoClient
from tqdm import tqdm, trange


class SpotifyAutoEncoder:
    spotify_feature_size = 35
    compressed_size = 100
    epochs = 10000
    model_directory = './spotify_encoder_trained_model'

    def __init__(self):
        self.minimums, self.maximums = self.get_max_and_mins()
        self.difference = [1.0 if n == 0 else n for n in (
                np.array(self.maximums) - np.array(self.minimums))]
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float64, [1, self.spotify_feature_size], name='x')
        self.encoded_x = tf.placeholder(tf.float64, [1, self.compressed_size], name='encoded_x')
        encoder0 = tf.layers.dense(self.x, self.compressed_size, activation=tf.nn.relu, name='encoder0')
        encoder1 = tf.layers.dense(encoder0, self.compressed_size, activation=tf.nn.relu, name='encoder1')
        encoder2 = tf.layers.dense(encoder1, self.compressed_size, activation=tf.nn.relu, name='encoder2')
        encoder3 = tf.layers.dense(encoder2, self.compressed_size, activation=tf.nn.relu, name='encoder3')
        encoder4 = tf.layers.dense(encoder3, self.compressed_size, activation=tf.nn.relu, name='encoder4')
        self.encoder = encoder4
        self.training_decoder = self.build_decoder(self.encoder, self.spotify_feature_size, reuse=False)
        self.decoder = self.build_decoder(self.encoded_x, self.spotify_feature_size)
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
    def build_decoder(x, output_shape, reuse=True):
        decoder0 = tf.layers.dense(x, SpotifyAutoEncoder.compressed_size, activation=tf.nn.relu, reuse=reuse, name='decoder0')
        decoder1 = tf.layers.dense(decoder0, SpotifyAutoEncoder.compressed_size, activation=tf.nn.relu, reuse=reuse, name='decoder1')
        decoder2 = tf.layers.dense(decoder1, SpotifyAutoEncoder.compressed_size, activation=tf.nn.relu, reuse=reuse, name='decoder2')
        decoder3 = tf.layers.dense(decoder2, SpotifyAutoEncoder.compressed_size, activation=tf.nn.relu, reuse=reuse, name='decoder3')
        decoding = tf.layers.dense(decoder3, output_shape, activation=tf.nn.relu, reuse=reuse, name='decoder_out')
        return decoding

    def get_max_or_min(self, maximum=True):
        tracks = MongoClient().albart.tracks
        template = self.json_to_spotify_feature(tracks.find_one(), True)
        aggregator = {'_id': None}
        for key in template:
            aggregator[key.split('.')[-1]] = {'$max' if maximum else '$min': '$' + key}
        results = tracks.aggregate([{'$group': aggregator}]).next()
        del results['_id']
        values = []
        for key, value in results.iteritems():
            results[key] = (0.0 if len(value) == 0 else value[0]) if isinstance(value, list) else value
            results[key] = 0.0 if str(results[key]).isspace() or results[key] == '' else results[key]
        for key in template:
            values.append(results[key.split('.')[-1]])
        return values

    def get_max_and_mins(self):
        return self.get_max_or_min(False), self.get_max_or_min(True)

    def start_training(self, load_existing_model=True):
        if load_existing_model and os.path.exists(self.model_directory):
            self.load_trained_model()
        else:
            self.sess.run(tf.global_variables_initializer())
        epoch_iterator = trange(self.epochs)
        for e in epoch_iterator:
            current_sum = 0
            count = 0
            try:
                for track in MongoClient().albart.tracks.find():
                    try:
                        feature = self.json_to_spotify_feature(track)
                        feed_dict = {self.x: [feature]}
                        loss, _ = self.sess.run([self.loss, self.train], feed_dict=feed_dict)
                        count += 1
                        current_sum += loss
                    except Exception as e:
                        print e
                        continue
            except Exception as e2:
                print e2
                continue
            finally:
                self.save_trained_model()
                epoch_iterator.set_description('Epoch {} average training loss: {}'.format(e, current_sum / count))

    def save_trained_model(self):
        self.saver.save(self.sess, os.path.join(self.model_directory, 'model'))

    def load_trained_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_directory))

    def encode(self, spotify_feature):
        return self.sess.run(self.encoder, feed_dict={self.x: spotify_feature})

    def decode(self, encoded_spotify_feature):
        return self.sess.run(self.decoder, feed_dict={self.encoded_x: encoded_spotify_feature})

    def json_to_spotify_feature(self, json, return_key_structure=False, normalize=True):
        spotify_feature = []
        analysis = json['analysis']
        analysis_keys = analysis.keys()
        for key in analysis_keys:
            if analysis[key] is not None and not isinstance(analysis[key], str):
                spotify_feature.append("analysis." + key if return_key_structure else analysis[key])
        features = json['features'][0]
        features['track_href'] = None
        features['analysis_url'] = None
        features['uri'] = None
        features['type'] = None
        features['id'] = None
        features_keys = features.keys()
        features_keys.sort()
        for key in features_keys:
            if features[key] is not None and not isinstance(features[key], str):
                spotify_feature.append("features." + key if return_key_structure else features[key])
        if not return_key_structure:
            spotify_feature = [0.0 if feature == '' else float(feature) for feature in spotify_feature]
        if return_key_structure:
            return spotify_feature
        else:
            result = np.array(spotify_feature)
            if normalize:
                return (result - np.array(self.minimums)) / self.difference
            else:
                return result


def main():
    auto_encoder = SpotifyAutoEncoder()
    auto_encoder.start_training()


if __name__ == '__main__':
    main()
