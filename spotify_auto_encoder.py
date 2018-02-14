import tensorflow as tf
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm


class SpotifyAutoEncoder:
    spotify_feature_size = 35

    def __init__(self):
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float16, [self.spotify_feature_size])
        encoder0 = tf.layers.dense(self.x, 128, activation=tf.nn.relu)
        encoder1 = tf.layers.dense(encoder0, 64, activation=tf.nn.relu)
        encoder2 = tf.layers.dense(encoder1, 32, activation=tf.nn.relu)
        encoder3 = tf.layers.dense(encoder2, 16, activation=tf.nn.relu)
        self.encoding = encoder3
        self.training_decoder = self.build_decoder(self.encoding, [self.spotify_feature_size])
        self.decoder = self.build_decoder(self.x, [self.spotify_feature_size])

    @staticmethod
    def build_decoder(x, output_shape, reuse=True):
        decoder0 = tf.layers.dense(x, 32, activation=tf.nn.relu, reuse=reuse)
        decoder1 = tf.layers.dense(decoder0, 32, activation=tf.nn.relu, reuse=reuse)
        decoder2 = tf.layers.dense(decoder1, 64, activation=tf.nn.relu, reuse=reuse)
        decoder3 = tf.layers.dense(decoder2, 128, activation=tf.nn.relu, reuse=reuse)
        decoding = tf.layers.dense(decoder3, output_shape, activation=tf.nn.relu, reuse=reuse)
        return decoding

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        for track in tqdm(MongoClient().albart.tracks.find()):
            if (len(self.json_to_spotify_feature(track))) != 35:
                print(len(self.json_to_spotify_feature(track)))

    @staticmethod
    def json_to_spotify_feature(json):
        spotify_feature = []
        analysis = json['analysis']
        analysis_keys = analysis.keys()
        for key in analysis_keys:
            if analysis[key] is not None and not isinstance(analysis[key], str):
                spotify_feature.append(analysis[key])
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
                spotify_feature.append(features[key])
        return np.array(spotify_feature)


def main():
    auto_encoder = SpotifyAutoEncoder()
    auto_encoder.train()


if __name__ == '__main__':
    main()
