import spotipy
from spotipy import util
from config import *


class Spotify:
    def __init__(self):
        token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.spotify = spotipy.Spotify(token.get_access_token())

    def get_client(self):
        return self.spotify

    def test_search(self, query):
        results = self.spotify.search(q=query, limit=20)
        return [t['name'] for t in results['tracks']['items']]
