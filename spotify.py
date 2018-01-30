import spotipy
from spotipy import util
from config import *
token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(token.get_access_token())

results = spotify.search(q='weezer', limit=20)
for i, t in enumerate(results['tracks']['items']):
    print(' ', i, t['name'])
