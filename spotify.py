import pymongo
from pymongo import MongoClient
import spotipy
from spotipy import util
from config import *
import sys
import os

current_location_file_name = 'albart_current_scrape_location.txt'
genres = ['Hip-Hop', 'Pop', 'Country', 'Latin', 'Electronic/Dance',
          'R&B', 'Rock', 'Christian', 'Classical', 'Indie',
          'Roots & Acoustic', 'K-Pop', 'Metal', 'Reggae', 'Soul',
          'Punk', 'Blues', 'Funk']
token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(token.get_access_token())
mongo = MongoClient()
db = mongo.albart


def main():
    if os.path.exists(current_location_file_name):
        with open(current_location_file_name) as current_location:
            genre_index, current_offset = current_location.readline().split('-')
            print(genre_index, current_offset)
    with open(current_location_file_name, 'w') as current_location:
        current_location.writelines('0-2')
        # for i, genre in enumerate(genres):
        #     current_offset = scrape_genre(genre, current_offset)
        #     current_location.write(i, current_offset)


def scrape_genre(genre, offset):
    album = get_album(genre, offset=offset)
    return offset


def get_album(genre, query='', type='album', limit=50, offset=0):
    query += ' genre:' + genre


def test_search(query):
    # Debugging method, not permanent
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    results = dict(spotify.search(q=query, limit=3, offset=0, type='album'))
    for item in results['albums']['items']:
        item['available_markets'] = None
    album_id = results['albums']['items'][0]['id']
    album_tracks = spotify.album_tracks(album_id)['items']
    for track in album_tracks:
        track_id = track['id']
        # pp.pprint(spotify.audio_analysis(track_id))
        print(spotify.audio_features([track_id]))
        break


if __name__ == '__main__':
    main()
