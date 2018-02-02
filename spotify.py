import pymongo
from pymongo import MongoClient
import spotipy
from spotipy import util
from config import *
import sys
import os
import time

location_file_name = 'albart_current_scrape_location.txt'
genres = ['Hip-Hop', 'Pop', 'Country', 'Latin', 'Electronic/Dance',
          'R&B', 'Rock', 'Christian', 'Classical', 'Indie',
          'Roots & Acoustic', 'K-Pop', 'Metal', 'Reggae', 'Soul',
          'Punk', 'Blues', 'Funk']
token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(token.get_access_token())
mongo = MongoClient()
db = mongo.albart
songs = db.songs


def main():
    location_exists = os.path.exists(location_file_name)
    with open(location_file_name, 'r+' if location_exists else 'w+') as location_file:
        genre_index, current_offset = init_location(location_file, location_exists)
        for i in range(genre_index, len(genres), 1):
            genre = genres[i]
            current_offset = scrape_genre(genre, current_offset)
            save_location(i, current_offset, location_file)


def scrape_genre(genre, offset):
    album = get_album(genre, offset=offset)
    #Save to db
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


def init_location(location_file, location_exists):
    location_strings = location_file.read().split('-')
    genre_index = int(location_strings[0]) if location_exists else 0
    current_offset = int(location_strings[1]) if location_exists else 0
    save_location(genre_index, current_offset, location_file)
    return genre_index, current_offset


def save_location(genre_index, current_offset, location_file):
    location_file.seek(0)
    location_file.truncate()
    location_file.writelines('{}-{}'.format(genre_index, current_offset))
    print("Saved location: {}-{}".format(genre_index, current_offset))


if __name__ == '__main__':
    main()
