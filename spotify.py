#! /usr/bin/python
import os
import sys

import spotipy
from pymongo import MongoClient
from spotipy import util

from config import *

location_file_name = 'albart_current_scrape_location.txt'
genres = ['Hip-Hop', 'Pop', 'Country', 'Latin', 'Electronic/Dance', 'R&B', 'Rock', 'Christian', 'Classical', 'Indie',
          'Roots & Acoustic', 'K-Pop', 'Metal', 'Reggae', 'Soul', 'Punk', 'Blues', 'Funk']
album_count_per_genre = 200
album_count_per_request = 50

token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(token.get_access_token())

mongo = MongoClient()
db = mongo.albart
songs = db.songs


def main():
    location_exists = os.path.exists(location_file_name)
    if len(sys.argv) > 1 and sys.argv[1] in 'fresh':
        print('Are you sure you want to delete the current scraping location and the entire database (y/n)?')
        if sys.stdin.readline().lower().strip() in ['y', 'yes', 'yup', 'yeah']:
            print('If you are really sure you want to delete everything, type "YES".')
            if sys.stdin.readline().strip() in 'YES':
                if location_exists:
                    os.remove(location_file_name)
                mongo.drop_database(db)
                print('The database and current location file have been deleted.')
        exit()
    with open(location_file_name, 'r+' if location_exists else 'w+') as location_file:
        genre_index, current_offset = init_location(location_file, location_exists)
        for i in range(genre_index, len(genres), 1):
            for j in range(current_offset, album_count_per_genre, album_count_per_request):
                current_offset = 0
                finished = scrape_genre(genres[i], j)
                save_location(i, j, location_file)
                if finished:
                    break
        print('Completely finished scraping, so deleting location file.')
        os.remove(location_file_name)


def scrape_genre(genre, offset):
    # Returns true when it has scraped all albums
    album = get_albums(genre, offset=offset, limit=album_count_per_request)
    songs.insert_one({'test': 'hello'})
    # Save to db
    return len(album) < album_count_per_request


def get_albums(genre, query='', entity_type='album', limit=50, offset=0):
    query += ' genre:' + genre
    album = ''
    return album


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
    if location_exists:
        print('Starting from location {}-{}'.format(genre_index, current_offset))
    return genre_index, current_offset


def save_location(genre_index, current_offset, location_file):
    location_file.seek(0)
    location_file.truncate()
    location_file.writelines('{}-{}'.format(genre_index, current_offset))
    print("Saved location: {}-{}".format(genre_index, current_offset))


if __name__ == '__main__':
    main()
