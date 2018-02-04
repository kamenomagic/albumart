#! /usr/bin/python
import os
import sys

import spotipy
from pymongo import MongoClient
from spotipy import util

from config import *

location_file_name = 'albart_current_scrape_location.txt'
start_year = 2000
# 2018
end_year = 2001
# 100000?
album_count_per_year = 3
# 50
album_count_per_request = 3

token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(auth=token.get_access_token())


mongo = MongoClient()
db = mongo.albart
tracks = db.tracks


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
        year, current_offset = init_location(location_file, location_exists)
        for i in range(year, end_year, 1):
            for j in range(current_offset, album_count_per_year, album_count_per_request):
                current_offset = 0
                finished = scrape_year(i, j)
                save_location(i, j, location_file)
                if finished:
                    break
        print('Completely finished scraping, so deleting location file.')
        os.remove(location_file_name)


def scrape_year(year, offset):
    # Returns true when it has scraped all albums
    albums = get_albums(year, offset=offset, limit=album_count_per_request)
    for album in albums:
        for track in spotify.album_tracks(album['id'])['items']:
            track['analysis'] = spotify.audio_analysis(track['id'])
            track['analysis']['available_markets'] = None
            track['features'] = spotify.audio_features([track['id']])
            tracks.insert_one(track)
            break
    return len(albums) < album_count_per_request


def get_albums(year, query='', entity_type='album', limit=50, offset=0):
    albums = dict(spotify.search(q='year:' + str(year), limit=limit, offset=offset, type='album'))['albums']['items']
    for album in albums:
        album['available_markets'] = None
    return albums


def init_location(location_file, location_exists):
    location_strings = location_file.read().split('-')
    year = int(location_strings[0]) if location_exists else start_year
    current_offset = int(location_strings[1]) if location_exists else 0
    save_location(year, current_offset, location_file)
    if location_exists:
        print('Starting from location {}-{}'.format(year, current_offset))
    return year, current_offset


def save_location(year, current_offset, location_file):
    location_file.seek(0)
    location_file.truncate()
    location_file.writelines('{}-{}'.format(year, current_offset))
    print("Saved location: {}-{}".format(year, current_offset))


if __name__ == '__main__':
    main()
