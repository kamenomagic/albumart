#! /usr/bin/python
import os
import sys
from tqdm import tqdm
from tqdm import trange

import spotipy
from pymongo import MongoClient
from spotipy import util

from config import *

start_year = 2017
end_year = 1999
album_count_per_year = 100000
album_count_per_request = 50

token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(auth=token.get_access_token())


mongo = MongoClient()
db = mongo.albart
tracks = db.tracks
location_collection = db.location


def main():
    if len(sys.argv) > 1 and sys.argv[1] in 'fresh':
        print('Are you sure you want to delete the current scraping location and the entire database (y/n)?')
        if sys.stdin.readline().lower().strip() in ['y', 'yes', 'yup', 'yeah']:
            print('If you are really sure you want to delete everything, type "YES".')
            if sys.stdin.readline().strip() in 'YES':
                mongo.drop_database(db)
                print('The database and current location have been deleted.')
        exit()
    year, current_offset = init_location()
    year_iterator = trange(year, end_year, -1, leave=True)
    for i in year_iterator:
        year_iterator.set_description('Year:   {:>100}'.format(i))
        album_iterator = trange(current_offset, album_count_per_year, album_count_per_request, leave=True)
        for j in album_iterator:
            album_iterator.set_description('Offset: {:>100}'.format(j))
            current_offset = 0
            finished = scrape_year(i, j)
            save_location(i, j)
            tqdm.write('Finished and saved location: {}-{}'.format(i, j))
            if finished:
                break


def scrape_year(year, offset):
    # Returns true when it has scraped all albums
    albums = get_albums(year, offset=offset, limit=album_count_per_request)
    album_iterator = tqdm(albums)
    for album in album_iterator:
        try:
            album_iterator.set_description('Album:  {:>100}'.format(album['name'].encode('ascii', 'ignore')))
            track_iterator = tqdm(spotify.album_tracks(album['id'])['items'])
            for track in track_iterator:
                try:
                    track_iterator.set_description('Track:  {:>100}'.format(track['name'].encode('ascii', 'ignore')))
                    analysis = spotify.audio_analysis(track['id'])['track']
                    analysis['codestring'] = None
                    analysis['echoprintstring'] = None
                    analysis['synchstring'] = None
                    analysis['rhythmstring'] = None
                    track['analysis'] = analysis
                    track['analysis']['available_markets'] = None
                    track['features'] = spotify.audio_features([track['id']])
                    track['available_markets'] = None
                    track['images'] = album['images']
                    tracks.insert_one(track)
                except Exception:
                    continue
        except Exception:
            continue
    return len(albums) < album_count_per_request


def get_albums(year, query='', entity_type='album', limit=50, offset=0):
    albums = dict(spotify.search(q='year:' + str(year), limit=limit, offset=offset, type='album'))['albums']['items']
    for album in albums:
        album['available_markets'] = None
    return albums


def init_location():
    location = location_collection.find_one()
    location_exists = location is not None
    location_strings = location['location'].split('-') if location_exists else None
    year = int(location_strings[0]) if location_exists else start_year
    current_offset = int(location_strings[1]) if location_exists else 0
    save_location(year, current_offset, exists=location_exists)
    if not location_exists:
        tqdm.write('Created location file; starting to scrape from location {}-{}'.format(year, current_offset))
    return year, current_offset


def save_location(year, current_offset, exists=True):
    if not exists:
        location_collection.insert_one({'location': '{}-{}'.format(year, current_offset)})
    location_collection.replace_one({'location': {'$exists': True}}, {'location': '{}-{}'.format(year, current_offset)})


if __name__ == '__main__':
    main()
