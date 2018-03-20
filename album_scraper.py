import os
from pymongo import MongoClient
from bson import ObjectId
from tqdm import tqdm
import pprint
import spotipy
from spotipy import util
from config import *


init_file = 'album_scraper_init.txt'
write_every = 100

def main():
    token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    spotify = spotipy.Spotify(auth=token.get_access_token())

    mongo = MongoClient()
    db = mongo.albart
    tracks = db.tracks
    albums = db.albums

    location_exists = os.path.exists(init_file)
    if not location_exists:
        with open(init_file, "w+") as f:
            f.write("start_idx:None")
    # Keep track of Mongo index of last track that album was obtained for.
    # IDs greater than this will be tracks we still need album for
    with open(init_file, "r") as f:
        start_idx = f.read().split(":")[1].strip()
        if start_idx == "None":
            start_idx = tracks.find_one()["_id"]
        else:
            start_idx = ObjectId(start_idx)

    idx = 0

    track_iterator = tqdm(tracks.find({'_id': {'$gte': start_idx}}))
    for track in track_iterator:
        track_iterator.set_description('Track:  {:>100}'.format(track['name'].encode('ascii', 'ignore')))

        if not track.get("album_id"):
            continue

        if not track.get("uri"):
            db.album_failures.insert_one({"_id": track["_id"], "artist": track['artists'][0]['name'], "track_name": track["name"]})
            continue

        album = albums.find({"_id": track["album_id"]})
        if album.count() == 0:
            db.album_failures.insert_one({"_id": track["_id"], "artist": track['artists'][0]['name'], "track_name": track["name"]})
            continue
        else:
            if not album[0].get("name"):
                alb = album[0]
                full_track_obj = spotify.track(track["uri"])
                try:
                    artist_uri = full_track_obj["artists"][0]["uri"]
                    artist = spotify.artist(artist_uri)
                    album_uri = full_track_obj["album"]["uri"]
                    album = spotify.album(album_uri)

                    genres = artist["genres"]
                    albums.update_one({"_id": alb["_id"]}, {"$set": {"name": album["name"], "artist": artist["name"],
                                                                     "uri": album["uri"], "genres": genres}})
                except KeyError:
                    db.album_failures.insert_one(
                        {"_id": track["_id"], "artist": track['artists'][0]['name'], "track_name": track["name"]})
                    continue
            else:
                continue
        # Update each track so that it points to album object
        updated_idx = track["_id"]
        start_idx = str(updated_idx)
        idx += 1
        if idx % write_every == 0:
            with open(init_file, "w+") as f:
                f.write("start_idx:" + str(start_idx))
    with open(init_file, "w+") as f:
        f.write("start_idx:" + str(start_idx))


if __name__ == '__main__':
    main()
