import os
import genius
from tqdm import tqdm
from pymongo import MongoClient

init_file = 'genius_init.txt'
write_every = 100


def main():
    g_api = genius.Genius()
    mongo = MongoClient()
    db = mongo.albart
    tracks = db.tracks
    lyrics_coll = db.lyrics

    location_exists = os.path.exists(init_file)
    if not location_exists:
        with open(init_file, "w+") as f:
            f.write('mongo_start_idx:None')
    # Keep track of Mongo index of last track that lyrics were obtained for.
    # IDs greater than this will be tracks we still need lyrics for
    with open(init_file, "r") as f:
        mongo_start_idx = f.read().split(":")[1]
        if mongo_start_idx == "None":
            mongo_start_idx = tracks.find_one()["_id"]

    idx = 0
    track_iterator = tqdm(tracks.find({'_id': {'$gte': mongo_start_idx}}))
    for track in track_iterator:
        track_iterator.set_description('Track:  {:>100}'.format(track['name'].encode('ascii', 'ignore')))

        # Only add track lyrics we don't already have
        if lyrics_coll.count({'_id': track['_id']}) == 0:
            api_result, perf_artist_match, perf_title_match = g_api.search_song(track['name'], track['artists'][0]['name'])
            lyr_obj = dict()
            # Lyric IDs correspond to track IDs
            lyr_obj['_id'] = track['_id']
            lyr_obj['artist'] = track['artists'][0]['name']
            lyr_obj['track_name'] = track['name']
            lyr_obj['perfect_artist_match'] = perf_artist_match
            lyr_obj['perfect_title_match'] = perf_title_match

            if not api_result:
                db.lyrics_failures.insert_one(lyr_obj)
            else:
                lyr_obj['lyrics'] = api_result.lyrics
                lyrics_coll.insert_one(lyr_obj)
                print("6")
        updated_idx = track['_id']
        mongo_start_idx = str(updated_idx)
        idx += 1
        if idx % write_every == 0:
            with open(init_file, "w+") as f:
                f.write('mongo_start_idx:' + mongo_start_idx)
    with open(init_file, "w+") as f:
        f.write('mongo_start_idx:' + mongo_start_idx)


if __name__ == '__main__':
    main()
