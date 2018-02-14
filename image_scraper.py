import os
from pymongo import MongoClient
from PIL import Image
import base64
import cStringIO
from tqdm import tqdm
import urllib2
import bson

init_file = 'image_scraper_init.txt'
write_every = 100

def main():
    mongo = MongoClient()
    db = mongo.albart
    tracks = db.tracks
    albums = db.albums

    location_exists = os.path.exists(init_file)
    if not location_exists:
        with open(init_file, "w+") as f:
            f.write("start_idx:None")
    # Keep track of Mongo index of last track that lyrics were obtained for.
    # IDs greater than this will be tracks we still need lyrics for
    with open(init_file, "r") as f:
        start_idx = f.read().split(":")[1]
        if start_idx == "None":
            start_idx = tracks.find_one()["_id"]

    idx = 0
    track_iterator = tqdm(tracks.find({'_id': {'$gte': start_idx}}))
    for track in track_iterator:
        track_iterator.set_description('Track:  {:>100}'.format(track['name'].encode('ascii', 'ignore')))

        if not track["images"]:
            db.image_failures.insert_one({"_id": track["_id"], "artist": track['artists'][0]['name'], "track_name": track["name"]})
            continue
        for image in track["images"]:
            if image["width"] == 300 and image["height"] == 300:
                target_image = image
                break
        if not target_image:
            db.image_failures.insert_one({"_id": track["_id"], "artist": track['artists'][0]['name'], "track_name": track["name"]})
            continue

        album = albums.find({"url": target_image["url"]})
        if album.count() == 0:
            buff = cStringIO.StringIO()
            img = Image.open(urllib2.urlopen(target_image["url"]))
            img.save(buff, format="JPEG")
            img_data = base64.b64encode(buff.getvalue())
            album_id = albums.insert_one({"url": target_image["url"], "image": img_data}).inserted_id
        else:
            album_id = album[0]["_id"]
        # Update each track so that it points to album object
        tracks.update_one({"_id": track["_id"]}, {"$set": {"album_id": album_id}})
        updated_idx = track["_id"]
        start_idx = str(updated_idx)
        idx += 1
        if idx % write_every == 0:
            with open(init_file, "w+") as f:
                f.write("start_idx:" + start_idx)
    with open(init_file, "w+") as f:
        f.write("start_idx:" + start_idx)


if __name__ == '__main__':
    main()
