from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import tensorflow as tf
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt
import random
import cgan
import image_finder
import genius_api


class Collager:
    def __init__(self, sess, model_dir="./cgan_model", max_objs=4, result_dim=256, font="ebgaramond.ttf", font_size=40):
        self.sess = sess
        self.image_finder = image_finder.ImageFinder()
        self.db = MongoClient().albart
        self.cgan_model_dir = model_dir
        self.cgan = cgan.CGAN(self.sess, n_batches=150)
        self.cgan.init_model()
        self.lyrics_api = genius_api.LyricsApi()
        self.max_objs = max_objs
        self.result_dim = result_dim
        self.font_size = font_size
        self.font_string = font
        self.font = ImageFont.truetype(font, font_size)

        self.allowed_genres = ["country", "pop", "rap", "reggae", "indie", "hip hop"]

    def create_collages(self):
        backgrounds = self.cgan.get_gen_imgs()
        tracks = self.sample_tracks(backgrounds.shape[0])
        if tracks is None:
            return None

        count = 0
        for track in tracks:
            init_img = backgrounds[count]
            background = Image.fromarray(init_img.astype(np.uint8))
            background = background.resize((self.result_dim, self.result_dim))

            name = track["name"]
            artist = track["artists"][0]["name"]
            lyrics = self.get_lyrics(artist, name)
            if lyrics is None:
                count += 1
                continue

            genres = self.get_genres(track["album_id"])
            genre = self.allowed_genre(genres)
            if genre is None:
                count += 1
                continue

            n_objs = random.randint(1, self.max_objs)
            objects = self.image_finder.find(lyrics, n_objs)

            obj_downsample = 1 + len(objects)
            img_locs = []
            for obj in objects:
                size = self.result_dim // obj_downsample,
                word = obj[1]
                obj = obj[0]
                obj = Image.fromarray(obj)
                obj.thumbnail((size, size))

                x, y = self.get_image_loc(obj.size, img_locs)
                self.paste_image(background, x, y, obj)
                img_locs.append((x, y, size))

            self.draw_text(background, name)
            self.draw_text(background, artist, bottom=False)

            background.show()
            background.save("samples/" + genre.replace(" ", "") + "_" +
                            name.replace(" ", "") + "_" + artist.replace(" ", "") + ".jpg")

            if count == backgrounds.shape[0]:
                break

    @staticmethod
    def paste_image(background, x, y, obj):
        background.paste(obj, (x, y), obj)

    def get_image_loc(self, obj_size, img_locs):
        x, y = np.random.randint(0, self.result_dim // 2), np.random.randint(0, self.result_dim // 2)
        x += (self.result_dim // 4) - (obj_size[0] // 2)
        y += (self.result_dim // 4) - (obj_size[1] // 2)
        return x, y
        #for x, y, size in img_locs:




    def draw_text(self, img, text, bottom=True, init_img=None):
        outline_color = "white"
        color = "black"

        text = text.encode("ascii", "ignore")
        draw = ImageDraw.Draw(img)
        w, h, font = self.get_font_size(draw, text)
        if bottom:
            x = (self.result_dim - w) / 2
            y = self.result_dim - h
            draw.text((x - 1, y - 1), text, font=font, fill=outline_color)
            draw.text((x + 1, y - 1), text, font=font, fill=outline_color)
            draw.text((x - 1, y + 1), text, font=font, fill=outline_color)
            draw.text((x + 1, y + 1), text, font=font, fill=outline_color)
            draw.text((x, y), text, fill=color, font=font)
        else:
            x = (self.result_dim - w) / 2
            y = 0
            draw.text((x - 1, y - 1), text, font=font, fill=outline_color)
            draw.text((x + 1, y - 1), text, font=font, fill=outline_color)
            draw.text((x - 1, y + 1), text, font=font, fill=outline_color)
            draw.text((x + 1, y + 1), text, font=font, fill=outline_color)
            draw.text((x, y), text, fill=color, font=font)

    def get_font_size(self, image_draw, text):
        font = self.font
        w, h = image_draw.textsize(text, font=font)

        count = 0
        while w > self.result_dim:
            size = self.font_size - ((count + 1) * 10)
            font = ImageFont.truetype(self.font_string, size)
            w, h = image_draw.textsize(text, font=font)
            count += 1

        return w, h, font


    def get_track(self, name):
        return self.db.tracks.find_one({"name": name})

    def get_lyrics(self, artist_name, track_name):
        #return self.db.lyrics.find_one({"_id": t_id})["lyrics"]
        return self.lyrics_api.get_lyrics(artist_name, track_name)

    def get_genres(self, album_id):
        return self.db.albums.find_one({"_id": album_id})["genres"]

    def sample_tracks(self, n):
        # Mongo "$sample" is erroring out with a small sample size, so sample more and just take what we need
        return self.db.tracks.aggregate([{"$sample": {"size": 20 + n}}])
        #  return self.db.tracks.aggregate([{"$match": {"genres": "rap"}}, {"$sample": {"size": 20 + n}}])

    def allowed_genre(self, genres):
        for genre in genres:
            if genre in self.allowed_genres:
                return genre
        return None





def main():
    with tf.Session() as sess:
        collager = Collager(sess)
        collager.create_collages()
        # config=tf.ConfigProto(device_count={'GPU': 0})


if __name__ == '__main__':
    main()


