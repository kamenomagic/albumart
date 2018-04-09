#! /usr/bin/python
import urllib2
from bs4 import BeautifulSoup
import requests
import io
import json
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from language import Language

debug = True


class ImageFinder:
    def __init__(self):
        self.language = Language()

    def find(self, lyrics, limit=None):
        nouns = self.language.get_top_nouns(lyrics)
        limited_nouns = nouns[:len(nouns) if limit is None or limit >= len(nouns) else limit]
        if debug:
            print('Nouns: ' + ','.join(nouns))
            if limit is not None:
                print('Limiting to ({}): {}'.format(limit, ','.join(limited_nouns)))
        images = []
        for noun in limited_nouns:
            try:
                images.append((self.get_images(noun, 1)[0], noun))
            except:
                pass
        return images

    @staticmethod
    def get_images(q, cnt):
        query = q
        if debug:
            print('Query: ' + q)
        query = query.split()
        query = '+'.join(query)
        url = "https://www.google.com/search?q=" + query + "&source=lnms&tbm=isch&tbs=itp:clipart,ic:trans"
        if debug:
            print('Query url: ' + url)

        header = {
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        if debug:
            print('Making request...')
        soup = BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')

        if debug:
            print('Parsing response...')
        actual_images = []
        for a in soup.find_all("div", {"class": "rg_meta"}):
            link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
            actual_images.append((link, Type))

        if debug:
            print('Processing images...')
        images = []
        for i, (img, Type) in enumerate(actual_images[:cnt]):
            try:
                req = requests.get(img)
                raw_img = Image.open(io.BytesIO(req.content))
                raw_img.thumbnail((152, 152), Image.ANTIALIAS)
                wid = raw_img.size[0] / 2
                hgt = raw_img.size[1] / 2
                raw_img = raw_img.crop((wid - 64, hgt - 64, wid + 64, hgt + 64))
                images.append(np.asarray(raw_img))
            except Exception as e:
                print("could not load : ", img)
        print('Finished query.')
        return images


def main():
    image_finder = ImageFinder()
    lyrics = "Woah\nWoah\nWoah\nWoah\nWoah\nWoah\nWoah\nWoah\nWoah\n\nLadies and gents," \
             " this is the moment you've waited for (woah)\nBeen searching in the dark," \
             "your sweat soaking through the floor (woah)\nAnd buried in your bones there's an ache that you can't " \
             "ignore\nTaking your breath," \
             " stealing your mind\nAnd all that was real is left behind\n\nDon't fight it," \
             " it's coming for you," \
             " running at ya\nIt's only this moment," \
             " don't care what comes after\nYour fever dream," \
             " can't you see it getting closer\nJust surrender 'cause you feel the feeling taking over\nIt's fire," \
             " it's freedom," \
             " it's flooding open\nIt's a preacher in the pulpit and you'll find devotion\n" \
             "There's something breaking at the brick of every wall\nIt's holding all that you know," \
             "so tell me do you wanna go?\n\nWhere it's covered in all the colored lights\nWhere the runaways are " \
             "running the night\nImpossible comes true," \
             " it's taking over you\nOh," \
             " this is the greatest show\nWe light it up," \
             " we won't come down\nAnd the sun can't stop us now\nWatching it come true," \
             " it's taking over you\nOh," \
             " this is the greatest show\n\n(Woah) colossal we come these renegades in the ring\n" \
             "(Woah) where the lost get found in the crown of the circus king\n\nDon't fight it," \
             " it's coming for you," \
             " running at ya\nIt's only this moment," \
             " don't care what comes after\nIt's blinding," \
             " outshining anything that you know\nJust surrender 'cause you're calling and you wanna go\n\n" \
             "Where it's covered in all the colored lights\nWhere the runaways are running the night\nImpossible " \
             "comes true," \
             " intoxicating you\nOh," \
             " this is the greatest show\nWe light it up," \
             " we won't come down\nAnd the sun can't stop us now\nWatching it come true," \
             " it's taking over you\nOh," \
             " this is the greatest show\n\nIt's everything you ever want\nIt's everything you ever need\n" \
             "And it's here right in front of you\nThis is where you wanna be (This is where you wanna be)\n" \
             "It's everything you ever want\nIt's everything you ever need\n" \
             "And it's here right in front of you\nThis " \
             "is where you wanna be\nThis is where you wanna be\n\nWhere it's covered in all the colored " \
             "lights\nWhere the runaways are running the night\nImpossible comes true," \
             " it's taking over you\nOh," \
             " this is the greatest show\nWe light it up," \
             " we won't come down\nAnd the sun can't stop us now\nWatching it come true," \
             " it's taking over you\nThis is the greatest show\nWhere it's covered in all the colored lights\n" \
             "Where the runaways are running the night\nImpossible comes true," \
             " it's taking over you\nOh," \
             " this is the greatest show\nWe light it up," \
             " we won't come down\nAnd the walls can't stop us now\nI'm watching it come true," \
             " it's taking over you\nOh," \
             "this is the greatest show\n\n'Cause everything you want is right in front of you\nAnd you see the " \
             "impossible is coming true\nAnd the walls can't stop us (now) now," \
             "yeah\n\nThis is the greatest show (Oh!)\nThis is the greatest show (Oh!)\nThis is the greatest show (" \
             "Oh!)\nThis is the greatest show (Oh!)\nThis is the greatest show (Oh!)\nThis is the greatest show (" \
             "Oh!)\n(This is the greatest show)\nThis is the greatest show (Oh!)\nThis is the greatest show! "
    for image_noun in image_finder.find(lyrics, 5):
        try:
            print(image_noun[1])
            plt.figure(1)
            plt.savefig('test.png')
            plt.imshow(image_noun[0])
            plt.show()
        except:
            pass


if __name__ == '__main__':
    main()
