from bs4 import BeautifulSoup
import urllib2
# import urllib
import requests
import io
import json
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')


def get_images(q, cnt):
    query = q
    image_type = query
    query= query.split()
    query='+'.join(query)
    url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
    # print(url)

    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    soup = get_soup(url,header)

    ActualImages=[]# contains the link for Large original images, type of  image
    for a in soup.find_all("div",{"class":"rg_meta"}):
        link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
        ActualImages.append((link,Type))
    # print("there are total" , len(ActualImages),"images")

    images = []
    for i , (img , Type) in enumerate( ActualImages[:cnt]):
        try:
            #req = urllib.request.Request(img)
            #raw_img = Image.open(urllib.request.urlopen(req))
            req = requests.get(img)
            raw_img = Image.open(io.BytesIO(req.content))
            raw_img.thumbnail((152, 152), Image.ANTIALIAS)
            wid = raw_img.size[0] / 2
            hgt = raw_img.size[1] / 2
            raw_img = raw_img.crop((wid - 64, hgt - 64, wid + 64, hgt + 64))
            images.append(np.asarray(raw_img))
        except Exception as e:
            print("could not load : ", img)
    return images


def main():
    query = "fire"
    fires = get_images(query, 10)
    index = input("Select an image index (0-{}): ".format(len(fires) - 1))
    fire = Image.fromarray(fires[index])
    trees = get_images("tree", 10)
    index = input("Select an image index (0-{}): ".format(len(trees) - 1))
    tree = Image.fromarray(trees[index])
    src = fire.split()
    mask = src[0].point(lambda i: i > 200 and 255)
    pixels = fire.load()
    mask_data = mask.load()
    for i in range(fire.size[0]):
        for j in range(fire.size[1]):
            if mask_data[i, j] == 0:
                pixels[i, j] = (128, 128, 128)
    img = Image.blend(fire, tree, 0.4)

    plt.figure(1)
    plt.imshow(img)
    plt.show()


if __name__=='__main__':
    main()
