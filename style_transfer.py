import tensorflow as tf
import utils
import os
import io

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from matplotlib import pyplot as plt

valid_genres = ["rap", "pop", "hiphop", "country", "reggae", "indie"]


def draw_text(img, text, bottom=True):
  outline_color = "white"
  color = "black"

  draw = ImageDraw.Draw(img)
  w, h, font = get_font_size(draw, text)
  if bottom:
    x = (256 - w) / 2
    y = 256 - h
    draw.text((x - 1, y - 1), text, font=font, fill=outline_color)
    draw.text((x + 1, y - 1), text, font=font, fill=outline_color)
    draw.text((x - 1, y + 1), text, font=font, fill=outline_color)
    draw.text((x + 1, y + 1), text, font=font, fill=outline_color)
    draw.text((x, y), text, fill=color, font=font)
  else:
    x = (256 - w) / 2
    y = 0
    draw.text((x - 1, y - 1), text, font=font, fill=outline_color)
    draw.text((x + 1, y - 1), text, font=font, fill=outline_color)
    draw.text((x - 1, y + 1), text, font=font, fill=outline_color)
    draw.text((x + 1, y + 1), text, font=font, fill=outline_color)
    draw.text((x, y), text, fill=color, font=font)


def get_font_size(image_draw, text):
  font = ImageFont.truetype("ebgaramond.ttf", 40)
  w, h = image_draw.textsize(text, font=font)

  count = 0
  while w > 256:
    size = 40 - ((count + 1) * 10)
    font = ImageFont.truetype("ebgaramond.ttf", size)
    w, h = image_draw.textsize(text, font=font)
    count += 1

  return w, h, font


def inference(model, name, artist, img_in, img_out, size=256):
  graph = tf.Graph()

  with graph.as_default():
    with tf.gfile.FastGFile(img_in, 'rb') as f:
      image_data = f.read()
      input_image = tf.image.decode_jpeg(image_data, channels=3)
      input_image = tf.image.resize_images(input_image, size=(size, size))
      input_image = utils.convert2float(input_image)
      input_image.set_shape([size, size, 3])

    with tf.gfile.FastGFile(model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    [output_image] = tf.import_graph_def(graph_def,
                                         input_map={'input_image': input_image},
                                         return_elements=['output_image:0'],
                                         name='output')

  with tf.Session(graph=graph) as sess:
    generated = output_image.eval()
    out_art = Image.open(io.BytesIO(generated))
    draw_text(out_art, name.replace("$", " "))
    draw_text(out_art, artist.replace("$", " "), bottom=False)
    out_art.save(img_out, "JPEG")


def transfer():
  for filename in os.listdir("samples"):
    file_split = filename.split("_")
    genre = file_split[0]
    name = file_split[1]
    artist = file_split[2].split(".")[0]
    inference("./transfer_graphs/album2" + genre + ".pb", name, artist, "./samples/" + filename, "./results/" + filename)


def transfer_genre(genre):
  for filename in os.listdir("samples"):
    file_split = filename.split("_")
    name = file_split[1]
    artist = file_split[2].split(".")[0]
    inference("./transfer_graphs/album2" + genre + ".pb", name, artist, "./samples/" + filename, "./results/" + genre + "/" + filename)


def all_transfer():
  for filename in os.listdir("samples"):
    file_split = filename.split("_")
    name = file_split[1]
    artist = file_split[2].split(".")[0]
    for genre in valid_genres:
      inference("./transfer_graphs/album2" + genre + ".pb", name, artist, "./samples/" + filename, "./results/" + genre + "_from_" + filename)


def main(unused_argv):
  transfer()


if __name__ == '__main__':
  tf.app.run()
