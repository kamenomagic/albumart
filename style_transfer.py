import tensorflow as tf
import utils
import os

valid_genres = ["rap", "pop", "hiphop", "country", "reggae", "indie"]

def inference(model, img_in, img_out, size=256):
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
    with open(img_out, 'wb') as f:
      f.write(generated)


def transfer():
  for filename in os.listdir("samples"):
    genre = filename.split("_")[0]
    inference("./transfer_graphs/album2" + genre + ".pb", "./samples/" + filename, "./results/" + filename)


def all_transfer():
  for filename in os.listdir("samples"):
    for genre in valid_genres:
      inference("./transfer_graphs/album2" + genre + ".pb", "./samples/" + filename, "./results/" + genre + "_from_" + filename)


def main(unused_argv):
  transfer()
  #all_transfer()


if __name__ == '__main__':
  tf.app.run()