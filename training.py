from Tkinter import *
from PIL import ImageDraw, Image, ImageFilter
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

b1 = "up"
xold, yold, drawing_area, root, draw, image1 = None, None, None, None, None, None
data, x = None, None
Weights, biases = None, None
img_size = 28
label = None

img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

num_classes = 10
white = (255, 255, 255)

def cnn_model_fn(features, labels, mode):
  # Input Layer
  print features['x'].shape
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
def imageprepare(name):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(name).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


def process():
    global image1, draw, label1
    image1.save('non-compressed.png')
    image1.thumbnail((28, 28),Image.ANTIALIAS)
    image1.save('compressed.png')
    del image1
    image1 = Image.new("RGB", (300, 300), white)
    draw = ImageDraw.Draw(image1)
    f = open('tmp.png', 'wb')
    f.write(bytearray(data.train.images[0]))
    f.close()
    img = imageprepare('compressed.png')

    classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(img, dtype=np.float32)},
        num_epochs=1,
        batch_size=784,
        shuffle=False)
    eval_results = list(classifier.predict(input_fn=eval_input_fn))
    predicted_classes = [p["probabilities"] * 100 for p in eval_results]
    arr = []
    for idx, val in enumerate(predicted_classes[0]):
        arr.append({'key': idx, 'percentage': val})
    newlist = sorted(arr, key=lambda k: -k['percentage'])
    output = [{p['key']: p['percentage']} for p in newlist[:3]]
    label1.configure(text=str(output))
    print newlist

def clearCanvas():
    drawing_area.delete('all')

def b1down(event):
    global b1
    b1 = "down"           # you only want to draw when the button is down
                          # because "Motion" events happen -all the time-

def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None           # reset the line when you let go of the button
    yold = None

def motion(event):
    if b1 == "down":
        global xold, yold, draw
        if xold is not None and yold is not None:
            event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE, width=10)
            draw.line((xold, yold, event.x, event.y), fill=0, width=50)
                          # here's where you draw it. smooth. neat.
        xold = event.x
        yold = event.y

if __name__ == "__main__":
    global resulting_text, label1
    data = input_data.read_data_sets('IMAGES/', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    data.test.cls = np.array([label.argmax() for label in data.test.labels])
    root = Tk()
    drawing_area = Canvas(root, width=300, height=300)
    drawing_area.pack()
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    clear_btn = Button(root, text="Clear", command=clearCanvas).pack()
    process_btn = Button(root, text="Process", command=process).pack()
    label1 = Label( root, text='Result', relief=RAISED )
    label1.pack()
    print label1
    image1 = Image.new("RGB", (300, 300), white)
    draw = ImageDraw.Draw(image1)
    root.update_idletasks()
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    # Train the model
    #  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #      x={"x": train_data},
    #      y=train_labels,
    #      batch_size=100,
    #      num_epochs=None,
    #      shuffle=True)
    #  mnist_classifier.train(
    #      input_fn=train_input_fn,
    #      steps=20000,
    #      hooks=[logging_hook])
    root.mainloop()

