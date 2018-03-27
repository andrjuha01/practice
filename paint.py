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
label1 = None

img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

num_classes = 10
white = (255, 255, 255)

def model():
    global label1
    x = tf.placeholder(tf.float32, [None, img_size_flat])
    Weights = tf.Variable(tf.zeros([784, 10]))
    biases = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, Weights) + biases
    y_true_cls = tf.placeholder(tf.int64, [None])
    y_true = tf.placeholder(tf.float32, [None, num_classes])
    pred = tf.nn.softmax(y)
    pred_cls = tf.argmax(pred, axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,
                                                        labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
    correct_prediction = tf.equal(pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    batch_size = 100
    for i in range(3000):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
    feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))
    img = imageprepare('compressed.png')
    real_img_dict= {x: [img]}
    print len(img)
    print len(data.test.images[0])
    print '__________'
    label1.configure(text=session.run([tf.argmax(pred, 1)], feed_dict={x: [img]})[0])

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

def main():
    global drawing_area, root, white, draw, image1, data, x, label1
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
    image1 = Image.new("RGB", (300, 300), white)
    draw = ImageDraw.Draw(image1)
    root.update_idletasks()
    root.mainloop()

def process():
    global image1, draw
    image1.save('non-compressed.png')
    image1.thumbnail((28, 28),Image.ANTIALIAS)
    image1.save('compressed.png')
    del image1
    image1 = Image.new("RGB", (300, 300), white)
    draw = ImageDraw.Draw(image1)
    f = open('tmp.png', 'wb')
    f.write(bytearray(data.train.images[0]))
    f.close()
    model()

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
            draw.line((xold, yold, event.x, event.y), fill=0, width=15)
                          # here's where you draw it. smooth. neat.
        xold = event.x
        yold = event.y

if __name__ == "__main__":
    main()
