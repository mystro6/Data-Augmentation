from DataAugmentation import DataAugmentation
import os
from PIL import Image



if __name__ == "__main__":

    path = "/home/ercantunc/Data-Augmentation-master/DataAugmentation/input/"
    inputImages = os.listdir(path)
    """
    #Convert png images to jpg and delete png images
    for image in inputImages:
        im = Image.open(path + image)
        rgb_im = im.convert("RGB")
        head, sep, tail = image.partition('.')
        rgb_im.save(path + head + ".jpg")

    os.system("rm /home/ercantunc/Data-Augmentation-master/DataAugmentation/input/*.png")
    """

    #inputImages = os.listdir(path)

    for image in inputImages:
        head, sep, tail = image.partition('.')
        outputDirectory = "/home/ercantunc/Data-Augmentation-master/DataAugmentation/output/" + head + "/"
        os.system("mkdir /home/ercantunc/Data-Augmentation-master/DataAugmentation/output/" + head +"/")
        augmentation = DataAugmentation()
        augmentation.augmentImages(path + image, image, outputDirectory)



"""

RESIZING IMAGE AND CROPPING IT

    image_decoded = tf.image.decode_jpeg(tf.read_file('/home/tunc/PycharmProjects/Tensorflow/test.jpg'), channels=3)
    cropped = tf.image.resize_image_with_crop_or_pad(image_decoded, 500, 500)
    enc = tf.image.encode_jpeg(cropped)
    fname = tf.constant('/home/tunc/2.jpg')
    fwrite = tf.write_file(fname, enc)

    sess = tf.Session()
    result = sess.run(fwrite)

"""

"""
fix the paths

sudo apt-get install python3-tk
sudo pip install matplotlib
sudo pip install tensorflow

"""