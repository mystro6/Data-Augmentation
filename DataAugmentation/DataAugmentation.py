
import PIL.Image as Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot
import os
import re


class DataAugmentation:

    def __init__(self):
        self.rotatedImages = []
        self.rotatedTensors = []
        self.images = []
        self.tensorImages = []

    def augmentImages(self,imagePath,fileName,outputDirectory):

        anglesToRotate = [45,90,270,315]
        fileName = re.sub("\.jpg", "", fileName)

        for angle in anglesToRotate:
            self.rotateImage(imagePath, angle, fileName,outputDirectory)
        """
        #TO FLIP IMAGES
        for tensor, path in zip(self.rotatedTensors, self.rotatedImages):

            rotatedFileName = os.path.basename(path)
            rotatedFileName = re.sub("\.jpg", "", rotatedFileName)
            tensor = self.horizontallyFlip(tensor)
            self.saveTensor(tensor, rotatedFileName, "_flipped", outputDirectory)
        
        """
        self.images = os.listdir(outputDirectory)

        for image in self.images:
            self.tensorImages.append(self.convertToTensor(outputDirectory + image))

        for tensor, imagePath in zip(self.tensorImages, self.images):
            print(imagePath)
            brightenTensor = self.brightenImage(tensor)
            self.saveTensor(brightenTensor, imagePath, "_brighten", outputDirectory)

        self.images = os.listdir(outputDirectory)

        for imageName in self.images:
            #print(outputDirectory + imageName)
            loadedImage = Image.open(outputDirectory + imageName)
            noiseAddedImage = self.noise_generator("gauss", self.PIL2array(loadedImage))
            self.array2PIL(noiseAddedImage,loadedImage.size).save(outputDirectory + imageName + "_noise.jpg")

    def rotateImage(self, imagePath, angle, fileName, outputDirectory):

        photo = Image.open(imagePath)

        resultPath = outputDirectory + fileName + "_rotated" + str(angle) + ".jpg"
        result = photo.rotate(angle)
        result.save(resultPath)
        self.rotatedImages.append(resultPath)
        self.rotatedTensors.append(self.convertToTensor(resultPath))

    def convertToTensor(self, path):

        img = pyplot.imread(path)
        return tf.convert_to_tensor(img)

    def horizontallyFlip(self, tensor):

        return tf.image.flip_left_right(tensor)

    #delta is the value how much we brighten or darken the image
    # if delta is less then 0 then image darkens
    def brightenImage(self, tensor):
        print("brighten")
        return tf.image.adjust_brightness(tensor, -0.4)

    def saveTensor(self, tensor, fileName, func, outputDirectory):

        jpgEncoded = tf.image.encode_jpeg(tensor)

        fileName = re.sub("\.jpg", "", fileName)
        print("SAVE")
        print(fileName)
        print(outputDirectory)
        print(outputDirectory + fileName + func + ".jpg")
        fname = tf.constant(outputDirectory + fileName + func + ".jpg")
        fwrite = tf.write_file(fname, jpgEncoded)

        sess = tf.Session()
        result = sess.run(fwrite)


    """
    def addGaussianNoise(self,tensor):
        with tf.name_scope('Add_gaussian_noise'):
            noise = tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=(50) / (255), dtype=tf.float32)
            noise_img = tensor + noise
            noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)

        return noise_img
    """
    """
        IN augmentImages method:
                For a directory full of images
                ------------------------------
                
                imageNames = os.listdir(imageSetPath)

                for path, angle in zip(imageNames,anglesToTurn):
                    self.rotateImage(path,angle)
    """

    def noise_generator(self, noise_type, image):
        """
        Generate noise to a given Image based on required noise type

        Input parameters:
            image: ndarray (input image data. It will be converted to float)

            noise_type: string
                'gauss'        Gaussian-distrituion based noise
                'poission'     Poission-distribution based noise
                's&p'          Salt and Pepper noise, 0 or 1
                'speckle'      Multiplicative noise using out = image + n*image
                               where n is uniform noise with specified mean & variance
        """
        row, col, ch = image.shape
        if noise_type == "gauss":
            mean = 0.0
            var = 0.2   #0.01
            sigma = var ** 0.5
            gauss = np.array(image.shape)
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return noisy.astype('uint8')
        elif noise_type == "s&p":
            s_vs_p = 0.5
            amount = 0.004
            out = image
            # Generate Salt '1' noise
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 255
            # Generate Pepper '0' noise
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        elif noise_type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_type == "speckle":
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy
        else:
            return image

    def PIL2array(self,img):
        return np.array(img.getdata(),
                           np.uint8).reshape(img.size[1], img.size[0], 3)

    def array2PIL(self ,arr, size):
        mode = 'RGBA'
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
        if len(arr[0]) == 3:
            arr = np.c_[arr, 255 * np.ones((len(arr), 1), np.uint8)]
        return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)