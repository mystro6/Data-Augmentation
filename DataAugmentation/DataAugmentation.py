
import PIL.Image as image
import tensorflow as tf
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

        for tensor, path in zip(self.rotatedTensors, self.rotatedImages):

            rotatedFileName = os.path.basename(path)
            rotatedFileName = re.sub("\.jpg", "", rotatedFileName)
            tensor = self.horizontallyFlip(tensor)
            self.saveTensor(tensor, rotatedFileName, "_flipped", outputDirectory)

        self.images = os.listdir(outputDirectory)

        for image in self.images:
            self.tensorImages.append(self.convertToTensor(outputDirectory + image))

        for tensor, imagePath in zip(self.tensorImages, self.images):
            print(imagePath)
            brightenTensor = self.brightenImage(tensor)
            self.saveTensor(brightenTensor, imagePath, "_brighten", outputDirectory)

    def rotateImage(self, imagePath, angle, fileName, outputDirectory):

        photo = image.open(imagePath)

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

    def brightenImage(self, tensor):
        print("brighten")
        return tf.image.adjust_brightness(tensor, 0.2)
        # if delta is less then 0 then image darkens


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
        IN augmentImages method:
                For a directory full of images
                ------------------------------
                
                imageNames = os.listdir(imageSetPath)

                for path, angle in zip(imageNames,anglesToTurn):
                    self.rotateImage(path,angle)
        """
