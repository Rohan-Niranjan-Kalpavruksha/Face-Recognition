import cv2 #Library of Python bindings designed to solve computer vision problems
import numpy as np #an alias for the namespace will be created, creats a link that points to numpy
from PIL import Image #pillow package
import os  #module include many functions to interact with file system and operating system

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create() #local binary pattern ([, radius[, neighbors[, grid_x[, grid_y[, threshold]]]]])
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");#class to detect objects in a video stream
#machine learning based approach where cascade function is trainedm from a lot of +ve &-ve images. It is then used to detect objects in other images 

# function to get the images and label data
def getImagesAndLabels(path): #Load the training images from dataSet folder, put them in a List of Ids and Face Samples and return it

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #this will creates the image path gives us the path of each images in the folder
    faceSamples=[] #list
    ids = [] #list

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # loading the image and convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8') #converting the PIL image into numpy array

        id = int(os.path.split(imagePath)[-1].split(".")[1]) #getting the Id from the image
        faces = detector.detectMultiScale(img_numpy) # extract the face from the training image sample

        for (x,y,w,h) in faces:  
            faceSamples.append(img_numpy[y:y+h,x:x+w]) #If a face is there then append that in the list as well as Id of it
            ids.append(id)

    return faceSamples,ids

print ("\n  Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path) #call the function and provide the data ,and we will also give the path to save our file that will be generated after training.
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n  {0} faces trained. Exiting Program".format(len(np.unique(ids))))
