import cv2 #Library of Python bindings designed to solve computer vision problems
import numpy as np #an alias for the namespace will be created, creats a link that points to numpy
import os #module include many functions to interact with file system and operating system

recognizer = cv2.face.LBPHFaceRecognizer_create() #local binary pattern ([, radius[, neighbors[, grid_x[, grid_y[, threshold]]]]])
recognizer.read('trainer/trainer.yml')   #load trained model
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath); #class to detect objects in a video stream
#machine learning based approach where cascade function is trainedm from a lot of +ve &-ve images. It is then used to detect objects in other images

font = cv2.FONT_HERSHEY_SIMPLEX #used to write text on the image and is a type of font 

#iniciate id counter, the number of persons you want to include
id = 2 #two persons 


names = ['','Smile','Sad']  #key in names, start from the second place, leave first empty

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0) #captures video, accepts device index or the name of a video file
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True: #initiates an infinite loop 

    ret, img =cam.read() #ret is a boolean regarding whether or not there was a return at all and img is each img that is returned, if there is no img, you wont get an error, you will get None

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #OpenCV reads colors as BGR and converted to gray

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
# we use to find faces in the video captured
    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #cv2.rectangle(image, start_point, end_point, color, thickness), it returns an image

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w]) #recognizer is predicting the user Id and confidence of the prediction respectively 

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence)) #confidence=0 means the confidence is 100% i.e perfect match
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2) # Set rectangle around face and name of the person
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) # Display the video frame with the bounded rectangle

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release() #releases the webcam, then closes all of the imshow() windows
cv2.destroyAllWindows() #releases the webcam, then closes all of the imshow() windows
