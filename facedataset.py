import cv2 #Library of Python bindings designed to solve computer vision problems
import os #module include many functions to interact with file system and operating system

cam = cv2.VideoCapture(0) #captures video, accepts device index or the name of a video file
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

#make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #class to detect objects in a video stream
#machine learning based approach where cascade function is trainedm from a lot of +ve &-ve images. It is then used to detect objects in other images 
# For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

#start detect your face and take 30 pictures
while(True): #initiates an infinite loop 

    ret, img = cam.read() #ret is a boolean regarding whether or not there was a return at all and img is each img that is returned, if there is no img, you wont get an error, you will get None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #OpenCV reads colors as BGR and converted to gray
    faces = face_detector.detectMultiScale(gray, 1.3, 5) # we use to find faces in the video captured

    for (x,y,w,h) in faces: 

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #cv2.rectangle(image, start_point, end_point, color, thickness), it returns an image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release() #releases the webcam, then closes all of the imshow() windows
cv2.destroyAllWindows() #releases the webcam, then closes all of the imshow() windows


