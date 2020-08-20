import cv2

#Load Cascade Classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

#Start cam
cap = cv2.VideoCapture(1)

while True:
    #read image from webcam
    res, color_img = cap.read()
    #Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    #Detect faces
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    #Display rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        crop_gray = gray_img[y:y+h, x:x+w]
        crop_color = color_img[y:y+h, x:x+w]

        #Detect eyes
        eyes = eye_cascade.detectMultiScale(crop_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(crop_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    #display image
    cv2.imshow('image', color_img)

    #press q to exit
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    #Release VideoCapture object
cap.release()
cv2.destroyAllWindows()


