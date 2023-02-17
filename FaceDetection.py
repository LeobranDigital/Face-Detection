# How to load image file and detect face in Python.
import cv2

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("C:\python work\Image Detection\data\haarcascades\haarcascade_frontalface_default.xml")

# Load the image file
img = cv2.imread("C:\python work\Image Detection\human2.png")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Faces detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
