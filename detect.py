import cv2
# Load the cascade
face_without = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('myhaar.xml')
cap = cv2.VideoCapture(0)   # To capture video from webcam. 

while True:
    _, img = cap.read() # Read the frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) # Detect the faces
    facesW = face_without.detectMultiScale(gray, 1.1, 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, 'Using Mask', (55,280), font,0.5,(0,255,0))

    for (x, y, w, h) in facesW:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, 'No Mask', (55,280), font,0.5,(0,0,255))

    #cv2.putText(img, 'No Mask', (20,200), font,0.5,(255,255,255))
        
    cv2.imshow('img', img)  # Display
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    # Release the VideoCapture object
cap.release()
