import cv2

eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')  # Load the eye cascade classifier
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')  # Load the face cascade classifier
camera = cv2.VideoCapture(0)  # Open the default camera
camera.set(3, 1920)  # Set the width of the captured video frame
camera.set(4, 1080)  # Set the height of the captured video frame

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('eye_detection.mp4', fourcc, 20.0, (1920, 1080))

while True:
    _, frame = camera.read()  # Read the video stream frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )  # Detect faces in the grayscale frame using the face cascade classifier

    for (x, y, w, h) in faces:  # For each detected face
        roi_gray = gray[y:y + h, x:x + w]  # Region of Interest (ROI) in grayscale
        roi_color = frame[y:y + h, x:x + w]  # Region of Interest (ROI) in color

        if len(faces) == 1:  # If only one face is detected
            eyes = eyeCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.5,
                minNeighbors=10,
                minSize=(30, 30)
            )  # Detect eyes in the ROI using the eye cascade classifier

            for i, (ex, ey, ew, eh) in enumerate(eyes):  # For each detected eye
                if i < 2:  # If it is one of the first two eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),
                                  2)  # Draw a green rectangle around the eye

    out.write(frame)  # Write the frame to the video file

    cv2.imshow('Frame', frame)  # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for the 'q' key to be pressed to exit
        break

# Release the video capture object and the VideoWriter object
camera.release()
out.release()

cv2.destroyAllWindows()  # Close all the windows
