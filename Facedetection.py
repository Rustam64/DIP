import cv2
import time

classifier_path = r'C:\Users\rrakh\Desktop\VJ\classifier\cascade.xml'
face_cascade = cv2.CascadeClassifier(classifier_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize variables to track time and size metrics
total_time = 0
total_frames = 0
min_face_size = float('inf')
max_face_size = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Record the start time
    start_time = time.time()

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(100, 100))

    # Calculate the time taken for face detection
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Update metrics
    total_time += elapsed_time
    total_frames += 1

    # Update min and max face size metrics
    for (x, y, w, h) in faces:
        face_size = w * h
        min_face_size = min(min_face_size, face_size)
        max_face_size = max(max_face_size, face_size)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the video feed with detected faces
    cv2.imshow('Face Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate the average time for face detection
average_time = total_time / total_frames

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(f"Average time for face detection: {average_time} seconds")
print(f"Minimum detectable face size: {min_face_size}")
print(f"Maximum detectable face size: {max_face_size}")
