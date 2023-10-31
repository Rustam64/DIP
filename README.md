# Viola-Jones Face Detection

Developed by Rakhimov Rustam, Munkhtulga Irmuun, and Khaitov Sardor.

## Objectives

- Train, test, and evaluate the performance of a HAAR cascade for face detection.
- Prepare an appropriate dataset.
- Understand GUI trainer configurations and useful fixes to errors.
- Understand the Python code and how to configure it for best results.

## Contributions

- Rustam: Adjusting GUI settings to train the model.
- Rustam: Adjusting Python code to make the best of the available model.

## Training the Model Using GUI

This section explains how to use the GUI settings for training the face detection model. The GUI provides options to configure the training process.

![Screenshot (371)](https://github.com/Rustam64/DIP/assets/83468895/4d23e22d-7458-4ee3-86c1-368712c538d9)


1. **Positive Image Usage**: This option decides how many positive (p) images are used when training the model. A value of 100 means all positive images are used, and a lower value indicates a lower percentage.

2. **Negative Image Count**: This option specifies the number of negative (n) images to be used during training.

3. **Force Positive Sample Count**: This setting forces the use of a certain number of positive images, regardless of the percentile.

If you encounter an error during training, consider reducing the "Positive Image Usage" percentile. The error message "cannot get new positive samples" may indicate insufficient positive samples. Similarly, you can address the "False Alarm" error by reducing the number of negative images used.

## Testing the Model

After training the model, you can test it using the following Python code:

```python
import cv2
import time

classifier_path = r'C:\Users\rrakh\Desktop\tester\classifier\cascade.xml'
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
```

The most important line in this code is:
```python
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(100, 100))
```

- `ScaleFactor`: It decides how in-depth the model works and how much the image is resized.
- `MinNeighbors`: It determines how strong the model is.
- `MinSize`: It is the minimum object size.

The output provides information on the minimum and maximum detectable face sizes and the time taken for face detection.

## Accuracy Analysis

- Viewpoint Variation: Little to no effect
- Deformation Occlusion: Little effect
- Illumination Conditions: Little to no effect
- Cluttered or Textured Background: Little effect
- Intra-class Variation: Little to no effect

The Viola-Jones Face Detection model performs well across various conditions, with little to no impact from these factors.

![image](https://github.com/Rustam64/DIP/assets/83468895/cf852332-2f02-4754-ad48-6de0b9b475be)

![image](https://github.com/Rustam64/DIP/assets/83468895/f041d143-c9e1-4177-91d7-fdde7b17d20c)

For further reference, you can explore the demo file provided with the code.
