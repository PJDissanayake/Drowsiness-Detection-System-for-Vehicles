import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="driving_behavior_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the input shape
input_shape = input_details[0]['shape']

# Define the class names (adjust these based on your specific classes)
class_names = ['DangerousDriving', 'Distracted', 'Drinking', 'SafeDriving', 'SleepyDriving', 'Yawn']

target_size = (224,224)


def preprocess_image(frame, target_size):
    # Resize the frame
    resized = cv2.resize(frame, target_size)
    # Convert to RGB (OpenCV uses BGR by default)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0).astype(np.float32)
    return input_data

net = cv2.dnn.readNetFromCaffe('deploy.prototxt','res10_300x300_ssd_iter_140000.caffemodel')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    h,w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),[104.0,177.0,123.0,],False,False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x,y,x1,y1) = box.astype('int')

            x = max(0,x)
            y = max(0,y)
            x1 = min(w,x1)
            y1 = min(h,y1)

            face_region = frame[y:y1,x:x1]

            # Preprocess the frame
            input_data = preprocess_image(face_region, (input_shape[1], input_shape[2]))

            # Set the tensor to point to the input data
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get the output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Debug: print the raw output from the model
            print("Raw model output:", output_data[0])

            # Apply softmax to convert logits into probabilities (optional, depends on your model)
            softmax_output = tf.nn.softmax(output_data[0]).numpy()

            # Process the results
            class_index = np.argmax(softmax_output)
            class_name = class_names[class_index]
            confidence = softmax_output[class_index]

            # Print the predicted class and confidence (for debugging)
            print(f"Predicted class: {class_name}, Confidence: {confidence}")
            label = f'{class_name}: {confidence:.2f}'
            cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
            # Display the result on the frame
            cv2.putText(frame,label,(x,y-10) ,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Driver Drowsiness Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
