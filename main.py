#importing the dependancies
import cv2
import numpy as np
import tensorflow as tf

#load the tflite model
interpreter  = tf.lite.Interpreter(model_path = 'model2.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

#Load the DNN model for face detection

net = cv2.dnn.readNetFromCaffe('deploy.prototxt','res10_300x300_ssd_iter_140000.caffemodel')

#capture the Video from webcam

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,420)

#Function to preprocess the face area

def preprocess_face(face_region):
    resized_face = cv2.resize(face_region,(input_shape[1],input_shape[0]))
    resized_face = np.expand_dims(resized_face, axis = 0)
    resized_face = resized_face.astype(np.float32) / 255.0
    return resized_face

#Main video loop

while True:
    ret,frame = cap.read()
    if not ret:
        break

    h,w = frame.shape[:2]

    #Face detection with DNN
    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),[104.0,177.0,123.0],False,False)
    net.setInput(blob)
    detections = net.forward()

    #extracting the face regions and Predicting the drowsiness
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.8:
            box =detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x,y,x1,y1) = box.astype('int')

            x = max(0,x)
            y = max(0,y)
            x1 = min(w,x1)
            y1 = min(h,y1)
            
            face_region = frame[y:y1,x:x1]

            input_data = preprocess_face(face_region)
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()


            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(output_data)
            prediction = np.squeeze(output_data)

            #displaying results
            if prediction[0] > prediction[1]:
                label = 'Drowsy'
                confidence = prediction[0]*100
                color = (0,0,255) #red color
            else:
                label = 'Non Drowsy'
                confidence  = prediction[1]*100
                color = (0,255,0) #gren color

            #annonate the frame with the result 
            text = f'{label} ({confidence:.2f}%)'
            cv2.putText(frame,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

            cv2.rectangle(frame,(x,y),(x1,y1),color,2)

            break
    
    #display the frame with annotations
    cv2.imshow('Drowsiness detection',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




