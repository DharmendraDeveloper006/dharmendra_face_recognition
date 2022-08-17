import face_recognition
import cv2, os
import numpy as np

data_dict={}
path="dataset/"
for persons in os.listdir(path):
    pname = path + persons
    for images in os.listdir(pname):
        image_path = "{}/{}".format(pname,images)
        name=persons
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(image, boxes)
        for encoding in encodings:
            data_dict[str(name)]=data_dict.get(name,encoding)
            
            np.save("embeddings_face.npy",data_dict)
