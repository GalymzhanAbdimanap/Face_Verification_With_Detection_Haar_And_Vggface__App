import cv2
from base_camera import BaseCamera
from time import localtime, strftime
import os
import time
import numpy as np
import cv2
import sys
import glob
from func import FaceControl

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
#video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture('rtsp://admin:Qq12345678@172.16.3.52:554/Streaming/Channels/101')
threshold=0.2
(width, height) = (224, 224)
idx=0



class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        fc = FaceControl()
        camera = cv2.VideoCapture(Camera.video_source)
        #camera = cv2.VideoCapture('rtsp://admin:Qq12345678@172.16.3.52:554/Streaming/Channels/101')
        
        model = fc.loadVggFaceModel()
        
        a=[]
        b={}
        ph=[]
        images=glob.glob('dataset/*.jpg')
        for i in range(len(images)):
            image2 = cv2.imread(images[i])
            image2 = cv2.resize(image2, (224, 224))
            image2 = image2.reshape(1,224,224,3)
            image2_preprocess = fc.preprocess_image(image2)
            img2_representation = model.predict(image2_preprocess)[0,:]
            b = {i:img2_representation}
            ph.append(os.path.basename(images[i]))
            a.append(b)


        while True:
            # read current frame
            ret, frame = camera.read()
            faces = faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(224, 224)
            )
            for (x, y, w, h) in faces:
                images=glob.glob('dataset/*.jpg')

                rrec = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face = frame[y:y + h, x:x + w] 
                face_resize = cv2.resize(face, (width, height))
                face_resize = cv2.cvtColor(face_resize, cv2.COLOR_BGR2RGB)
                face_resize = face_resize.reshape(1,224,224,3)
                
                #face_resize.astype('float32')
                #face_resize = face_resize - 255/2

                image1_preprocess = fc.preprocess_image(face_resize)
                img1_representation = model.predict(image1_preprocess)[0,:]           
                

                cosines=[]
                for i in range(len(a)):
                    cosine_similarity = fc.findCosineSimilarity(img1_representation, a[i][i])
                    cosines.append(cosine_similarity)
        
                minimum = cosines.index(min(cosines))
                #percent = (0.4-float(cosines[minimum]))*100/0.4
                #print(str(percent)+'%')
                print(cosines[minimum])
                print(ph[minimum])
                print('-----------------------------')
        
                if float(cosines[minimum])<0.2:
                    text=str(images[minimum])
            
                if float(cosines[minimum])>0.35:
                    text='not found'
                    idx=len(a)
                    pathname=os.path.basename()
                    write_name = 'dataset/new_face.'+str(idx)+'.jpg'
                    cv2.imwrite(write_name, face)
            
                    b={idx:img1_representation}
                    a.append(b)
                    print('Person added')
                    print('---------------------')
                
                #cv2.putText(rrec,text,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(167,34,56),1,cv2.LINE_AA)
                    # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
