import cv2
import datetime
import imutils
import numpy as np

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def main():
    # cap = cv2.VideoCapture('video2.mp4')
    cap = cv2.VideoCapture('videoreal.avi')
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0



    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
       
        detector.setInput(blob)
        person_detections = detector.forward()

        
        red_mask = define_colors(frame,"red_mask",'red_lower','red_upper',(136, 87, 111),(180, 255, 255))
        green_mask = define_colors(frame,"green_mask",'green_lower','green_upper',(35, 43, 52),(77, 255, 255))
        blue_mask = define_colors(frame,"blue_mask",'blue_lower','blue_upper',(110, 50, 50),(130, 255, 255))

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
                detectColor(red_mask,frame,"Red color",rgb=(0,0,255),startX=startX,startY=startY,endX=endX,endY=endY)
                detectColor(green_mask,frame,"Green color",rgb=(0,255,0),startX=startX,startY=startY,endX=endX,endY=endY)
                detectColor(blue_mask,frame,"Blue color",rgb=(255,0,0),startX=startX,startY=startY,endX=endX,endY=endY)



        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def detectColor(mask,frame,name,rgb,startX,startY,endX,endY):
    contours, hierarchy = cv2.findContours(mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            if(int(startX) > int(x) and int(x) < int(endX) and int(startY) < int(y) and int(y) < int(endY)):
                frame = cv2.rectangle(frame, (x, y),  
                                        (x + w, y + h),  
                                        rgb, 2) 
                cv2.putText(frame, name, (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        rgb)  

def define_colors(frame,nameMask,nameLower, nameUpper, rgbMin,RgbMax):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    nameLower = np.array([rgbMin], np.uint8) 
    nameUpper = np.array([RgbMax], np.uint8) 
    nameMask = cv2.inRange(hsvFrame, nameLower, nameUpper) 

    kernal = np.ones((5, 5), "uint8")

    nameMask = cv2.dilate(nameMask, kernal) 
    res_red = cv2.bitwise_and(frame, frame,  
                                mask = nameMask) 
    return nameMask

main()
