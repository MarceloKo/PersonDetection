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

    lower = {'red': (166, 84, 141),
            'blue': (97, 100, 117),
            'yellow': (23, 59, 119)}

    upper = {'red': (186, 255, 255),
            'blue': (117, 255, 255),
            'yellow': (54, 255, 255)}

    # define standard colors for circle around the object
    colors = {'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 217)}

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
                for key, value in upper.items():
                    # construct a mask for the color from dictionary`1, then perform
                    # a series of dilations and erosions to remove any small
                    # blobs left in the mask
                    kernel = np.ones((9, 9), np.uint8)
                    mask = cv2.inRange(hsv, lower[key], upper[key])
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                  
                    # find contours in the mask and initialize the current
                    # (x, y) center of the ball
                    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center = None
              
                    # only proceed if at least one contour was found
                    if len(cnts) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c = max(cnts, key=cv2.contourArea)
                       
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        M = cv2.moments(c)
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                        # only proceed if the radius meets a minimum size. Correct this value for your obect's size
                        if (radius > 0 and radius < 40):
                            if(int(startX) > int(x) and int(x) < int(endX) and int(startY) < int(y) and int(y) < int(endY)):
                                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                                cv2.putText(frame, key + " object", (int(x-radius), int(y - radius)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[key], 2)
                            # draw the circle and centroid on the frame,
                            # then update the list of tracked points

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


main()
