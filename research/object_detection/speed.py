# USAGE
# RUN COMMAND
# python speed.py --prototxt text1_graph.pbtxt --inference_model inference_graph/frozen_inference_graph.pb --confidence 0.5

# import the necessary packages
import cv2
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
from scipy.spatial import distance

# construct the argument parse and parse the arguments
# ye sab command line se input lene ke liye hai
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Tensorflow 'deploy' prototxt file")
ap.add_argument("-m", "--inference_model", required=True,
                help="path to trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow(args["inference_model"], args["prototxt"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = cv2.VideoCapture("Fast moving cars.mp4")
time.sleep(2.0)

# loop over the frames from the video stream

start_time = time.time()
previous_centroid = (0, 0)
dist = 0.0
while True:
    # read the next frame from the video stream and resize it
    # ret m store hota hai whether the frame is captured or not, and frame m pixel values of frame
    ret, frame = vs.read()

    if ret == True:  # agar frame capture hua hai toh hi hoga
        frame = imutils.resize(frame, width=1200)   # frame ko resize kr rhe h
        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the frame, pass it through the network,
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                     (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []

        # loop over the detections
        counter = 0
        current_speed = 0
        for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                # bounding box ke coordinates rects m daal rhe
                rects.append(box.astype("int"))

                # draw a bounding box surrounding the object so we can
                # visualize it
                (startX, startY, endX, endY) = box.astype("int")
                new_centroid = ((startX + endX)/2, (startY + endY)/2)
                dist = distance.euclidean(new_centroid, previous_centroid)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                end_time = time.time()
                if not((end_time - start_time) == 0):
                    new_speed = dist/(end_time - start_time)
                    new_speed /= 2  # random approximation taken to get reasonable values
                    new_speed = round(new_speed, 2)

                    if not(new_speed > 1000):
                        speed = current_speed*counter
                        current_speed = (speed + new_speed)/(counter+1)
                        counter += 1

                    current_speed = round(current_speed, 2)
                    text = "{} kmph".format(current_speed)
                    org = (startX, startY)
                    cv2.putText(frame, text, org,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            previous_centroid = new_centroid
            start_time = end_time

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
