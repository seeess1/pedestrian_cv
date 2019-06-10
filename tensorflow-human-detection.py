# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = 'packages/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    # Set your standard for human detection here (lower means less strict, higher is more strict)
    threshold = 0.7
    cap = cv2.VideoCapture('data/4-25-19_crop2.mp4')
    peds = 0
    jays = 0
    frame = 0
    start_time = time.time()
    print("Start time: " + str(start_time))

    while True:
        r, img = cap.read()
        # If you want to shrink the video
        shrinker = 1
        # Width and height
        frame_w = int(cap.get(3)*shrinker)
        frame_h = int(cap.get(4)*shrinker)
        # Resize the video
        img = cv2.resize(img, (frame_w, frame_h))

        # Rectangle 1 multipliers
        x1 = .01
        x2 = .9
        y1 = .42
        y2 = .55
        # Rectangle 1 (drawn from x1, x2, ...)
        pts1 = np.array([[int(frame_w * x1), int(frame_h * y1)], \
        [int(frame_w * x1), int(frame_h * y2)], \
        [int(frame_w * x2), int(frame_h * y2)], \
        [int(frame_w * x2), int(frame_h * y1)]], np.int32)
        # Reshape and draw
        pts1 = pts1.reshape((-1,1,2))
        cv2.polylines(img, [pts1], True, (0,255,255), 4)

        # Multipliers to find crosswalk signal
        csx1 = .688
        csx2 = .72
        csy1 = .03
        csy2 = .065
        # Crosswalk signal
        ul = [int(frame_w * csx1), int(frame_h * csy1)]
        bl = [int(frame_w * csx1), int(frame_h * csy2)]
        br = [int(frame_w * csx2), int(frame_h * csy2)]
        ur = [int(frame_w * csx2), int(frame_h * csy1)]
        lt_pts = np.array([ul, bl, br, ur])
        # Reshape and draw
        lt_pts = lt_pts.reshape((-1,1,2))
        cv2.polylines(img, [lt_pts], True, (255,0,255), 2)
        # Read crosswalk signal
        signal = img[ul[1]:bl[1], ul[0]:br[0]]
        # Color detection
        boundaries = [
            ([100, 100, 100], [255, 255, 255])
        ]
        # Loop over the boundaries
        for (lower, upper) in boundaries:
            # Create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
            # Mask on the signal and get a point count
            signal_mask = cv2.inRange(signal, lower, upper)
            point_count = np.count_nonzero(signal_mask)

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                # This gets the centroid of that rectangle
                cx = int((box[1]+box[3])/2)
                cy = int((box[0]+box[2])/2)
                cv2.circle(img,(cx,cy), 5, (0,0,255), -1)
                # Check if person's bounding box falls in jaywalk check zone 1
                if ((cx >= int(frame_w * x1)) & (cx <= int(frame_w * x2)) & (cy >= \
                int(frame_h * y1)) & (cy <= int(frame_h * y2))): # | \
                    # If the crosswalk signal says go:
                    if point_count > 25:
                        peds += 1
                    else:
                        jays += 1
        cv2.imshow("cnn_preview", img)
        # Print counts
        print("Peds: " + str(int(peds)))
        print("Jays: " + str(int(jays)))
        frame += 1
        print("Time elapsed in video: {} minutes".format(frame/120))
        process_time = time.time()
        print("Processing time thus far: {} minutes".format(str(round(\
            (process_time-start_time)/60, 3))))
        print("")

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Measure elapsed time for script
    end_time = time.time()
    print("Elapsed Time: " + str(end_time-start_time))

    results = {'peds': peds, 'jays': jays}
    with open('tensor_results.csv', 'w') as f:
        w = csv.DictWriter(f, results.keys())
        w.writeheader()
        w.writerow(results)
