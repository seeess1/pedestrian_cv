import cv2
import time
import os
import numpy as np

# Load in data for human detection
person_cascade = cv2.CascadeClassifier('packages/haarcascade_fullbody.xml')
cap = cv2.VideoCapture("data/4-25-19_crop.mp4")
start_time = time.time()
print("Start time: " + str(start_time))
peds = 0
jays = 0
frame = 0
while True:
    r, img = cap.read()
    if r:
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

        # Find potential people
        gray_frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Haar-cascade classifier needs a grayscale image
        rects = person_cascade.detectMultiScale(gray_frame)

        # Hone in on people
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),2)
            # This gets the centroid of that rectangle
            cx = int((2*x + w)/2)
            cy = int((2*y + h)/2)
            cv2.circle(img,(cx,cy), 5, (0,0,255), -1)
            # Check if person's bounding box falls in jaywalk check zone 1
            if ((cx >= int(frame_w * x1)) & (cx <= int(frame_w * x2)) & (cy >= \
            int(frame_h * y1)) & (cy <= int(frame_h * y2))):
                # If the crosswalk signal says go:
                if point_count > 25:
                    peds += 1
                else:
                    jays += 1

        cv2.imshow("cascade_preview", img)
        # Print counts
        print("Peds: " + str(int(peds)))
        print("Jays: " + str(int(jays)))
        frame += 1
        print("Progress: {}%".format(round(frame/cap.get(7)*100, 3)))
        print("Time elapsed in video: {} minutes".format(frame/120))
        process_time = time.time()
        print("Processing time thus far: {} seconds".format(str(round(\
            (process_time-start_time), 3))))
        print("")

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break

# Measure elapsed time for script
end_time = time.time()
print("Elapsed Time: " + str(end_time-start_time))

results = {'peds': peds, 'jays': jays, 'time': (end_time-start_time)}
with open('cascade_results.csv', 'w') as f:
    w = csv.DictWriter(f, results.keys())
    w.writeheader()
    w.writerow(results)
