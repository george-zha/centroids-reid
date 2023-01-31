import cv2
import os

for filename in os.listdir('.'):
    capture = cv2.VideoCapture(os.getcwd() + "/" + filename)
    foldername = '/home/george/datasets/people-tracking-videos/images/' + filename.split(".")[0]
    os.mkdir(foldername)

    success, frame = capture.read()
    count = 0

    while success:
        if not count % 24:
            cv2.imwrite(foldername + "/" + str(count // 24) + ".jpg", frame)
        count += 1
        success, frame = capture.read()

    capture.release()
    print("extracted and deleted " + foldername)