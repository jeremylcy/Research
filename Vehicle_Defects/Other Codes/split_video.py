import cv2

cam = cv2.VideoCapture("video.mp4")

currentframe = 0

while(True):
    x,frame = cam.read()

    if x:
        name = './video' + '_frame_' + str(currentframe) + '.jpg'
        cv2.imwrite(name,frame)
        currentframe += 1
    else:
        break