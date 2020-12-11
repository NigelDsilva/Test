import numpy as np
import cv2

PERSON_SIZE = 600
incount = 0
outcount = 0


class Person:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def dist(self, x, y):
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** .5


persons = []

out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (640, 480))

clf = cv2.CascadeClassifier('cascadeH5.xml')


def draw_detections(img, rects, fgmask, thickness=2):
    global incount, outcount
    for x, y, w, h in rects:
        # print(x,y,w,h)
        if w * h > PERSON_SIZE:
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            roi_color = img[y:y + h, x:x + w]
            roi_mask = fgmask[y:y + h, x:x + w]
            roi = cv2.bitwise_and(roi_color, roi_color, roi_mask)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            rects = clf.detectMultiScale(roi)
            for (x, y, w, h) in rects:
                cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            (x_0, y_0) = (x + w / 2, y + h / 2)
            midline = img.shape[0] // 2
            left = img.shape[0] * 0.1
            right = img.shape[0] * 0.9
            new = True
            active = left < x_0 < right
            for id, person in enumerate(persons):
                if active and person.dist(x_0, y_0) <= w and person.dist(x_0, y_0) <= h:
                    new = False
                    
                    outcount = len(persons)
                    
                    person.x = x_0
                    person.y = y_0
                    break
                if not active:
                    persons.pop(id)
            if active and new:
                persons.append(Person(len(persons) + 1, x_0, y_0))
    cv2.putText(img, 'people detected :{}'.format(outcount), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (57, 255, 20), 1,
                cv2.LINE_AA)
    cv2.imshow('Vision', img)


file = 'test.mp4'
# file = '11.mp4'

cap = cv2.VideoCapture(file)

fgbg = cv2.createBackgroundSubtractorKNN(history=1000, detectShadows=True)

lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, 1.5) * 255.0, 0, 255)

def skeletionize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    i = 3
    while (i):
        i-=1
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
    return skel

while (1):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.LUT(frame, lookUpTable)
    # frame = cv2.convertScaleAbs(frame, alpha=-1.2, beta=-1)
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    gray = cv2.bitwise_and(frame,frame,mask=fgmask)

    #fgmask = cv2.Canny(gray,100,200)
    ( allContours, _) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(frame, allContours, -1, (0, 255, 0), 3)
    allContours = [cv2.boundingRect(c) for c in allContours]
    draw_detections(frame, allContours, fgmask)
    cv2.line(frame, (frame.shape[0] // 2, 0), (frame.shape[0] // 2, frame.shape[1]), (255, 0, 0), 5)
    cv2.imshow('Casscadeview', fgmask)#cv2.bitwise_and(frame,frame,mask=fgmask))
    #out.write(frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
