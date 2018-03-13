import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
#from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

img = cv2.imread('img11.jpg')

m0 = cv2.imread('0.jpg',0)
m1 = cv2.imread('1.png',0)
m2 = cv2.imread('2.png',0)
m3 = cv2.imread('3.png',0)
m4 = cv2.imread('4.png',0)
m5 = cv2.imread('5.png',0)
m6 = cv2.imread('6.png',0)
m7 = cv2.imread('7.png',0)
m8 = cv2.imread('8.png',0)
m9 = cv2.imread('9.png',0)


box = cv2.imread('box.png')

cv2.rectangle(box, (10,10), (40,40), (0,0,0), 1)

template = box[9:41,9:41]

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.startWindowThread()
cv2.imshow('gray_img', gray_img)

f = open('train_digits.csv', 'r')


sud = cv2.imread('sudoku.jpg')
sud_gray = cv2.cvtColor(sud, cv2.COLOR_BGR2GRAY)

sud_th = cv2.adaptiveThreshold(sud_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)

edges = cv2.Canny(sud_gray, 50, 150, apertureSize = 3)

minLinelength = 200
maxLineGap = 10

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLinelength,maxLineGap)
distance = []
points = []
for x1,y1,x2,y2 in lines[0]:
    cv2.line(sud,(x1,y1),(x2,y2),(0,0,255),3)
    l1 = x1**2 + y1**2
    l2 = x2**2 + y2**2
    distance.append(l1)
    distance.append(l2)
    points.append([x1,y1])
    points.append([x2,y2])

i2  = distance.index(max(distance))
i1 = distance.index(min(distance))

#cv2.line(sud, (points[i1][0], points[i1][1]), (points[i2][0], points[i2][1]), (0,255,0), 3)  

sudoku = sud[points[i1][0]:points[i2][0], points[i1][1]: points[i2][1]]  



kernel = np.ones((3,3))


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


sudoku_ = edges[points[i1][0]:points[i2][0], points[i1][1]: points[i2][1]]

sudoku_bw = cv2.dilate(sudoku_, kernel, iterations = 1)
cv2.bitwise_not(sudoku_bw, sudoku_bw)






t = min((points[i2][0]-points[i1][0])/10,(points[i2][1]-points[i1][1])/10 )

cv2.rectangle(sudoku, (330-60,336-30), (330-30,336), (255,0,0),1   )

#print points[i2][0]-points[i1][0], points[i2][1]-points[i1][1], t
#print sudoku_bw
for i in xrange(points[i2][0]-points[i1][0]-27):
    for j in xrange(points[i2][1]-points[i1][1]-27):
        l = np.sum(sudoku_bw[i:i+28, j:j+28]/255)
        if l < 250:
            cv2.rectangle(sudoku,(i,j),(i+28,j+28), (0,255,0), 1)
            
            
            


cv2.imshow('sud_thres', sud)
cv2.imshow('edges', edges)

cv2.imshow('sudoku', sudoku)
cv2.imshow('sudoku_bw', sudoku_bw)






digit1 = gray_img[0:28,0:28]
#digit2 = gray_img[28:56,0:28]
#digit3 = gray_img[0:28,28:56]
#digit4 = gray_img[28:56,28:56]

d1 = np.reshape(np.array(digit1), (1,784))
d1 = 255-d1
dd1 = np.reshape(d1, (28,28))
plt.imshow(dd1, interpolation = 'nearest')
plt.show()
'''
d2 = np.reshape(np.array(digit2), (1,784))
d2 = 255-d2
dd2 = np.reshape(d2, (28,28))
plt.imshow(dd2, interpolation = 'nearest')
plt.show()

d3 = np.reshape(np.array(digit3), (1,784))
d3 = 255-d3
dd3 = np.reshape(d3, (28,28))
plt.imshow(dd3, interpolation = 'nearest')
plt.show()

d4 = np.reshape(np.array(digit4), (1,784))
d4 = 255-d4
dd4 = np.reshape(d4, (28,28))
plt.imshow(dd4, interpolation = 'nearest')
plt.show()
'''
test_data = []
labels = []
reader = csv.DictReader(f)
for row in reader:
    i = row.keys()
    arr = [0]*784
    for key in i:
        if key != "label":
            g = int(key[5:])
            arr[g] = float(row[key])
    p = row['label']
    #if int(p) == 1:
        #print arr
    test_data.append(arr)
    labels.append(int(p))
    
l = np.array(test_data)
Y = np.reshape(np.array(labels), (42000,1))
print np.shape(l)

#print l[:,0]

Y_train = np.reshape(Y[0:20000,0], (20000,))
X_train = np.reshape(l[0:20000,:], (20000,784))

Y_test = np.reshape(Y[40000:, 0], (2000,))
X_test = np.reshape(l[40000:, :], (2000,784))

print np.shape(X_test)


clf = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes = (500,100), random_state=1)

clf.fit(X_train, Y_train)




#p1 = clf.predict(d1)
#p2 = clf.predict(d2)
#p3 = clf.predict(d3)
#p4 = clf.predict(d4)

#print p1
s = 0
'''
for i in xrange(2000):
    #print X_train[i, :]
    t = clf.predict(np.reshape(X_test[i, :], (1,785)))
    y = Y_test[i]
    print t[0], y
    if t[0] == y:
        s += 1
print s
#digits = load_digits()

#print len(digits.images )
#plt.gray() 
#plt.matshow(digits.images[0]) 
#plt.show() 
'''
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)
    cv2.bitwise_not(threshold, threshold) 
    cv2.imshow('threshold', threshold)
    cv2.rectangle(gray1,(490,490),(532,532),(0,0,255),1)
    

    recog = threshold[520:548,520:548]
    
    
    
    
    median = cv2.medianBlur(recog,5)
            

    t = np.reshape(np.array(median), (1,784))
    cv2.imshow('num', median)
    
    value = clf.predict(t)
    if value == 0:
        r = m0.shape[0]
        c = m0.shape[1]
        gray1[0:r, 0:c] = m0
    elif value == 1:
        r = m1.shape[0]
        c = m1.shape[1]
        gray1[0:r, 0:c] = m1
    elif value == 2:
        r = m2.shape[0]
        c = m2.shape[1]
        gray1[0:r, 0:c] = m2
    elif value == 3:
        r = m3.shape[0]
        c = m3.shape[1]
        gray1[0:r, 0:c] = m3
    elif value == 4:
        r = m4.shape[0]
        c = m4.shape[1]
        gray1[0:r, 0:c] = m4
    elif value == 5:
        r = m5.shape[0]
        c = m5.shape[1]
        gray1[0:r, 0:c] = m5
    elif value == 6:
        r = m6.shape[0]
        c = m6.shape[1]
        gray1[0:r, 0:c] = m6
    elif value == 7:
        r = m7.shape[0]
        c = m7.shape[1]
        gray1[0:r, 0:c] = m7
    elif value == 8:
        r = m8.shape[0]
        c = m8.shape[1]
        gray1[0:r, 0:c] = m8
    elif value == 9:
        r = m9.shape[0]
        c = m9.shape[1]
        gray1[0:r, 0:c] = m9
        
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray1,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('frame', gray1)
    
    #cv2.imshow('sudoku', sudoku)
    
    #cv2.imshow('img',gray)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


# When everything done, release the capture
cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)

