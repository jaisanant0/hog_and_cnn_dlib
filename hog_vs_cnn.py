import dlib
import cv2
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help='path to image file')
parser.add_argument('-w', '--weights', default='mmod_human_face_detector.dat',
                help='path to weights file')
parser.add_argument('-o','--output', help='path to save output image')

args = parser.parse_args()

img_name = args.image.split('.')

# load image
image = cv2.imread(args.image)

image_height = image.shape[0]
image_width = image.shape[1]

if image is None :
      print("[-] Can't read image ")
      exit()

#detectors
hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

# face detection hog
hog_start = time.time()

hog_face = hog_face_detector(image,1)

hog_end = time.time()
print("[+] HOG Execution time : " + str(format(hog_end-hog_start,'.3f')))

for face in hog_face :
      x = face.left()
      y = face.top()
      w = face.right() - x
      h = face.bottom() - y

      # draw rectangle over face
      cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 1)

# face detector cnn
cnn_start = time.time()

cnn_face = cnn_face_detector(image,1)

cnn_end = time.time()
print("[+] CNN Execution time : " + str(format(cnn_end-cnn_start,'.3f')))

for face in cnn_face :
      x = face.rect.left()
      y = face.rect.top()
      w = face.rect.right() - x
      h = face.rect.bottom() - y

      #draw rectangle over face
      cv2.rectangle(image,(x,y),(x+w,y+h), (0,150,150),1)

# labels
cv2.putText(image, "CNN", (image_width-100,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,150,150), 2)
cv2.putText(image, "HOG", (image_width-100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)


# show image
cv2.imshow('CNN vs HOG',image)
cv2.waitKey()

# save image
if args.output != None :
      cv2.imwrite(args.output + '/' + img_name[0] + '.png',image)

cv2.destroyAllWindows()

      
