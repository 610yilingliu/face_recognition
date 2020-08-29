import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model


def regularize_img(img_path, dist_path, dist_w, dist_h):
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    dets = detector(image, 1)
    if not dets:
        return
    for idx, face in enumerate(dets):
        left = max(0, face.left())
        top = max(0, face.top())
        right = max(0, face.right())
        bottom = max(0, face.bottom())

        w = right - left
        h = bottom - top
        # if too small, do not analyze it
        if w < image.shape[1]//15 or h < image.shape[1]//15:
            return
        cropped = image[top:bottom, left:right]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cropped = cv2.resize(cropped, (dist_h, dist_w))
        if not os.path.exists(dist_path):
            os.mkdir(dist_path)
        fname = img_path.split('/')[-1]
        fname = fname.split('.')[0]
        fpath = dist_path + '/' + fname + '_' + str(idx) + '.pgm'
        cv2.imwrite(fpath , cropped)

def compare(face_path, valid_faces_path, model_path, h, w):
    model = load_model(model_path)
    faces = os.listdir(face_path)
    valid_faces = os.listdir(valid_faces_path)
    
    if not faces or not valid_faces:
        print('No face found in Image or no person valid! ')
        return
    for face in faces:
        img = Image.open(face_path + '/' + face)
        img_np = np.array(img)/255
        np1 = np.zeros([1, 1, w, h])
        np1[0, 0, :, :] = img_np
        np1 = tf.transpose(np1, [0, 2, 3, 1])
        found = False

        for valid_face in valid_faces:
            img2 = Image.open(valid_faces_path + '/' + valid_face)
            img2_np = np.array(img2)/255
            np2 = np.zeros([1, 1, w, h])
            np2[0, 0, :, :] = img2_np
            np2 = tf.transpose(np2, [0, 2, 3, 1])
            pre = model.predict([np1, np2])
            # print(pre)
            if pre >= 0.5:
                print(face + ' matches ' + valid_face + ', match rate: ' + str(round(pre[0][0], 3)))
                found = True
                break
        if found == False:
            print("No valid person found! for " + face)

    