from compare import *

valid_faces = os.listdir('./valid_faces')
for vf in valid_faces:
    regularize_img('./valid_faces/' + vf, './valid_faces_trans',92, 92)

test_faces = os.listdir('./test')
for t in test_faces:
    regularize_img('./test/' + t, './test_trans', 92, 92)

compare('./test_trans', './valid_faces_trans', './models/model.h5', 92, 92)
