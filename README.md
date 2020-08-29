**Specific report and usage could be found in [report](report/report.md), please read it for understanding the limitation of this project and the way to improve it.**

Author: Yiling Liu

Student ID: 22214014

# File Structure

## dir `build_model`

### dir `train_faces`

Original dataset for training(AT&T dataset with 40 faces)

### `visualize_not_for_training.py`
Just for visualize the faces in `train_faces`, it is a single file not necessary for data training, could be remove safely.

### `__main__.py`
Runner to run `.py` files in the current directory(except `visualize_not_for_training.py`)

### `filetree.py`
Generate an nary-tree for manage directories with `.pgm` files inside it

### `cut_train.py`
Remove hair from the training picture to met the faces cropped by dlib

### `file_processor.py`
Sample data for training and put it into a .csv file. Notice this is just for project presentation, the better way is to store the numpy array in an `.npy` file, so the computer do not need to load the `.csv` file - this is an extra step!

### `build_model.py`
Train a Siamese Network, generate `model.h5` under dir `models`

### other files and directories
Those are generated by `.py` files mentions above

## dir `detect_face`

### dir `models`
`model.h5` is a trained model copied from `build_model/models/model.h5`

### `__main__.py`
Runner

### `compare.py`
Use dlib to crop faces(If face in current picture is too small, skip it) from taken photo, then resize it to met the trained model. Use the existed model to compare the distance between faces in taken pictures.