import numpy as np
import pandas as pd
import os
import random

# First do a simple sampling. We sample two images from different folders randomly.If these two images are from the same folder, we label it as 1(True), else 0 (False)
# Build a n-ary tree to store the path and nparray of pictures so it will be easier for to manage.

class nTreeNode:
    def __init__(self, fname):
        self.path =fname
        self.lastdir = False
        self.children = []

def generate_ntree(rootdir):
    """
    :type rootdir: String(path of current file/dir)
    :rtype: nTreeNode
    """
    root = nTreeNode(rootdir)
    def router(node):
        if os.path.isdir(node.path):
            for f in os.listdir(node.path):
                curpath = node.path + '/' + f
                if os.path.isdir(curpath):
                     newnode = nTreeNode(curpath)
                     node.children.append(newnode)
                     router(newnode)
                else:
                    if curpath.endswith('.pgm'):
                        newnode = nTreeNode(curpath)
                        node.lastdir = True
                        node.children.append(newnode)
    router(root)
    return root


def sampling(root, num, same_label):
    """
    :type root: nTrereNode
    :type num: int(Time of sampling)
    :type same_label: bool(sample in the same lastdir or not)
    :rtype: pandas.DataFrame
    """
    paths = []
    def traveller(start):
        if start.lastdir == False:
            for p in start.children:
                traveller(p)
        else:
            paths.append(start)
    traveller(root)

    df = pd.DataFrame(columns = ("img1", "img2", "same_face"))
    if same_label == True:
        for _ in range(num):
            idx = random.randint(0, len(paths) - 1)
            curnode = paths[idx]
            l = len(curnode.children)
            imgidx1 = random.randint(0, l - 1)
            imgidx2 = random.randint(0, l - 1)           
            img1 = curnode.children[imgidx1].path
            img2 = curnode.children[imgidx2].path
            df = df.append({"img1": img1, "img2": img2, "same_face": 1}, ignore_index=True)
        return df
    else:
        if len(paths) < 2:
            print("Invalid number of folders")
            return
        for _ in range(num):
            img1_path = random.randint(0, len(paths) - 1)
            img2_path = random.randint(0, len(paths) - 1)
            while img1_path == img2_path:
                img2_path = random.randint(0, len(paths) - 1)
            img1_parent = paths[img1_path]
            img2_parent = paths[img2_path]
            img1 = img1_parent.children[random.randint(0, len(img1_parent.children) - 1)].path
            img2 = img2_parent.children[random.randint(0, len(img2_parent.children) - 1)].path
            df = df.append({"img1": img1, "img2": img2, "same_face": 0}, ignore_index=True)
        return df

def export_files(file_tree, total_num, shuffle = True):
    """
    Generate training and test set with 8:2 principle.
    Half of data are True and another half are False
    """
    half_train = int(total_num * 0.4)
    half_test = int(total_num * 0.1)
    train_1, train_2 = sampling(file_tree, half_train, True), sampling(file_tree, half_train, False)
    df_train = pd.concat([train_1, train_2])
    test_1, test_2 = sampling(file_tree, half_test, True), sampling(file_tree, half_test, False)
    df_test = pd.concat([test_1, test_2])
    if not os.path.exists('./middleware'):
        os.mkdir('./middleware')
    if shuffle:
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_train.to_csv('./middleware/train.csv', index=False)
    df_test.to_csv('./middleware/test.csv', index=False)
    print('Finished generating training and test set \n')
    print('Training size: ' + str(half_train * 2) + '\n')
    print('Test size: ' + str(half_test * 2) + '\n')
