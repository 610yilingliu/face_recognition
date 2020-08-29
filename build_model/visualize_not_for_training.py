import math
from PIL import Image
import matplotlib.pyplot as plt
from filetree import *

def show_pics(path):
    file_tree = generate_ntree(path)
    paths = []
    def traveller(start):
        if start.lastdir == False:
            for p in start.children:
                traveller(p)
        else:
            paths.append(start)
    traveller(file_tree)
    rownum = int(math.sqrt(len(paths)))
    if rownum == 0:
        print('empty paths')
        return
    colnum = len(paths) // rownum if len(paths) % rownum == 0 else len(paths) // rownum + 1
    plt.figure(figsize=(10, 10))
    for i in range(len(paths)):
        p = paths[i]
        first_img = p.children[0].path
        cur_pic = Image.open(first_img)
        p = plt.subplot(colnum, rownum, i + 1)
        p.axis('off')
        plt.imshow(cur_pic,cmap = plt.cm.gray)
    f = plt.gcf()
    plt.show()
    plt.draw()
    f.savefig('./report_figs/face_preview.png')


if __name__ == '__main__':
    show_pics('./train_faces')
        

        