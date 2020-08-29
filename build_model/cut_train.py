
import os
import cv2

def cutall(path_tree, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    paths = []
    base = output_dir + '/'
    def traveller(start):
        if start.lastdir == False:
            for p in start.children:
                traveller(p)
        else:
            paths.append(start)
    traveller(path_tree)
    for p in paths:
        path = p.path.split('/')
        suffix = path[-1]
        if not os.path.exists(base + suffix):
            os.mkdir(base + suffix)
        pics = p.children
        for pic in pics:
            pic_path = pic.path.split('/')
            pic_name = pic_path[-1]
            img = cv2.imread(pic.path)
            h = img.shape[0]
            w = img.shape[1]
            size = min(h, w)
            cropped = img[h - size : h, w - size : w]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(base + suffix + '/' + pic_name, cropped)
    
    