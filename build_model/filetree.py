import os

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
                        # img = Image.open(curpath)
                        # newnode.val = np.array(img)/255
                        node.lastdir = True
                        node.children.append(newnode)
    router(root)
    return root