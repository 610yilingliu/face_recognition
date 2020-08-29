from build_model import *
from cut_train import *
from filetree import *
from file_processor import *
import matplotlib.pyplot as plt

tree = generate_ntree('./train_faces')
cutall(tree, './train_cutted')

files = generate_ntree('./train_cutted')
random.seed(1)
# save for present, it could also be stored in memory instead of doing I/O to speed up the process
export_files(files, 30000, False)
train = pd.read_csv('./middleware/train.csv')
test = pd.read_csv('./middleware/test.csv')
train_npylen = train.shape[0]
test_npylen = test.shape[0]
print('Size of Training set is: ' + str(train_npylen) + '\n')
print('Size of Test set is: ' + str(test_npylen) + '\n')

pic_example = load_img(train.iloc[0, 0])
h = pic_example.shape[0]
w = pic_example.shape[1]
print('Height of single image = ' + str(h) + '\n')
print('Width of single image = ' + str(w) + '\n')

train_pic1s, train_pic2s, Y_train, input_dim = np_forSia(train, h, w)
test_pic1s, test_pic2s, Y_test, input_dim = np_forSia(test, h, w)

siamese_net = Sia_net(input_dim)
parameters = {
'batch_size' : 128 ,
'epochs' : 25,
'verbose': 2
}
siamese_net.fit([train_pic1s, train_pic2s], Y_train, hyperparameters = parameters)

if not os.path.exists('./models'):
    os.mkdir('./models')
siamese_net.save_model('models/model.h5')
pre = siamese_net.predict([test_pic1s, test_pic2s])
res = compute_accuracy(pre, Y_test)
print('Accuracy :' + str(round(1 - res, 5)))

def loss_show(loss_ls):
    y_arr = loss_ls
    x_arr = [_ for _ in range(len(loss_ls))]
    plt.figure(figsize=(15, 5))
    plt.title('Loss in each epoch')
    for x, y in zip(x_arr, y_arr):
        plt.text(x, y, str(round(y, 3)))
    plt.plot(x_arr, y_arr)

def reward_show(loss_ls):
    y_arr = []
    for i in range(1, len(loss_ls)):
        y_arr.append((loss_ls[i - 1] - loss_ls[i])/loss_ls[i - 1])
    x_arr = [_ for _ in range(len(loss_ls) - 1)]
    plt.figure(figsize=(15, 5))
    plt.title('Reward within epochs')
    for x, y in zip(x_arr, y_arr):
        plt.text(x, y, str(round(y, 3)))
    plt.plot(x_arr, y_arr)

loss_show(siamese_net.losses[0]['loss'])
if not os.path.exists('./report_figs'):
    os.mkdir('./report_figs')
plt.savefig('./report_figs/epochs_losses.png')
reward_show(siamese_net.losses[0]['loss'])
plt.savefig('./report_figs/epochs_rewards.png')