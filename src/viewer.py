import numpy as np
import matplotlib.pyplot as plt


X_train = np.load("x_train.npy")
Y_train = np.load("Y_train.npy")
X_test = np.load("x_test.npy")

preds_train_t = np.load("preds_train_t.npy")
preds_val_t = np.load("preds_val_t.npy")

def display_predictions(ix, preds_train_t):

    print('ix value : ', ix)
    imgs_ids = []
    imgs_labels = []
    imgs_ids.append(X_train[ix])
    imgs_labels.append("Image")

    ### For printing purposes, we should reshape it without the last channel
    Y_2 = np.reshape(Y_train[ix], (Y_train[ix].shape[0], Y_train[ix].shape[1]))
    imgs_ids.append(Y_2)
    imgs_labels.append("Label")

    ## train preds
    tpreds = np.reshape(preds_train_t[ix], (preds_train_t[ix].shape[0], preds_train_t[ix].shape[1]))
    imgs_ids.append(tpreds)
    imgs_labels.append("Prediction")

    print('len imgs_ids : ', len(imgs_ids))
    fig, axs = plt.subplots(1, len(imgs_ids), figsize=(10, 4))

    print('axs : ', axs)
    for i, ax_i in enumerate(axs):
        ax_i.imshow(imgs_ids[i])
        ax_i.set_title(imgs_labels[i])
        ax_i.grid('off')
        ax_i.axis('off')
    plt.show()
    #plt.savefig(str(ix))

for i in range(0,len(preds_train_t)):
    display_predictions(i, preds_train_t)
