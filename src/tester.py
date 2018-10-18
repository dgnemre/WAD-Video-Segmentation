
import numpy as np
import matplotlib.pyplot as plt


X_train = np.load("x_train.npy")
Y_train = np.load("Y_train.npy")
X_test = np.load("x_test.npy")

preds_val_t = np.load("preds_val_t.npy")
preds_test_t = np.load("preds_test_t.npy")
preds_test = np.load("preds_test.npy")
sizes_test = np.load("sizes_test.npy")

def display_predictions(ix, preds_test_t):
    # imgs_ids = [c for c in category_to_product.keys() if category_id in str(c)]
    # ix = random.randint(0, len(train_ids))
    #ix = random.randint(nb, len(preds_train_t) - 1)
    print('ix value : ', ix)
    imgs_ids = []
    imgs_labels = []
    imgs_ids.append(X_test[ix])
    imgs_labels.append("Image")

    ### For printing purposes, we should reshape it without the last channel

    ## train preds
    tpreds = np.reshape(preds_test_t[ix], (preds_test_t[ix].shape[0], preds_test_t[ix].shape[1]))
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
    #plt.show()
    plt.savefig("test_"+str(ix))

for i in range(0,len(preds_test_t)):
    display_predictions(i, preds_test_t)
