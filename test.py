from keras import backend as K
import matplotlib.pyplot as plt

def get_activations(model, layer, X_batch):

    get_activations = K.function([model.layers[0].input, K.learning_phase()],
    [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

def compare_images(left_image, right_image):
    #print(image.shape)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(left_image)
    ax1.set_title('Shape '+ str(left_image.shape),
                  fontsize=50)
    ax2.imshow(right_image)
    ax2.set_title('Shape '+ str(right_image.shape)
                  , fontsize=50)
    plt.show()
