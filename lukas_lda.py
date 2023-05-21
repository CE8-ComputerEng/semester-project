
#import dataimporter_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot as plt
import joblib
import matplotlib.colors as mcolors
# Data set split
#data_np, val_np, data_labels, val_labels = sklearn.model_selection.train_test_split(training_np, data_labels, test_size=0.2, random_state=420)

# Possible spectrogram types are "stft", "mel", "mfcc", and "pncc"
spectrogram_type = "stft"
classes_get = ['JUMP','BOAT','SEAGUL_SCREAM','BRIDGE','SCOOTER','PEE','OBJECT_SPLASH','UFO','IDLE_MOTOR','SEAGUL_SPLASH','VOICE', 'SWIM']
FULL_TRAIN_DATASET_PATH = 'data/full_data/'+spectrogram_type+'/train_data.npy'
FULL_TRAIN_LABEL_PATH = 'data/full_data/'+spectrogram_type+'/train_labels.npy'
FULL_VAL_DATASET_PATH = 'data/full_data/'+spectrogram_type+'/val_data.npy'
FULL_VAL_LABEL_PATH = 'data/full_data/'+spectrogram_type+'/val_labels.npy'

data_np_one_hot = np.load(FULL_TRAIN_DATASET_PATH)
data_label_object = np.load(FULL_TRAIN_LABEL_PATH, allow_pickle=True)
data_labels = []
data_np = []
classes_get_np = np.asarray(classes_get)
# the LDA expects a single label for each sample, so we need to convert the one-hot encoded labels to single labels
# If we meet a sample with multiple labels, we make multiple copies of the sample, one for each label
for data, labels in zip(data_np_one_hot, data_label_object):
    plt.imshow(data)
    plt.pause(0.05)
    plt.clf()
    class_indexes = np.where(labels.label == 1)[0]
    get_classes = np.take(classes_get_np, class_indexes)
    #print(get_classes)
    for class_index in class_indexes:
        data_labels.append(class_index)
        data_np.append(data)

data_np = np.asarray(data_np)
data_labels = np.asarray(data_labels)


#val_np = np.load(FULL_VAL_DATASET_PATH)
#val_labels = np.load(FULL_VAL_LABEL_PATH, allow_pickle=True)

val_data_np_one_hot = np.load(FULL_VAL_DATASET_PATH)
val_data_label_object = np.load(FULL_VAL_LABEL_PATH, allow_pickle=True)
val_data_labels = []
val_data_np = []
classes_get_np = np.asarray(classes_get)

for data, labels in zip(val_data_np_one_hot, val_data_label_object):
    class_indexes = np.where(labels.label == 1)[0]
    get_classes = np.take(classes_get_np, class_indexes)
    #print(get_classes)
    for class_index in class_indexes:
        val_data_labels.append(class_index)
        val_data_np.append(data)    

val_np = np.asarray(val_data_np)
val_labels = np.asarray(val_data_labels)
val_classes = np.unique(val_labels)
    
data_np_flatten = np.reshape(data_np, (len(data_np), -1))
val_np_flatten = np.reshape(val_np, (len(val_np), -1))



# If model is not trained yet, train it
if not os.path.exists("LDA/lda_model_"+spectrogram_type+".pkl"):
    print("Training LDA model...")
    LDA = LinearDiscriminantAnalysis(n_components=np.unique(data_labels).shape[0] - 1)
    fitted_LDA = LDA.fit_transform(data_np_flatten, data_labels)

    num_components = fitted_LDA.shape[1]
    print(f"Number of components: {num_components}")
    # Save the model:
    joblib.dump(LDA, "LDA/lda_model_"+spectrogram_type+".pkl")
else:
    # Load the model:
    LDA = joblib.load("LDA/lda_model_"+spectrogram_type+".pkl")
    #joblib.dump(fitted_LDA, 'LDA/lda_model.pkl')
    fitted_LDA = LDA.transform(data_np_flatten)
    num_components = fitted_LDA.shape[1]
    print(f"Number of components: {num_components}")

colors = plt.cm.tab20(np.linspace(0, 1, len(classes_get)))

cmap = plt.cm.get_cmap('gray')  # Gray colormap for all classes
cmap.set_over('blue')  # Set a different color for class 8 (blue in this example)


######### Plotting the LDA components #########
if num_components >= 1:
    fig, axs = plt.subplots(num_components, 1, figsize=(6, 2*num_components), sharex=True)
    label_ints = [np.where(np.unique(data_labels) == label)[0][0] for label in data_labels]

    for i in range(num_components):
        axs[i].scatter(data_labels, fitted_LDA[:, i])
        axs[i].set_ylabel(f"Component {i+1}")
        axs[i].set_xticks(range(len(classes_get)))
        axs[i].set_xticklabels(classes_get, rotation=90)
        axs[i].set_xlabel("Class")

    if not os.path.exists("LDA"):
        os.makedirs("LDA")

    # Save the figure to the LDA folder
    plt.savefig(f"LDA/LDA_{num_components}_Components_"+spectrogram_type+".png", dpi=300)
    plt.tight_layout()
    #plt.show()

######### Plotting the 1st and 2nd component of the LDA #########
if num_components >= 2:
    # Showing the LDA components on 
    fig, ax = plt.subplots()

    from matplotlib.colors import ListedColormap
    # Define a list of colors
    colors = ['red'] + ['gray'] * (len(classes_get) - 1)

    # Create a custom colormap using the list of colors
    cmap = ListedColormap(colors)

    # Plot the data, the color map should tab20, but only the first 14 colors are used, so the rest is set to gray
    scatter = ax.scatter(fitted_LDA[:, 0], fitted_LDA[:, 1], c=label_ints, cmap=cmap, alpha=1)
    mask_class_8 = (label_ints == 0)
    scatter.set_facecolor(np.where(mask_class_8, mcolors.to_rgba('red'), 'gray'))


    # Add colorbar and axis-labels
    cbar = plt.colorbar(scatter, ticks=np.arange(len(classes_get)))
    cbar.ax.set_yticklabels(classes_get)
    ax.set_xlabel('LDA Component 1')
    ax.set_ylabel('LDA Component 2')
    ax.set_title('LDA Components 1 and 2')
    ax.legend()

    import matplotlib.colors as mcolors
    # Get the unique class labels
    unique_labels = np.unique(label_ints)

    # Create a new color map with the first 14 colors from 'tab20' and the rest set to gray
    colors = plt.get_cmap('tab20')(np.arange(14))
    colors = np.append(colors, mcolors.to_rgba('gray'))


    # Save the figure to the LDA folder
    plt.savefig(f"LDA/LDA_Component_1_2_"+spectrogram_type+".png", dpi=300)
    ax.legend()
    #plt.show()

######### Plotting the 3rd and 4th component of the LDA #########
if num_components >= 4:
    # Plots the third and forth component of the LDA
    fig, ax = plt.subplots()
    scatter = ax.scatter(fitted_LDA[:, 2], fitted_LDA[:, 3], c=label_ints, cmap=cmap, alpha=1)
    mask_class_8 = (label_ints == 10)
    scatter.set_facecolor(np.where(mask_class_8, mcolors.to_rgba('red'), 'gray'))

    # Add labels and legend
    ax.set_xlabel('LDA Component 3')
    ax.set_ylabel('LDA Component 4')
    ax.set_title('LDA Components 3 and 4')
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(scatter, ticks=np.arange(len(classes_get)))
    cbar.ax.set_yticklabels(classes_get)

    # Save the figure to the LDA folder
    plt.savefig(f"LDA/LDA_Component_3_4_"+spectrogram_type+".png", dpi=300)
    # Show the plot
    #plt.show()
    
    
# Validation test
cm = sklearn.metrics.confusion_matrix(val_labels, LDA.predict(val_np_flatten))
print(cm)
disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_classes)

disp.plot()
plt.savefig(f"LDA/LDA_Confusion_Matrix_"+spectrogram_type+".png", dpi=300)