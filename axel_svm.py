# %%
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import GridSearchCV
import joblib 
import os
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
    class_indexes = np.where(labels.label == 1)[0]
    get_classes = np.take(classes_get_np, class_indexes)
    #print(get_classes)
    for class_index in class_indexes:
        data_labels.append(class_index)
        data_np.append(data)

data_np = np.asarray(data_np)
train_labels = np.asarray(data_labels)


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
    
train_spectrograms = np.reshape(data_np, (len(data_np), -1))
val_spectrograms = np.reshape(val_np, (len(val_np), -1))


print('Spectrogram shape:', train_spectrograms.shape)
print('Labels shape:', train_labels.shape)

#print('Spectrogram dtype:', spectrograms.dtype)

#print(labels[3])

print('Training samples:', train_spectrograms.shape[0])
print('Validation samples:', val_spectrograms.shape[0])
#print('Test samples:', test_spectrograms.shape[0])

# Print the distribution of labels across the two datasets
unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
unique_val_labels, val_counts = np.unique(val_labels, return_counts=True)

print('Training distribution:')
for label, count in zip(unique_train_labels, train_counts):
    print(f' - {label}: {count}')

print('Validation distribution:')
for label, count in zip(unique_val_labels, val_counts):
    print(f' - {label}: {count}')


print('Spectrogram Shape:', train_spectrograms.shape)


print("Shape data_set_train:", train_spectrograms.shape)
print("Shape data_set_val:", val_spectrograms.shape)

# %%
dmr = True
components = 2
# Implement LDA to reduce dimensionality
if dmr:
    #lda = LDA(n_components=components)
    #lda.fit(train_spectrograms, train_labels)

    lda = joblib.load("LDA/lda_model_"+spectrogram_type+".pkl")
    #joblib.dump(fitted_LDA, 'LDA/lda_model.pkl')
    
    train_data = lda.transform(train_spectrograms)
    val_data = lda.transform(val_spectrograms)

    print("Shape data_set_train:", train_data.shape)
    print("Shape data_set_val:", val_data.shape)

    #Reshape to 2d array of 3rd and 4th component
    #train_data = train_data[:, 2:4]
    #val_data = val_data[:, 2:4]

    print("Shape data_set_train:", train_data.shape)
    print("Shape data_set_val:", val_data.shape)


# %%
# Plot the data
"""
fig, axs = plt.subplots(components, 1, figsize=(6, 2*components), sharex=True)
for i in range(components):
    axs[i].scatter(train_classes, train_data[:, i])
    axs[i].set_ylabel(f"Component {i+1}")
    axs[i].set_xticks(range(len(classes)))
    axs[i].set_xticklabels(classes, rotation=90)

plt.tight_layout()
plt.show()

# ax.set_xlim([np.min(fitted train_data[:, 0]), np.max(fitted train_data[:, 0])])
# ax.set_ylim([np.min(fitted train_data[:, 1]), np.max(fitted train_data[:, 1])])

plt.show()
"""

# %%
# Train data
"""
model = make_pipeline(StandardScaler(), SVC(gamma='scale'))
model.fit(train_data, train_classes)
print("Model trained")
"""

# %%
### Parameter Sweep 1 ###

kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']

#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")

for i in range(4):
    if os.path.isfile('SVM/model_'+spectrogram_type+'_'+kernels[i]+'.pkl'):
        svclassifier = joblib.load('SVM/model_'+spectrogram_type+'_'+kernels[i]+'.pkl')
    else:
        svclassifier = getClassifier(i) 
        svclassifier.fit(train_data, train_labels)# Make prediction
        joblib.dump(svclassifier, 'SVM/model_'+spectrogram_type+'_'+kernels[i]+'.pkl')
    x_pred = svclassifier.predict(train_data)# Evaluate our model
    y_pred = svclassifier.predict(val_data)# Evaluate our model
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(val_labels,y_pred))
    # Plot confusion matrix of training data using  metrics.ConfusionMatrixDisplay, with labels

    cm = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(train_labels, x_pred), display_labels=classes_get)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.subplots_adjust(left=0.2)
    cm.plot(ax=ax, xticks_rotation='vertical')
    cm.ax_.set_title('Confusion matrix of the classifier : \n '+kernels[i]+' kernel on training data')
    plt.savefig('SVM/confusion_matrix_'+spectrogram_type+'_'+kernels[i]+'_train.png')
    # Put labels on the confusion matrix
    #plt.show()

    # Plot confusion matrix of validation data using  metrics.ConfusionMatrixDisplay, with labels
    cm = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(val_labels, y_pred), display_labels=classes_get)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.subplots_adjust(left=0.2)
    cm.plot(ax=ax, xticks_rotation='vertical')
    cm.ax_.set_title('Confusion matrix of the classifier : \n '+kernels[i]+' kernel on validation data')
    plt.savefig('SVM/confusion_matrix_'+spectrogram_type+'_'+kernels[i]+'_val.png')
    #plt.show()


# %%
### Parameter Sweep 2 ###
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
if os.path.isfile('SVM/grid_model_'+spectrogram_type+'.pkl'):
    grid = joblib.load('SVM/grid_model_'+spectrogram_type+'.pkl')
else:
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    grid.fit(train_data, train_labels)
    # Save grid model
    joblib.dump(grid, 'SVM/grid_model_'+spectrogram_type+'.pkl')
print(grid.best_estimator_)
# Save best estimator to txt file
with open('SVM/best_estimator_'+spectrogram_type+'.txt', 'w') as f:
    f.write(str(grid.best_estimator_))
    f.close()
# Make prediction
grid_predictions = grid.predict(val_data)
print(confusion_matrix(val_labels,grid_predictions))

cm = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(val_labels, grid_predictions), display_labels=classes_get)
fig, ax = plt.subplots(figsize=(10,10))
plt.subplots_adjust(left=0.2)
cm.plot(ax=ax, xticks_rotation='vertical')
cm.ax_.set_title('Confusion matrix of the classifier with best estimator : \n '+str(grid.best_estimator_)+ ' on validation data')
plt.savefig('SVM/confusion_matrix_'+spectrogram_type+'_best_estimator.png')
#plt.show()
print(classification_report(val_labels ,grid_predictions))
# Sace classification report to txt file
with open('SVM/classification_report_'+spectrogram_type+'_best_estimator.txt', 'w') as f:
    f.write(classification_report(val_labels ,grid_predictions))
    f.close()


# %%
# Training accuracy
"""
train_predictions = model.predict(train_data)
train_accuracy = accuracy_score(train_classes, train_predictions)
print("Training Accuracy:", train_accuracy)
"""

# %%
# Validation accuracy
"""
start_val_time = time.time()

validation_predictions = model.predict(val_data)
validation_accuracy = accuracy_score(val_classes, validation_predictions)

print("Validation Accuracy:", validation_accuracy)
print("Validation Computation Time:", time.time()-start_val_time)
"""

# %%




