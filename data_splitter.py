import numpy as np
import random
import os
class DistributionDataset:
    def __init__(self, data_np, labels_np, CLASSES):
        label_count = np.zeros(len(CLASSES))
        for data_np, labels_np in zip(data_np, labels_np):
            label_count += labels_np.label
        self.index = CLASSES
        self.values = label_count



def split_data_val_train(data_np, labels_np, CLASSES, TRAINING_RATIO = 0.8, VALIDATION_RATIO = 0.2, TRAIN_MAX_SAMPLES_PER_CLASS = None):
    TRAIN_INDICIES_PATH = "data/full_data/train_indicies.npy"
    VAL_INDICIES_PATH = "data/full_data/val_indicies.npy"
    random.seed(42)
    
    if not os.path.exists(TRAIN_INDICIES_PATH) and not os.path.exists(VAL_INDICIES_PATH):
        if TRAIN_MAX_SAMPLES_PER_CLASS is None:
            VAL_MAX_SAMPLES_PER_CLASS = None
        else:
            VAL_MAX_SAMPLES_PER_CLASS = TRAIN_MAX_SAMPLES_PER_CLASS * VALIDATION_RATIO / TRAINING_RATIO
        # First we go through the data and find the number of samples for each class, and store their indices in a dictionary
        classes = CLASSES
        label_count = np.zeros(len(classes))
        label_indices = {}
        for single_class in classes:
            label_indices[single_class] = []
        for i in range(len(labels_np)):
            label_count += labels_np[i].label
            for j, single_class in zip(range(len(classes)), classes):
                if labels_np[i].label[j] == 1:
                    label_indices[single_class].append(i)
        # sort classes by number of indices for each class
        sorted_classes = [x for _,x in sorted(zip(label_count,classes))]

        # Now we figure out how many class based sample groups we can split the data into.
        # Here, a class based sample group is a sequence of samples that all have same label and are next to each other in the data.
        # So as an example, if index 1-10 have label 1, and index 11-20 have label 2, then we have two class based sample groups.
        # We want to only split the data when we meet a new class based sample group, so that we don't split a class based sample group in half.

        # We do this by iterating through each data sample, and checking if the next sample has the same label and is 
        # part of the same class based sample group (part of the sequence index wise).

        # We also want to keep track of the indices of the class based sample groups, so that we can split the data later.

        first_index = True
        first_index_of_class = 0
        class_label_indices = {}
        previous_index = 0
        for single_class in classes: # for each class
            class_label_indices[single_class] = []
            for i in label_indices[single_class]: # for each index of the class
                if first_index: # Check if this is the first index of new class based sample group
                    first_index_of_class = i
                    previous_index = i
                    first_index = False
                elif i != previous_index + 1: # Check if they are part of same sequence
                    class_label_indices[single_class].append([first_index_of_class, previous_index]) # If not, we have reached the end of the class based sample group, and we add the indices to the dictionary
                    first_index = True
                else:
                    previous_index = i # If they are part of same sequence, we update the previous index
                    
        # Now we have the indices of the class based sample groups, and we can split the data into train and validation.

        # We first split the data into train and test sets.

        # First we create the train validation and test dictionaries:
        # These dictionaries will only contain the start and end indices of each class based sample group.
        train_indices = {} # 
        val_indices = {}
        
        # These dictionaries will contain the entire range of indices for each class based sample group.
        train_full_indices = {} 
        val_full_indices = {}
        
        train_total_list = []
        val_total_list = []
        total_samples = 0
        for single_class in classes: # for each class
            train_indices[single_class] = []
            train_full_indices[single_class] = []
            val_indices[single_class] = []
            val_full_indices[single_class] = []
            #print("------------------")
            #print("for class: ", single_class)
            # Find total number of samples for this class
            total_class_samples = 0
            for i in class_label_indices[single_class]: # for each class based sample group
                total_class_samples += len(range(i[0], i[1]+1))
            train_sample_size = int(total_class_samples * TRAINING_RATIO)
            val_sample_size = int(total_class_samples * VALIDATION_RATIO)

            # Now we pick random indices for the train and validation sets until we have reached the sample size
            train_sample_count = 0
            val_sample_count = 0
            used_indices = []
            not_used_indices = list(range(len(class_label_indices[single_class])))
            failed_indices = []
            while len(not_used_indices) > 0:
                random_index = random.choice(not_used_indices) # pick random index
                random_val_or_train = random.choice([0,1]) # 0 = validation, 1 = train
                sample_count = len(range(class_label_indices[single_class][random_index][0], class_label_indices[single_class][random_index][1]+1))
                if random_index not in used_indices:
                    if random_val_or_train == 1 and train_sample_count < train_sample_size and sample_count + train_sample_count <= train_sample_size:
                        train_indices[single_class].append(class_label_indices[single_class][random_index])
                        train_sample_count += sample_count
                        used_indices.append(random_index)
                        not_used_indices.remove(random_index)
                    elif random_val_or_train == 0 and val_sample_count < val_sample_size and sample_count + val_sample_count <= val_sample_size:
                        val_indices[single_class].append(class_label_indices[single_class][random_index])
                        val_sample_count += sample_count
                        used_indices.append(random_index)
                        not_used_indices.remove(random_index)
                    else: # if we can't add the sample to either train or validation, we try again
                        if random_index not in failed_indices: # if we haven't already tried this index
                            failed_indices.append(random_index)
                        else: # if we have already tried this index.
                            # Check if it is the only index available for this class based sample group
                            if len(not_used_indices) == 1:
                                # We then split the class based sample group in two, with the ratio of train and validation samples
                                # We then add the indices to the train and validation sets
                                start_index = class_label_indices[single_class][random_index][0]
                                end_index = class_label_indices[single_class][random_index][1]
                                val_index_split = int(sample_count * VALIDATION_RATIO)
                                val_indices[single_class].append([start_index, start_index + val_index_split])
                                train_indices[single_class].append([start_index + val_index_split + 1, end_index])
                                val_sample_count += val_index_split
                                train_sample_count += sample_count - val_index_split
                                used_indices.append(random_index)
                                not_used_indices.remove(random_index)
                                
                                    
            # NEXT TODO:
            # Try to implement a way to balance the classes, so that we don't have a lot of samples from one class and few from another.
            # Combine the indices of the class based sample groups, so that we have a list of indices for the train and validation sets.
            # We then need to make sure that we don't pick the same index twice, and that we don't pick an index that is part of the same class based sample group as a previously picked index.
            if val_sample_count == 0:
                val_sample_ratio = 0
            else:
                val_sample_ratio = val_sample_count/total_class_samples
            if train_sample_count == 0:
                train_sample_ratio = 0
            else:
                train_sample_ratio = train_sample_count/total_class_samples
            #print("Total indices: ", class_label_indices[single_class])
            print("Train indices: ", train_indices[single_class])
            print("Val indices: ", val_indices[single_class])
            #print("total class samples: ", total_class_samples)
            #print("train sample size: ", train_sample_count, " train sample ratio: ", train_sample_ratio)
            #print("val sample size: ", val_sample_count, " val sample ratio: ", val_sample_ratio)
            
            # Now we combine the indices of the class based sample groups into one list of indices for the train and validation sets.
            for i in train_indices[single_class]:
                train_full_indices[single_class] += list(range(i[0], i[1]+1))
            for i in val_indices[single_class]:
                val_full_indices[single_class] += list(range(i[0], i[1]+1))
            #print("---- BEFORE BALANCING ----")
            #print("train full indices length: ", len(train_full_indices[single_class]))
            #print("val full indices length: ", len(val_full_indices[single_class]))
            
            if TRAIN_MAX_SAMPLES_PER_CLASS != None:
                # We now randomly remove indices from the train and validation sets until we reach TRAIN_MAX_SAMPLES_PER_CLASS and VAL_MAX_SAMPLES_PER_CLASS
                if len(train_full_indices[single_class]) > TRAIN_MAX_SAMPLES_PER_CLASS:
                    while len(train_full_indices[single_class]) > TRAIN_MAX_SAMPLES_PER_CLASS:
                        random_index = random.choice(train_full_indices[single_class])
                        train_full_indices[single_class].remove(random_index)
                if len(val_full_indices[single_class]) > VAL_MAX_SAMPLES_PER_CLASS:
                    while len(val_full_indices[single_class]) > VAL_MAX_SAMPLES_PER_CLASS:
                        random_index = random.choice(val_full_indices[single_class])
                        val_full_indices[single_class].remove(random_index)
            #print("---- AFTER BALANCING ----")
            #print("train full indices length: ", len(train_full_indices[single_class]))
            #print("val full indices length: ", len(val_full_indices[single_class]))
            
            
            # 
            train_total_list += train_full_indices[single_class]
            val_total_list += val_full_indices[single_class]
            
            total_samples += total_class_samples
            
        #print("---- BEFORE REMOVING DUPLICATES ----")
        #print("total samples: ", total_samples)
        #print("total labels: ", len(labels_np))

        #print("---- AFTER REMOVING DUPLICATES ----")
        #print("train total list length: ", len(train_total_list))
        #print("val total list length: ", len(val_total_list))
        # We now remove duplicates from the train and validation sets
        train_total_list = list(set(train_total_list))
        val_total_list = list(set(val_total_list))
        
        # Check if there are any duplicates between the train and validation sets
        for i in train_total_list:
            if i in val_total_list:
                # Remove duplicates from the train set
                train_total_list.remove(i)
        for i in val_total_list:
            if i in train_total_list:
                # Remove duplicates from the validation set
                val_total_list.remove(i)
                
        
        # We now save the total train and validation indicies to np arrays, so we can reuse them later.
        
        np.save(TRAIN_INDICIES_PATH, train_total_list)
        np.save(VAL_INDICIES_PATH, val_total_list)
        
    else:
        # We load the train and validation indicies from np arrays
        train_total_list = np.load(TRAIN_INDICIES_PATH)
        val_total_list = np.load(VAL_INDICIES_PATH)
    
    #print("---- NUMPY ARRAYS AFTER SPLIT: ----")
    train_data_np = np.take(data_np, train_total_list, axis=0)
    train_labels_np = np.take(labels_np, train_total_list, axis=0)
    val_data_np = np.take(data_np, val_total_list, axis=0)
    val_labels_np = np.take(labels_np, val_total_list, axis=0)
    #print("train data shape: ", train_data_np.shape)
    #print("train labels shape: ", train_labels_np.shape)
    #print("val data shape: ", val_data_np.shape)
    #print("val labels shape: ", val_labels_np.shape)
    
    
    return train_data_np, train_labels_np, val_data_np, val_labels_np
    
    



