#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import numpy as np
from sklearn.decomposition import KernelPCA
import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
# from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean

import seaborn as sns

import itertools
from itertools import product

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

print("done")


# In[2]:


seq_data = np.load("E:/RA/IJCAI/Dataset/Original/seq_data_7000.npy")
attribute_data = np.load("E:/RA/IJCAI/Dataset/Original/seq_data_variant_names_7000.npy")


seq_data = seq_data[0:10]
attribute_data = attribute_data[0:10]

# seq_data = np.load("E:/RA/Pitari/Dataset/Protein_Subcellular_Localization/Sequences_Protein_Subcellular_Localization_5959.npy")
# attribute_data = np.load("E:/RA/Pitari/Dataset/Protein_Subcellular_Localization/Attributes_Protein_Subcellular_Localization_5959.npy")


attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[","")
    aa_1 = aa.replace("]","")
    aa_2 = aa_1.replace("\'","")
    attr_new.append(aa_2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)
    
print("Attribute data preprocessing Done")


# In[ ]:





# In[ ]:





# In[3]:


def generate_kmers(sequence, k):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers


# # List of 20 amino acids
# aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


total_vals = 21
# sequences = ['GSVSSAANMMAASERFGTREWAASERFGTREWGQVLKNPREQ','AYKVDAASERFGTREWTVLNACCKAASERFGTREWTTYSGTD',
#              'CCKAASERFGTREWTTYSGTDDQTNYKWAASERFGTREWQAF',
#              'DQTNYKWAASERFGTREWQAFAASERFGTREWGQVLKNPREQ','AASERFGTREWGQVLKNPREQGSVSSAANMMAASERFGTREW']
# target_label = [0,0,1,1,2]

sequences = seq_data
target_label = int_hosts[:]

k = 4  # Use k=3 for trimer k-mers
embedding_dim = total_vals**k  # Embedding dimension is 8000 for trimer k-mers

# Generate embeddings for each sequence
final_sparse_embedding = []
for i in range(len(sequences)):
#     embeddings = generate_embeddings(sequence, k, embedding_dim)
    sequence = sequences[i]
    print("i: ",i,"/",len(sequences))
    kmers = generate_kmers(sequence, k)
    
    ###########################################################################
    encoded_kmers = []
    
    for kmer in kmers:
        encoded_kmer = np.zeros(total_vals**len(kmer))
        for i, aa in enumerate(kmer):
            pos = i * total_vals**(len(kmer) - i - 1)
            if aa == 'A':
                encoded_kmer[pos:pos+total_vals] = np.array([1] + [0]*20)
            elif aa == 'C':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 1] + [0]*19)
            elif aa == 'D':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 1] + [0]*18)
            elif aa == 'E':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 1] + [0]*17)
            elif aa == 'F':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 1] + [0]*16)
            elif aa == 'G':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 1] + [0]*15)
            elif aa == 'H':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 1] + [0]*14)
            elif aa == 'I':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 1] + [0]*13)
            elif aa == 'K':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*12)
            elif aa == 'L':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*11)
            elif aa == 'M':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*10)
            elif aa == 'N':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*9)
            elif aa == 'P':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*8)
            elif aa == 'Q':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*7)
            elif aa == 'R':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*6)
            elif aa == 'S':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*5)
            elif aa == 'T':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*4)
            elif aa == 'V':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*3)
            elif aa == 'W':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*2)
            elif aa == 'Y':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*1)
        encoded_kmers.append(encoded_kmer)
    ###########################################################################
    final_sparse_embedding.append(np.array(encoded_kmers).flatten())
    ########################################################


# In[6]:


21**3


# In[5]:


################ Train-Test Splits (start) ##########################
total_splits = 1

sss = ShuffleSplit(n_splits=total_splits, test_size=0.3)

x_y = np.array(final_sparse_embedding)
a_y = np.array(int_hosts)

# for t in range(total_splits):
sss.get_n_splits(x_y, a_y)
train_index, test_index = next(sss.split(x_y, a_y)) 

x_x_train, x_x_test = x_y[train_index], x_y[test_index]
a_y_train, a_y_test = a_y[train_index], a_y[test_index]
################ Train-Test Splits (ends) ##########################

model = Lasso(alpha=0.001, max_iter=10000)
# qq = np.array(encoded_kmers).T
model.fit(x_x_train, a_y_train)

important_feature_indices = np.where(model.coef_ != 0)[0]


X_train = np.array(x_x_train)[:, important_feature_indices]
X_test = np.array(x_x_test)[:, important_feature_indices]
y_train = a_y_train
y_test = a_y_test
########################################################
# important_features
print("Lasso Done!!!")


# In[ ]:


# In[4]
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    
    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)

def svm_fun_kernel(X_train,y_train,X_test,y_test,kernel_mat):
    import time
    
    start = timeit.default_timer()
    

#     clf = svm.SVC()
    clf = svm.SVC(kernel=kernel_mat)
    
    #Train the model using the training sets
    clf.fit(kernel_mat, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("SVM Kernel Time : ", time_final)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix SVM : \n", confuse)
#     print("SVM Kernel Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)
    
# In[5]
##########################  SVM Classifier  ################################
def svm_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("SVM Time : ", time_final)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix SVM : \n", confuse)
#     print("SVM Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)
    


# In[5]
##########################  NB Classifier  ################################
def gaus_nb_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("NB Time : ", time_final)


    NB_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Gaussian NB Accuracy:",NB_acc)

    NB_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Precision:",NB_prec)
    
    NB_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Recall:",NB_recall)
    
    NB_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB F1 weighted:",NB_f1_weighted)
    
    NB_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Gaussian NB F1 macro:",NB_f1_macro)
    
    NB_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Gaussian NB F1 micro:",NB_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix NB : \n", confuse)
#     print("NB Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    check = [NB_acc,NB_prec,NB_recall,NB_f1_weighted,NB_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  MLP Classifier  ################################
def mlp_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    # Feature scaling
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test_2 = scaler.transform(X_test)


    # Finally for the MLP- Multilayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
    mlp.fit(X_train, y_train)


    y_pred = mlp.predict(X_test_2)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("MLP Time : ", time_final)
    
    MLP_acc = metrics.accuracy_score(y_test, y_pred)
#     print("MLP Accuracy:",MLP_acc)
    
    MLP_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("MLP Precision:",MLP_prec)
    
    MLP_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("MLP Recall:",MLP_recall)
    
    MLP_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("MLP F1:",MLP_f1_weighted)
    
    MLP_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("MLP F1:",MLP_f1_macro)
    
    MLP_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("MLP F1:",MLP_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix MLP : \n", confuse)
#     print("MLP Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [MLP_acc,MLP_prec,MLP_recall,MLP_f1_weighted,MLP_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  knn Classifier  ################################
def knn_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("knn Time : ", time_final)

    knn_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Knn Accuracy:",knn_acc)
    
    knn_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Knn Precision:",knn_prec)
    
    knn_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Knn Recall:",knn_recall)
    
    knn_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Knn F1 weighted:",knn_f1_weighted)
    
    knn_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Knn F1 macro:",knn_f1_macro)
    
    knn_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Knn F1 micro:",knn_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix KNN : \n", confuse)
#     print("KNN Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [knn_acc,knn_prec,knn_recall,knn_f1_weighted,knn_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  Random Forest Classifier  ################################
def rf_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 100)
    # Train the model on training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("RF Time : ", time_final)

    fr_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Random Forest Accuracy:",fr_acc)
    
    fr_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Random Forest Precision:",fr_prec)
    
    fr_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Random Forest Recall:",fr_recall)
    
    fr_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Random Forest F1 weighted:",fr_f1_weighted)
    
    fr_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Random Forest F1 macro:",fr_f1_macro)
    
    fr_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Random Forest F1 micro:",fr_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix RF : \n", confuse)
#     print("RF Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [fr_acc,fr_prec,fr_recall,fr_f1_weighted,fr_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
    ##########################  Logistic Regression Classifier  ################################
def lr_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("LR Time : ", time_final)

    LR_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    LR_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    LR_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    LR_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    LR_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    LR_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix LR : \n", confuse)
#     print("LR Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [LR_acc,LR_prec,LR_recall,LR_f1_weighted,LR_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)


def fun_decision_tree(X_train,y_train,X_test,y_test):
    from sklearn import tree
    import time
    
    start = timeit.default_timer()


    
    clf = tree.DecisionTreeClassifier()    
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("DT Time : ", time_final) 
    
    dt_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    dt_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    dt_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    dt_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    dt_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    dt_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix DT : \n", confuse)
#     print("DT Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [dt_acc,dt_prec,dt_recall,dt_f1_weighted,dt_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)


# In[ ]:


import timeit

# print("Accuracy   Precision   Recall   F1 (weighted)   F1 (Macro)   F1 (Micro)   ROC AUC")
svm_table = []
gauu_nb_table = []
mlp_table = []
knn_table = []
rf_table = []
lr_table = []
dt_table = []


from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

x_x_train, x_x_test = x_y[train_index], x_y[test_index]
a_y_train, a_y_test = a_y[train_index], a_y[test_index]


X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


#     start = timeit.default_timer()
gauu_nb_return = gaus_nb_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 

#     start = timeit.default_timer()
mlp_return = mlp_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("MLP Time : ", stop - start) 

#     start = timeit.default_timer()
knn_return = knn_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("KNN Time : ", stop - start) 

#     start = timeit.default_timer()
rf_return = rf_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("RF Time : ", stop - start) 

#     start = timeit.default_timer()
lr_return = lr_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("LR Time : ", stop - start) 

#     start = timeit.default_timer()
dt_return = fun_decision_tree(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("DT Time : ", stop - start) 

#     start = timeit.default_timer()
svm_return = svm_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("SVM Time : ", stop - start) 

gauu_nb_table.append(gauu_nb_return)
mlp_table.append(mlp_return)
knn_table.append(knn_return)
rf_table.append(rf_return)
lr_table.append(lr_return)
dt_table.append(dt_return)
svm_table.append(svm_return)

svm_table_final = DataFrame(svm_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
gauu_nb_table_final = DataFrame(gauu_nb_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
mlp_table_final = DataFrame(mlp_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
knn_table_final = DataFrame(knn_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
rf_table_final = DataFrame(rf_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
lr_table_final = DataFrame(lr_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])


# In[ ]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.mean()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.mean()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.mean()))))
final_mean_mat.append(np.transpose((list(knn_table_final.mean()))))
final_mean_mat.append(np.transpose((list(rf_table_final.mean()))))
final_mean_mat.append(np.transpose((list(lr_table_final.mean()))))
final_mean_mat.append(np.transpose((list(dt_table_final.mean()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime (Sec.)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

final_avg_mat


# In[ ]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.std()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.std()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.std()))))
final_mean_mat.append(np.transpose((list(knn_table_final.std()))))
final_mean_mat.append(np.transpose((list(rf_table_final.std()))))
final_mean_mat.append(np.transpose((list(lr_table_final.std()))))
final_mean_mat.append(np.transpose((list(dt_table_final.std()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime (Sec.)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

final_avg_mat


# In[ ]:





# In[7]:


# def generate_gapped_kmers(sequence, k=3, gap=1):
#     kmers = []
#     for i in range(len(sequence) - k + 1):
#         kmer = sequence[i:i+k]
#         for j in range(len(kmer) - gap):
#             kmers.append(kmer[:j] + '-'*gap + kmer[j+gap:])
#     return kmers

# generate_gapped_kmers("MKTITLEVE")


# In[29]:


import itertools
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state



def kmers_string_kernel(X, Y=None, k=5, d=2, p=1):
#     kmers = [''.join(x) for x in itertools.product('ATCG', repeat=k)]
    kmers = [''.join(x) for x in itertools.product('ACDEFGHIKLMNPQRSTVWY', repeat=k)]
    n1 = X.shape[0]
    if Y is None:
        n2 = n1
        Y = X
    else:
        n2 = Y.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kmers_kernel(X[i], Y[j], kmers=kmers, k=k, d=d, p=p)
    return K


def kmers_kernel(x, y, kmers, k=5, d=2, p=1):
    x_kmer_counts = np.array([count_kmer(x, kmer, d=d) for kmer in kmers])
    y_kmer_counts = np.array([count_kmer(y, kmer, d=d) for kmer in kmers])
    kernel_value = (1 + np.dot(x_kmer_counts, y_kmer_counts)) ** p
    return kernel_value

def count_kmer(seq, kmer, d=0):
    count = 0
    for i in range(len(seq) - k + 1):
        if hamming_distance(seq[i:i+k], kmer) <= d:
            count += 1
    return count

def hamming_distance(s1, s2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def nystrom_approximation(X, n_landmarks, k=5, d=2, p=1, random_state=None):
    rng = check_random_state(random_state)
    landmarks = rng.choice(X, size=n_landmarks, replace=False)
#     print("landmarks: ",landmarks)
    K_lm = kmers_string_kernel(landmarks, k=k, d=d, p=p)
    D = np.diag(1 / np.sqrt(K_lm.sum(axis=0)))
    K_approx = K_lm.dot(D).dot(D).dot(K_lm)
    return K_approx

# Example usage
X = np.array(["ATCGAAA", "AGTCCCCC", "ACGGGAGA"])
n_landmarks = 3
K_approx = nystrom_approximation(X, n_landmarks, k=5, d=2, p=1, random_state=0)
print(K_approx)


# In[ ]:





# # New

# In[1]:


import numpy as np

import numpy as np
from sklearn.decomposition import KernelPCA
import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
# from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean

import seaborn as sns

import itertools
from itertools import product

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

print("Packages Loaded!!!")


# In[2]:


# In[2]:


#seq_data = np.loadtxt("/alina-data1/Zara/TCell_Receptor/Dataset/Sequences.npy", dtype=str)
#attribute_data = np.loadtxt("/alina-data1/Zara/TCell_Receptor/Dataset/Labels.npy", dtype=str)


#seq_data = seq_data[0:1000]
#attribute_data = attribute_data[0:1000]


seq_data = np.load("E:/RA/T_Cell_Receptor/Sequences_reduced_122340.npy")
attribute_data = np.load("E:/RA/T_Cell_Receptor/Labels_reduced_122340.npy")

#########################################
train_data, test_data, train_labels, test_labels = train_test_split(seq_data, attribute_data, test_size=0.98, stratify=attribute_data)
seq_data = train_data[:]
attribute_data = train_labels[:]
############################################

# seq_data = np.load("E:/RA/Pitari/Dataset/Protein_Subcellular_Localization/Sequences_Protein_Subcellular_Localization_5959.npy")
# attribute_data = np.load("E:/RA/Pitari/Dataset/Protein_Subcellular_Localization/Attributes_Protein_Subcellular_Localization_5959.npy")


attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[","")
    aa_1 = aa.replace("]","")
    aa_2 = aa_1.replace("\'","")
    attr_new.append(aa_2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)
    
print("Attribute data preprocessing Done")


# In[15]:


def generate_kmers(sequence, k):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers

total_vals = 21
# sequences = ['GSVSSAANMMAASERFGTREWAASERFGTREWGQVLKNPREQ','AYKVDAASERFGTREWTVLNACCKAASERFGTREWTTYSGTD',
#              'CCKAASERFGTREWTTYSGTDDQTNYKWAASERFGTREWQAF',
#              'DQTNYKWAASERFGTREWQAFAASERFGTREWGQVLKNPREQ','AASERFGTREWGQVLKNPREQGSVSSAANMMAASERFGTREW']
# target_label = [0,0,1,1,2]

sequences = seq_data
target_label = int_hosts[:]

k = 3  # Use k=3 for trimer k-mers
embedding_dim = total_vals**k  # Embedding dimension is 8000 for trimer k-mers

# Generate embeddings for each sequence
final_sparse_embedding = []
for ij in range(len(sequences)):
#     embeddings = generate_embeddings(sequence, k, embedding_dim)
    sequence = sequences[ij]
    if ij%1000==0:
        print("ij: ",ij,"/",len(sequences))
    kmers = generate_kmers(sequence, k)
    
    ###########################################################################
    encoded_kmers = []
    
    for kmer in kmers:
        encoded_kmer = np.zeros(total_vals**len(kmer))
        for i, aa in enumerate(kmer):
            pos = i * total_vals**(len(kmer) - i - 1)
            if aa == 'A':
                encoded_kmer[pos:pos+total_vals] = np.array([1] + [0]*20)
            elif aa == 'C':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 1] + [0]*19)
            elif aa == 'D':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 1] + [0]*18)
            elif aa == 'E':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 1] + [0]*17)
            elif aa == 'F':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 1] + [0]*16)
            elif aa == 'G':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 1] + [0]*15)
            elif aa == 'H':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 1] + [0]*14)
            elif aa == 'I':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 1] + [0]*13)
            elif aa == 'K':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*12)
            elif aa == 'L':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*11)
            elif aa == 'M':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*10)
            elif aa == 'N':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*9)
            elif aa == 'P':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*8)
            elif aa == 'Q':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*7)
            elif aa == 'R':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*6)
            elif aa == 'S':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*5)
            elif aa == 'T':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*4)
            elif aa == 'V':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*3)
            elif aa == 'W':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*2)
            elif aa == 'Y':
                encoded_kmer[pos:pos+total_vals] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*1)
        encoded_kmers.append(encoded_kmer)
    ###########################################################################
    asdq = np.array(encoded_kmers).flatten()
    max_length = 166698
    if len(asdq)<max_length:
        for j in range(len(asdq),max_length):
            asdq = np.concatenate([asdq,[0]])
            #asdq.append(0)
    final_sparse_embedding.append(asdq)
    ########################################################
    
print("Done")


# In[ ]:


np.array(final_sparse_embedding).shape
print("Embedding done with shape: ",np.array(final_sparse_embedding).shape)


# In[ ]:


################ Train-Test Splits (start) ##########################
total_splits = 1

sss = ShuffleSplit(n_splits=total_splits, test_size=0.3)

x_y = np.array(final_sparse_embedding)
a_y = np.array(int_hosts)

# for t in range(total_splits):
sss.get_n_splits(x_y, a_y)
train_index, test_index = next(sss.split(x_y, a_y)) 

x_x_train, x_x_test = x_y[train_index], x_y[test_index]
a_y_train, a_y_test = a_y[train_index], a_y[test_index]
################ Train-Test Splits (ends) ##########################


# In[ ]:


print("Applying Lasso Regression now")
model = Lasso(alpha=0.001, max_iter=10000)
# qq = np.array(encoded_kmers).T
model.fit(x_x_train, a_y_train)

important_feature_indices = np.where(model.coef_ != 0)[0]


X_train = np.array(x_x_train)[:, important_feature_indices]
X_test = np.array(x_x_test)[:, important_feature_indices]
y_train = a_y_train
y_test = a_y_test
########################################################
# important_features
print("Lasso Done!!!")


# In[ ]:


X_train.shape,X_test.shape


# In[ ]:


# In[ ]:


# In[4]
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    
    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)

def svm_fun_kernel(X_train,y_train,X_test,y_test,kernel_mat):
    import time
    
    start = timeit.default_timer()
    

#     clf = svm.SVC()
    clf = svm.SVC(kernel=kernel_mat)
    
    #Train the model using the training sets
    clf.fit(kernel_mat, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("SVM Kernel Time : ", time_final)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix SVM : \n", confuse)
#     print("SVM Kernel Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)
    
# In[5]
##########################  SVM Classifier  ################################
def svm_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("SVM Time : ", time_final)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix SVM : \n", confuse)
#     print("SVM Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)
    


# In[5]
##########################  NB Classifier  ################################
def gaus_nb_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("NB Time : ", time_final)


    NB_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Gaussian NB Accuracy:",NB_acc)

    NB_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Precision:",NB_prec)
    
    NB_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Recall:",NB_recall)
    
    NB_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB F1 weighted:",NB_f1_weighted)
    
    NB_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Gaussian NB F1 macro:",NB_f1_macro)
    
    NB_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Gaussian NB F1 micro:",NB_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix NB : \n", confuse)
#     print("NB Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    check = [NB_acc,NB_prec,NB_recall,NB_f1_weighted,NB_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  MLP Classifier  ################################
def mlp_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    # Feature scaling
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test_2 = scaler.transform(X_test)


    # Finally for the MLP- Multilayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
    mlp.fit(X_train, y_train)


    y_pred = mlp.predict(X_test_2)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("MLP Time : ", time_final)
    
    MLP_acc = metrics.accuracy_score(y_test, y_pred)
#     print("MLP Accuracy:",MLP_acc)
    
    MLP_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("MLP Precision:",MLP_prec)
    
    MLP_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("MLP Recall:",MLP_recall)
    
    MLP_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("MLP F1:",MLP_f1_weighted)
    
    MLP_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("MLP F1:",MLP_f1_macro)
    
    MLP_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("MLP F1:",MLP_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix MLP : \n", confuse)
#     print("MLP Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [MLP_acc,MLP_prec,MLP_recall,MLP_f1_weighted,MLP_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  knn Classifier  ################################
def knn_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("knn Time : ", time_final)

    knn_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Knn Accuracy:",knn_acc)
    
    knn_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Knn Precision:",knn_prec)
    
    knn_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Knn Recall:",knn_recall)
    
    knn_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Knn F1 weighted:",knn_f1_weighted)
    
    knn_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Knn F1 macro:",knn_f1_macro)
    
    knn_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Knn F1 micro:",knn_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix KNN : \n", confuse)
#     print("KNN Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [knn_acc,knn_prec,knn_recall,knn_f1_weighted,knn_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  Random Forest Classifier  ################################
def rf_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 100)
    # Train the model on training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("RF Time : ", time_final)

    fr_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Random Forest Accuracy:",fr_acc)
    
    fr_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Random Forest Precision:",fr_prec)
    
    fr_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Random Forest Recall:",fr_recall)
    
    fr_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Random Forest F1 weighted:",fr_f1_weighted)
    
    fr_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Random Forest F1 macro:",fr_f1_macro)
    
    fr_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Random Forest F1 micro:",fr_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix RF : \n", confuse)
#     print("RF Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [fr_acc,fr_prec,fr_recall,fr_f1_weighted,fr_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
    ##########################  Logistic Regression Classifier  ################################
def lr_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("LR Time : ", time_final)

    LR_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    LR_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    LR_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    LR_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    LR_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    LR_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix LR : \n", confuse)
#     print("LR Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [LR_acc,LR_prec,LR_recall,LR_f1_weighted,LR_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)


def fun_decision_tree(X_train,y_train,X_test,y_test):
    from sklearn import tree
    import time
    
    start = timeit.default_timer()


    
    clf = tree.DecisionTreeClassifier()    
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("DT Time : ", time_final) 
    
    dt_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    dt_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    dt_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    dt_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    dt_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    dt_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix DT : \n", confuse)
#     print("DT Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [dt_acc,dt_prec,dt_recall,dt_f1_weighted,dt_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)


# In[ ]:


import timeit

# print("Accuracy   Precision   Recall   F1 (weighted)   F1 (Macro)   F1 (Micro)   ROC AUC")
svm_table = []
gauu_nb_table = []
mlp_table = []
knn_table = []
rf_table = []
lr_table = []
dt_table = []


#     start = timeit.default_timer()
gauu_nb_return = gaus_nb_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 

#     start = timeit.default_timer()
mlp_return = mlp_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("MLP Time : ", stop - start) 

#     start = timeit.default_timer()
knn_return = knn_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("KNN Time : ", stop - start) 

#     start = timeit.default_timer()
rf_return = rf_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("RF Time : ", stop - start) 

#     start = timeit.default_timer()
lr_return = lr_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("LR Time : ", stop - start) 

#     start = timeit.default_timer()
dt_return = fun_decision_tree(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("DT Time : ", stop - start) 

#     start = timeit.default_timer()
svm_return = svm_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("SVM Time : ", stop - start) 

gauu_nb_table.append(gauu_nb_return)
mlp_table.append(mlp_return)
knn_table.append(knn_return)
rf_table.append(rf_return)
lr_table.append(lr_return)
dt_table.append(dt_return)
svm_table.append(svm_return)

svm_table_final = DataFrame(svm_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
gauu_nb_table_final = DataFrame(gauu_nb_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
mlp_table_final = DataFrame(mlp_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
knn_table_final = DataFrame(knn_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
rf_table_final = DataFrame(rf_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
lr_table_final = DataFrame(lr_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])


# In[ ]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.mean()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.mean()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.mean()))))
final_mean_mat.append(np.transpose((list(knn_table_final.mean()))))
final_mean_mat.append(np.transpose((list(rf_table_final.mean()))))
final_mean_mat.append(np.transpose((list(lr_table_final.mean()))))
final_mean_mat.append(np.transpose((list(dt_table_final.mean()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime (Sec.)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

print(final_avg_mat)


# In[ ]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.std()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.std()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.std()))))
final_mean_mat.append(np.transpose((list(knn_table_final.std()))))
final_mean_mat.append(np.transpose((list(rf_table_final.std()))))
final_mean_mat.append(np.transpose((list(lr_table_final.std()))))
final_mean_mat.append(np.transpose((list(dt_table_final.std()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime (Sec.)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

print(final_avg_mat)


# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


import numpy as np

# Define the number of sequences and properties
num_sequences = 10
num_properties = 5

# Define the possible values for each property
hla_types = ["HLA-A2", "HLA-B7", "HLA-DRB1*15:01", "HLA-B27", "HLA-DRB1*04:05", "HLA-B35", "HLA-DRB1*01:01"]
gene_mutations = ["BRCA1", "BRCA2", "TP53", "PIK3CA", "APC", "KRAS", "BRAF", "TP53", "CTNNB1", "AXIN1", "ARID1A", "FGFR3", "RB1"]
clinical_char = ["Tumor size", "Grade", "Stage", "Estrogen receptor status", "Progesterone receptor status", "HER2 status", "Tumor location", "Differentiation grade", "Lymph node involvement", "Tumor size", "Number of nodules", "Portal vein invasion", "AFP levels", "Tumor stage", "Tumor grade", "Tumor location"]
immuno_features = ["Tumor-infiltrating lymphocytes", "PD-1 expression", "PD-L1 expression", "CTLA-4 expression", "High expression of immune regulatory genes such as FOXP3 and IDO"]
epigen_mod = ["DNA methylation of BRCA1 and other genes", "DNA methylation of genes involved in colorectal cancer development", "DNA methylation of genes involved in liver cancer development and progression", "DNA methylation of genes involved in urothelial cancer development"]

# Generate random t-cell sequences and corresponding labels
sequences = ["".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=9)) for i in range(num_sequences)]
labels = np.random.choice(["Breast Cancer", "Colorectal Cancer", "Liver Cancer", "Urothelial Cancer"], size=num_sequences)

# Define the length of each property embedding
embedding_lengths = [len(hla_types), len(gene_mutations), len(clinical_char), len(immuno_features), len(epigen_mod)]

# Create embeddings for each sequence based on the five properties
embeddings = np.zeros((num_sequences, sum(embedding_lengths)))
start_idx = 0
for i in range(num_properties):
    if i == 0:
        property_values = hla_types
    elif i == 1:
        property_values = gene_mutations
    elif i == 2:
        property_values = clinical_char
    elif i == 3:
        property_values = immuno_features
    elif i == 4:
        property_values = epigen_mod
    property_dict = {property_values[j]: j+start_idx for j in range(len(property_values))}
    for j in range(num_sequences):
        label = labels[j]
        if i == 0:
            if label == "Breast Cancer":
                prop_value = "HLA-A2"
            elif label == "Colorectal Cancer":
                prop_value = "HLA-B7"
            elif label == "Liver Cancer":
                prop_value = "HLA-DRB1*15:01"
            elif label == "Urothelial Cancer":
                prop_value = "HLA-B27"
        elif i == 1:
            if label == "Breast Cancer":
                prop_value = "BRCA1"
            elif label == "Colorectal Cancer":
                prop_value = "APC"
            elif label == "Liver Cancer":
                prop_value = "CTNNB1"
            elif label == "Urothelial Cancer":
                prop_value = "KRAS"
        elif i == 2:
            if label == "Breast Cancer":
                prop_value = "Tumor size"
            elif label == "Colorectal Cancer":
                prop_value = "Grade"
            elif label == "Liver Cancer":
                prop_value = "Stage"
            elif label == "Urothelial Cancer":
                prop_value = "Estrogen receptor status"
        elif i == 3:
            if label == "Breast Cancer":
                prop_value = "PD-L1 expression"
            elif label == "Colorectal Cancer":
                prop_value = "High expression of immune regulatory genes such as FOXP3 and IDO"
            elif label == "Liver Cancer":
                prop_value = "CTLA-4 expression"
            elif label == "Urothelial Cancer":
                prop_value = "Tumor-infiltrating lymphocytes"
        elif i == 4:
            if label == "Breast Cancer":
                prop_value = "DNA methylation of BRCA1 and other genes"
            elif label == "Colorectal Cancer":
                prop_value = "DNA methylation of genes involved in colorectal cancer development"
            elif label == "Liver Cancer":
                prop_value = "DNA methylation of genes involved in liver cancer development and progression"
            elif label == "Urothelial Cancer":
                prop_value = "DNA methylation of genes involved in urothelial cancer development"
        prop_idx = property_dict[prop_value]
        embeddings[j, prop_idx] = 1
    start_idx += len(property_values)

    # Concatenate the sequence and property embeddings to get the final embeddings
#     final_embeddings = np.concatenate((np.array([list(seq) for seq in sequences]), embeddings), axis=1)
#     final_embeddings = np.append(embeddings)



# In[35]:


np.concatenate((np.array([1,2,3]), np.array([4,5,6])))


# In[31]:


np.array([list(seq) for seq in sequences])


# In[26]:


embeddings

