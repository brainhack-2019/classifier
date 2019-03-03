import numpy as np

# ------------------------------------------------------------------------------
# Define functions
# ------------------------------------------------------------------------------

def fit_model(data, labels):
    
    THRESHOLD = 1.0
    
    data_G = [0, 0, 0, 0, 0]
    data_G[0] = data[labels == 0]
    data_G[1] = data[labels == 1]
    data_G[2] = data[labels == 2]
    data_G[3] = data[labels == 3]
    data_G[4] = data[labels == 4]
    
    
    corr_ls = []
    for g in range(len(data_G)):
        corr_gesture = []
        for i in range(len(data_G[g])):
            corr_signal = np.corrcoef(data_G[g][i])
            corr_gesture.append(corr_signal)
        corr_ls.append(corr_gesture)
    
    mean_corr_G = [0, 0, 0, 0, 0]
    for g in range(len(data_G)):
        corr_array = np.asarray(corr_ls[g])
        # mean_corr_G[g] = np.matrix.mean(corr_array, axis = 0)
        mean_corr_G[g] = corr_array.mean(0)
   
    return(mean_corr_G, THRESHOLD)
 
    
    
# ------------------------------------------------------------------------------
    
#    data_G = [0, 0, 0, 0, 0]
#    data_G[0] = data_test[labels_test == 0]
#    data_G[1] = data_test[labels_test == 1]
#    data_G[2] = data_test[labels_test == 2]
#    data_G[3] = data_test[labels_test == 3]
#    data_G[4] = data_test[labels_test == 4]
    
#    corr_dev_matrix = np.zeros((len(data_G), len(data_G)))   
#    for g in range(len(data_G)):
#        for g_fit in range(len(data_G)):
#            corr_dev_gg = []
#            for t in range(8):
#                corr_observed_signal = np.corrcoef(data_G[g][t])
#                corr_dev_gg.append(matrix_norm(corr_observed_signal - mean_corr_G[g_fit]))
#           
#            corr_dev_matrix[g][g_fit] = np.mean(corr_dev_gg)   
#            
            

    
# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

if False:
        
    m1 = np.array([[1, 2], [3, 4]])
    m2 = np.array([[-1,-2], [3, 4]])
    m_all = np.array([m1, m2])       
    m_all.mean(0)
    data_1 = np.array([ [0.1, 0.1, 1, 1], [0, 0, 1, 1], ])

    chdir("/Users/ZatorskiJacek/Git/classifier")
    data = np.load('epoched_data/epoched_data_bi256.npy')
    labels = np.load('epoched_data/all_labels.npy')
    
    data_train = data[:150]
    data_test = data[150:]
    
    labels_train = labels[:150]
    labels_test = labels[150:]
    labels = labels_train
    
    fit_model(data, labels)
   
