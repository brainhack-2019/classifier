import numpy as np


# ------------------------------------------------------------------------------
# Define functions
# ------------------------------------------------------------------------------

# Frobenius matrix norm
def matrix_norm(matrix):
    """
    Evaluate Frobenius matrix norm
    INPUT
    matrix -- a matrix (numpy array)
    OUTPUT
    Frobenius norm of the matrix (numpy float)
    """
    return(np.sqrt(np.nansum(np.square(matrix))))


# Predict (recognize) a gesture given observed signal
def predict(fitted_model, observed_signal):  
    """
    Function predicts (classifies) an observed signal either as one of
    the gestures or no gesture.
    INPUT
    fitted_model -- a list: [fitted_corr_matrics, threshold],
                    where: fitted_corr_matrics = np.array(corr_gesture_1, ...)
                    threshold is a detection threshold
    observed_signal -- recorded EMG for n-channels
    OUTPUT
    Gesture index (int) if a gesture was detected, -1 for no gesture
    """

    
    # Evaluate correlations between channels of observed signal
    #TODO Check if with real data we need transposition (.T)
    corr_observed_signal = np.corrcoef(observed_signal)
    
    # Compute distances between observed signal and fitted models
    distance_ls = [ matrix_norm(corr_observed_signal - fitted_model[0][i])\
                                for i in range(len(fitted_model[0])) ]
    
    # Detection
    threshold = fitted_model[1]
    min_distance_index = distance_ls.index(min(distance_ls))
    if distance_ls[min_distance_index] < threshold:
        return(min_distance_index)
    else:
        return(-1)
    

# ------------------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------------------

if False:
    
    # Load data
    chdir("/Users/ZatorskiJacek/Git/classifier")
    data = np.load('epoched_data/epoched_data_bi256.npy')
    labels = np.load('epoched_data/all_labels.npy')
    
    # Fit model
    fitted_model = fit_model(data, labels)
    
    # Predict
    observed_signal = data[0]
    labels[0]
    predict(fitted_model, observed_signal)
    
    
    
    
    
        
    
    
    

    