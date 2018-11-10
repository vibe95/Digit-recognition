import data
import numpy as np
import matplotlib.pyplot as plt

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    a=2  #alpha value
    b=2  #beta value

    #loop through each digit and each feature to find thetamap 
    for i in range(0,eta.shape[0]):
        current_data = data.get_digits_by_label(train_data, train_labels, i) #get the data realated to the current digit
        current_features = np.zeros((current_data.shape[1]))     #create any empty array of 64 features
        for feature in range(0,current_data.shape[1]): #for each feature caluculate theta_map
            Nc = np.count_nonzero(current_data[:,feature])  #count number of ones for current feature accross all current_datapoints(datapoints for current digit)
            N = current_data.shape[0]   #total number of current data points
            thetha_map = (Nc + a -1)/(N+a+b-2)
            current_features[feature] = thetha_map  #save thethamap to this feature
        eta[i] = current_features #save the array of features for this digit 
        
    return eta

def main():
    #load the data 
    N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()
    train_size = 10000 #number of train_data images
    bin_train_data = np.where(bin_train_data[:train_size,:] >=0.5, 1, 0)
    bin_test_data = np.where(bin_train_data >=0.5, 1, 0)
    print(N_data)
        

if __name__ == '__main__':
    main()