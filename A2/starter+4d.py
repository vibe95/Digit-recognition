from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd import elementwise_grad
from autograd.misc.optimizers import adam
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import expit as sigmoid

import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve

from data import load_mnist, plot_images, save_images

# Load MNIST and Set Up Data
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:10000])
train_labels = train_labels[0:10000]
test_images = np.round(test_images[0:10000])


#Q1C code 
def find_theta_MAP(train_images, train_labels):
    theta_map = np.zeros((10,784)) 
    
    #find theta for each digit
    for i in range(0,theta_map.shape[0]):
        img_digit_locs = train_labels[:,i]  #get all images locations that have a digit i
        current_data = np.transpose(train_images)
        Nd = np.dot(current_data,img_digit_locs)       #multiply data with current digit locs to get data only for current digit (true digit)
        N = np.sum(img_digit_locs)                     #get total points for current digit
        current_theta = np.divide((Nd+1),(N+2))
        theta_map[i,:] = current_theta #save the theta for this digit     
    return theta_map

#run Q1C code and save image
theta_map = find_theta_MAP(train_images, train_labels)
save_images(theta_map,'Q1C')



#Q1e code 
def find_log_likelihood(images, theta_map, pi_c):
    #find using formula generated in q1d
    likelihoods = np.zeros((images.shape[0],10))
    
    #find the likelihood for each digit/datapoint
    for digit in range(0,10): #loop through each digit 
        for i in range(0,images.shape[0]):
            current_data = images[i,:]  #get current data point
            current_theta = theta_map[digit]   #get theta for current digit
            likelihoods[i,digit]  = np.dot(current_data,np.log(current_theta))+np.dot((1-current_data),np.log(1-current_theta)) #q1d
            
    return likelihoods + np.log(pi_c) 

def avg_likelihood(images,label, log_likelihood):
    sum_likelihood = 0
    
    #go through each datapoint find all likelihoods for current_label
    for i in range(0,images.shape[0]):
        sum_likelihood = sum_likelihood + np.sum(log_likelihood[i,:]*label[i,:])  # sum of likelihoods for each image wrt. its label
    
    avg_likelihood = sum_likelihood/images.shape[0]    #divide by number of images to get average
    return avg_likelihood
    
    
def predict(images, theta_map, log_likelihood):
    predictions = np.zeros((images.shape[0],log_likelihood.shape[1])) #N by 10
    
    #find best class true class for each image
    for i in range(0,images.shape[0]):
        current_likelihood = log_likelihood[i,:]  #get likelihoods for current datapoint
        best_class = np.argmax(current_likelihood)   #choose the class with highest likelihood
        predictions[i,best_class] = 1  #set index = digit to 1 as it is the best prediction for current image
    
    return predictions    
    
def Q1E_report(train_images,train_labels,test_images,test_labels,theta_map):
    #average for train
    log_likelihood_train = find_log_likelihood(train_images, theta_map, 1/10)    
    avg_likelihood_train = avg_likelihood(train_images,train_labels,log_likelihood_train)
    print("Avg train log likelihood ",avg_likelihood_train)
     
    #average for test
    log_likelihood_test = find_log_likelihood(test_images, theta_map, 1/10)    
    avg_likelihood_test = avg_likelihood(test_images,test_labels,log_likelihood_test)
    print("Avg test log likelihood ",avg_likelihood_test)    
    
    #predictions for train
    predict_train = predict(train_images, theta_map, log_likelihood_train) 
    total_correct_train = np.sum(np.nonzero(predict_train)[1] == np.nonzero(train_labels)[1]) #get total number of correct predictions
    accuracy_train = total_correct_train/float(train_labels.shape[0])    #get accuracy
    print('Train Accuracy ',accuracy_train)
    
    #predictions for test
    predict_test = predict(test_images, theta_map, log_likelihood_test)
    total_correct_test = np.sum(np.nonzero(predict_test)[1] == np.nonzero(test_labels)[1]) #get total number of correct predictions
    accuracy_test = total_correct_test/float(test_labels.shape[0])    #get accuracy    
    print('Test Accuracy ',accuracy_test)

    
#run Q1e code
#Q1E_report(train_images,train_labels,test_images,test_labels,theta_map)



#Q2c code
def create_samples(num_of_samples,theta_map):
    sample_data = np.random.rand(num_of_samples,784)
    rand_cs = np.random.randint(num_of_samples, size=(1, 10))[0] #generate random numbers(digit class)

    for i in range(0,num_of_samples):
        #using p(x_d | c, θ_cd ) i.e. ancestral sampling
        c = rand_cs[i] #get the rand c
        current_theta = theta_map[c,:]  # get the thetha for the specific digit 
        xd = sample_data[i,:]    #get current rand sample

        #pick current random data point based on c value. Also binarize 
        xd[xd < current_theta] = 0
        xd[xd >= current_theta] = 1
        
        #save the updated xd 
        sample_data[i,:] = xd
    
    return sample_data



#run Q2C code 
#sample_data = create_samples(10,theta_map)
#save_images(sample_data,'Q2C')



#Code for Q2F
def mult_bern_likelihood(x, theta_c,c,stop):
    #compute multiple p(x_d|c,0_cd) = ber(x_d|0_cd)
    likelihood = np.ones(stop)
    
    for d in range(0,stop):
        theta_cd = theta_c[d]
        x_d = x[d]
        likelihood[d] = likelihood[d]*((theta_cd**x_d)*(1-theta_cd)**(1-x_d))
        
    return likelihood

def advanced_bayes(images, theta_map):
    half_size = int(images.shape[1]/2)  # where the top half of the image ends, should be 392

    #find for each image 
    for img_num in range(0, images.shape[0]):
        #numerator (P(x_inbottom and x_top))
        sum_overc_nume = np.zeros((half_size))
        for c in range (0,10):
            #p(x_inbottom|0)
            theta_c = theta_map[c,:]
            
            #p(x_top|0) from 0 to 392
            likelihood_num = mult_bern_likelihood(images[img_num,:half_size],theta_c[:half_size],c,half_size)
            
            #sum over c for p(x_inbottom|0)*p(x_top|0)
            sum_overc_nume = sum_overc_nume + likelihood_num*theta_c[half_size:]
            
        #Denominator (P(x_top))
        sum_overc_dem = np.zeros((half_size))
        for c in range (0,10):
            #p(x_inbottom|0)
            theta_c = theta_map[c,:]
            
            #p(x_top|0) from 0 ot 392
            likelihood_dem = mult_bern_likelihood(images[img_num,:half_size],theta_c[:half_size],c, half_size)
            
            #sum over c for p(x_top|0)
            sum_overc_dem = sum_overc_dem + likelihood_dem
            
        #divide num by dem to get final result (x in bottom) for current image
        images[img_num,half_size:] = np.divide(sum_overc_nume,sum_overc_dem)
    
    return images

#run 2F code
#results = advanced_bayes(train_images[0:20,:],theta_map)
#save_images(results,'Q2F')



#Q3C code
def one_per_class(images, labels):
    out_images = np.zeros((10,images.shape[1]))
    out_labels = np.zeros((10,10))
    classes = np.where(labels == 1)[1] # get the class digit for each image by getting column idx of ones in labels

    #get first image in training set with each class label
    for i in range(0,10):
        img_num = np.where(classes == i)[0][0]
        out_images[i,:] = images[img_num,:]
        out_labels[i,:] = labels[img_num,:]
    return out_images,out_labels

def cost_function(w):
    sum_final = 0 #temporary create sum_final var 
    dem = logsumexp(np.dot(np.transpose(w),grad_images))
    
    #mutliclass likelihood function is sum from 0 to k of label*predictive_log_likelihood
    for k in range(0,10):
        log_pc_x = np.dot(np.transpose(w[:,k]),grad_images) - dem

        if k == 0:
            sum_final = np.dot(grad_labels[k],log_pc_x)
        else:
            sum_final = sum_final + np.dot(grad_labels[k],log_pc_x)

    return sum_final
        


def logistic_gradient_desc(iterations,lr):
    #set globals so that cost function can access these values after usign autograd w.r.t. w
    global current_c
    global grad_images
    global grad_labels
    
    w = np.zeros((784,10)) #create the weights
    for i in range(0,iterations):
        for img_num in range(0,10):
            
            #get gradient of cost function/likelihood
            grad_images = new_images[img_num,:] #get current image
            grad_labels = new_labels[img_num,:] #get labels for current image
            current_c = img_num  #sinces we sampled 1 image for each class in order c = img_num
            cost_grad = elementwise_grad(cost_function)
            
            #update weights 
            w = w + lr*cost_grad(w)
            
        print(i)
    return w
    

#run Q3C code 
new_images,new_labels = one_per_class(train_images,train_labels)
grad_images=grad_labels = new_images #temporary just to create a gobal var for use with autograd
current_c = 0 #temporary just to create a gobal var for use with autograd
#weights = logistic_gradient_desc(5000, 0.01) #5000 iterations with a common learning rate of 0.01
#save_images(np.transpose(weights),'Q3c')



#Q3d code
def avg_pred_log(w,images):
    log_pc_x = 0
    for i in range(0,images.shape[0]):
        current_log_pc_x = np.dot(np.transpose(w),images[i,:]) - logsumexp(np.dot(np.transpose(w),images[i,:]))
        log_pc_x = log_pc_x + current_log_pc_x
        
    return np.sum(log_pc_x)/float(images.shape[0])

def predict_regression(images, w):
    predictions = np.zeros((images.shape[0],w.shape[1])) #N by 10
    
    #find best class true class for each image
    for i in range(0,images.shape[0]):
        best_class = np.argmax(np.dot(np.transpose(w),images[i,:]))   #choose the class with highest 
        predictions[i,best_class] = 1  #set index = digit to 1 as it is the best prediction for current image
    
    return predictions    
def Q3D_report(train_images,train_labels,test_images,test_labels,w):
    #average for train
    avg_likelihood_train = avg_pred_log(w,train_images)  
    print("Avg train log likelihood ",avg_likelihood_train)
     
    #average for test
    avg_likelihood_test = avg_pred_log(w,test_images) 
    print("Avg test log likelihood ",avg_likelihood_test)    
    
    #predictions for train
    predict_train = predict_regression(train_images, w) 
    total_correct_train = np.sum(np.nonzero(predict_train)[1] == np.nonzero(train_labels)[1]) #get total number of correct predictions
    accuracy_train = total_correct_train/float(train_labels.shape[0])    #get accuracy
    print('Train Accuracy ',accuracy_train)
    
    #predictions for test
    predict_test = predict_regression(test_images, w) 
    total_correct_test = np.sum(np.nonzero(predict_test)[1] == np.nonzero(test_labels)[1]) #get total number of correct predictions
    accuracy_test = total_correct_test/float(test_labels.shape[0])    #get accuracy    
    print('Test Accuracy ',accuracy_test)

    
#run Q3D code
#Q3D_report(train_images,train_labels,test_images,test_labels,weights)

# Starter Code for 4d
# A correct solution here only requires you to correctly write the neglogprob!
# Because this setup is numerically finicky
# the default parameterization I've given should give results if neglogprob is correct.
K = 30
D = 784

# Random initialization, with set seed for easier debugging
# Try changing the weighting of the initial randomization, default 0.01
init_params = npr.RandomState(0).randn(K, D) * 0.01

# Implemented batching for you
batch_size = 10
num_batches = int(np.ceil(len(train_images) / batch_size))
def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

# This is numerically stable code to for the log of a bernoulli density
# In particular, notice that we're keeping everything as log, and using logaddexp
# We never want to take things out of log space for stability
def bernoulli_log_density(targets, unnormalized_logprobs):
    # unnormalized_logprobs are in R
    # Targets must be 0 or 1
    t2 = targets * 2 - 1
    # Now t2 is -1 or 1, which makes the following form nice
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*t2)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def batched_loss(params, iter):
    data_idx = batch_indices(iter)
    return neglogprob(params, train_images[data_idx, :])

def neglogprob(params, data):
    # Implement this as the solution for 4c!
    ##METHOD 1     
    #pi_c = 1/K
    ##sum from 0 to 784 of second part in 4b 
    #log_prob = []
    #for i in range(0,data.shape[0]):
        #k_sum = np.zeros((1,10))
        
        #for c in range(1,K): 
            #mult = np.ones((1,10))
            ##mult = ((params[c,:]**data[i,:])*((1-params[c,:])**(1-data[i,:])))  #vectorized version of the loop, i hope
            #for d in range(0,data.shape[1]):
                #mult = mult* ((params[c,d]**data[i,d])*((1-params[c,d])**(1-data[i,d])))
                #mult = mult *bernoulli_log_density(data[i,d], params[c,d])

            #k_sum = k_sum + pi_c*mult
        #log_prob.append(np.sum(np.array(-1*np.log(k_sum)[0]))/10) 
    #print(log_prob)
    #return log_prob


    #METHOD 2 
    #derived formula from 4b
    pi_c = float(1)/K
    results = []
    for i in range (0,data.shape[0]):
        x = data[i,:]
        theta = params[i,:]
        #first term from 4b eq
        log_pi_c = np.log(pi_c)
        
        #second term from 4b eq
        second_term = 0
        for d in range(0,data.shape[1]):
            second_term = second_term + bernoulli_log_density(x[d], theta[0])
        print(second_term)
        #third term from 4b eq
        third_term = 0
        for c in range(2,K):
            first_power = log_pi_c
            for d in range(0,data.shape[1]):
                first_power = first_power + bernoulli_log_density(x[d], theta[c]) 
            power = first_power-(log_pi_c+second_term)
            third_term = third_term + 10**power
        log_prob = log_pi_c + second_term + np.log(1+third_term)
        print(log_prob)
        results.append(log_prob)
            
    return -1*np.array(results)

# Get gradient of objective using autograd.
objective_grad = elementwise_grad(batched_loss)

def print_perf(params, iter, gradient):
    if iter % 30 == 0:
        #save_images(sigmoid(params), 'q4plot.png')
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(sigmoid(params), ax)        
        print(batched_loss(params, iter))

# The optimizers provided by autograd can optimize lists, tuples, or dicts of parameters.
# You may use these optimizers for Q4, but implement your own gradient descent optimizer for Q3!
optimized_params = adam(objective_grad, init_params, step_size=0.2, num_iters=10, callback=print_perf)
#optimized_params = adam(objective_grad, init_params, step_size=0.2, num_iters=10000, callback=print_perf)


#Q4D code: just need to run 2fs code given optimized_params
results = advanced_bayes(train_images[0:20,:],optimized_params)
fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(111)
plot_images(images, ax)
save_images(results,'Q4D')
