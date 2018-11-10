from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
from autograd import elementwise_grad

import autograd.scipy.stats.norm as norm
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.multivariate_normal as mvn

from data import load_mnist, plot_images, save_images
import matplotlib.pyplot as plt

# Load MNIST and Set Up Data
N = 300
D = 784
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:N])
train_labels = train_labels[0:N]
test_images = np.round(test_images[0:10000])
test_labels = test_labels[0:10000]



#A3 Q1A
#A2 Q3C code
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
        for img_num in range(0,new_images.shape[0]):
            
            #get gradient of cost function/likelihood
            grad_images = new_images[img_num,:] #get current image
            grad_labels = new_labels[img_num,:] #get labels for current image
            current_c = img_num  #sinces we sampled 1 image for each class in order c = img_num
            cost_grad = elementwise_grad(cost_function)
            
            #update weights 
            w = w + lr*cost_grad(w)
            
        print(i)
    return w
    

#run A2 Q3C code 
new_images = train_images
new_labels = train_labels
grad_images=grad_labels = new_images #temporary just to create a gobal var for use with autograd
current_c = 0 #temporary just to create a gobal var for use with autograd
weights = logistic_gradient_desc(1000, 0.01) #5000 iterations with a common learning rate of 0.01
save_images(np.transpose(weights),'Q1a')



#A2 Q3d code
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

    
#run A2 Q3D code
Q3D_report(train_images,train_labels,test_images,test_labels,weights)  #REPORT FOR A3 Q1a


#A3 Q1C 

#logitsic regression with gradient descent using map
def grad_desc(iterations,lr,sigma):
    #set globals so that cost function can access these values after usign autograd w.r.t. w
    global grad_images
    global grad_labels
    
    w = np.zeros((784,10)) #create the weights
    for i in range(0,iterations):
        for img_num in range(0,new_images.shape[0]):
            
            #get gradient of cost function/likelihood
            grad_images = new_images[img_num,:] #get current image
            grad_labels = new_labels[img_num,:] #get labels for current image
            cost_grad = elementwise_grad(cost_function)
            
            #update weights 
            w = w + lr*cost_grad(w)
        
        #NEW ADDITION FOR A3
        w = w - w/sigma**2
            
        print(i)
    return w    



#run cod for A3 q1c 
print("Map logitsic regression")

##Testing for best sigma value was 36
#for i in range(1,10):
    #sigma = i**2  # from 5 to 100
    #print(sigma)
    #map_weights = grad_desc(100, 0.01,sigma) #5000 iterations with a common learning rate of 0.01
    #save_images(np.transpose(map_weights),'Q1c')
    #Q3D_report(train_images,train_labels,test_images,test_labels,map_weights)
    
    
    #sigma = 1/i**2   #from 1 to 1/100
    #print(sigma)
    #map_weights = grad_desc(100, 0.01,sigma) #5000 iterations with a common learning rate of 0.01
    #save_images(np.transpose(map_weights),'Q1c')
    #Q3D_report(train_images,train_labels,test_images,test_labels,map_weights)    
sigma = 36  
map_weights = grad_desc(1000, 0.01,sigma) #5000 iterations with a common learning rate of 0.01
save_images(np.transpose(map_weights),'Q1c')
Q3D_report(train_images,train_labels,test_images,test_labels,map_weights)




K = 10
prior_std = 1.0

# Choose two pixels and plot the K specific weights against eachother
contourK = 2
px1 = 392 # Middle Pixel
px2 = px1 + 28*5 +1 # Middle Pixel + 5 rows down
#px2 = px1+14 # Middle left-most edge

# Random initialization, with set seed for easier debugging
# Try changing the weighting of the initial randomization, default 0.01
init_params = (npr.RandomState(0).randn(K, D) * 0.01, npr.RandomState(1).randn(K, D) * 0.01)



def logistic_logprob(params, images, labels):
    # params is a block of S x K x D params
    # images is N x D
    # labels is N x K one-hot
    # return S logprobs, summing over N
    mul = np.einsum('skd,nd->snk', params, images)
    normalized = mul - logsumexp(mul, axis=-1, keepdims=True)
    return np.einsum('snk,nk->s', normalized, labels)

def diag_gaussian_log_density(x, mu, log_std):
    # assumes that mu and log_std are (S x K X D),
    # so we sum out the last two dimensions.
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=(-1, -2))

def sample_diag_gaussian(mean, log_std, num_samples, rs):
    return rs.randn(num_samples, *np.shape(mean)) * np.exp(log_std) + mean




def elbo_estimate(var_params, logprob, num_samples, rs):
    """Provides a stochastic estimate of the variational lower bound.
    var_params is (mean, log_std) of a Gaussian."""
    mean, log_std = var_params
    samples = sample_diag_gaussian(mean,log_std,num_samples,rs)     
    log_ps = logprob(samples)
    log_qs = diag_gaussian_log_density(samples,mean,log_std)   
    E_q = np.sum(log_ps-log_qs)/num_samples   # E_q(z|x)[log p(x,z) - log q(z|x)]
    return E_q    

def logprob_given_data(params):
    data_logprob = logistic_logprob(params,train_images,train_labels)   
    prior_logprob =  np.sum(np.sum(-np.log(np.sqrt(2*np.pi*prior_std))-(params**2)/(2*prior_std),axis=2),axis=1) 
    return data_logprob + prior_logprob

def objective(var_params, iter):
    return -elbo_estimate(var_params, logprob_given_data,
                          num_samples=100, rs=npr.RandomState(iter))


# Code for plotting the isocontours below
def logprob_given_two(params, two_params):
    N = two_params.shape[0]

    params_adjust = np.zeros((N, K, D))
    params_adjust[:, contourK, px1] = two_params[:, 0]
    params_adjust[:, contourK, px2] = two_params[:, 1]

    adjusted_params = params_adjust #+ params

    return logprob_given_data(adjusted_params)

# Set up plotting code
def plot_isocontours(ax, func, xlimits=[-10, 10], ylimits=[-10, 10], numticks=21, **kwargs):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, **kwargs)
    ax.set_yticks([])
    ax.set_xticks([])

def plot_posterior_contours(mean_params,logstd_params):
    plt.clf()
    logprob_adj = lambda two_params: logprob_given_two(mean_params, two_params)
    plot_isocontours(ax, logprob_adj, cmap='Blues')
    mean_2d = mean_params[contourK, [px1,px2]]
    logstd_2s = logstd_params[contourK, [px1,px2]]
    variational_contour = lambda x: mvn.logpdf(x, mean_2d, np.diag(np.exp(2*logstd_2s)))
    plot_isocontours(ax, variational_contour, cmap='Reds')
    plt.draw()
    plt.pause(10)

# Set up figure.
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

# Get gradient of objective using autograd.
objective_grad = grad(objective)

def print_perf(var_params, iter, gradient):
    mean_params, logstd_params = var_params
    print(".", end='')
    if iter % 30 == 0:
        save_images(mean_params, 'a3plotmean.png')
        save_images(logstd_params, 'a3plotsgd.png')
        sample = sample_diag_gaussian(mean_params, logstd_params, num_samples=1, rs=npr.RandomState(iter))
        save_images(sample[0, :, :], 'a3plotsample.png')

        ## uncomment for Question 2f)
        plot_posterior_contours(mean_params,logstd_params)

        print(iter)
        print(objective(var_params,iter))

# The optimizers provided by autograd can  optimize lists, tuples, or dicts of parameters.
# You may use these optimizers for Q4, but implement your own gradient descent optimizer for Q3!
optimized_params = adam(objective_grad, init_params, step_size=0.05, num_iters=1000, callback=print_perf)


#predictions for test
predict_test = predict_regression(test_images, np.transpose(optimized_params[0])) #A3 q1/A2 code 
total_correct_test = np.sum(np.nonzero(predict_test)[1] == np.nonzero(test_labels)[1]) #get total number of correct predictions
accuracy_test = total_correct_test/float(test_labels.shape[0])    #get accuracy    
print('\nTest Accuracy ',accuracy_test)

##testing a bunch of std values, std = 1 is best with 77.68% and std =9 is second best 77.09%
#for i in range(1,10):
    #prior_std = i**2  # from 5 to 100
    #print(prior_std)
    #optimized_params = adam(objective_grad, init_params, step_size=0.05, num_iters=100, callback=print_perf) 
    #predict_test = predict_regression(test_images, np.transpose(optimized_params[0])) #A3 q1/A2 code 
    #total_correct_test = np.sum(np.nonzero(predict_test)[1] == np.nonzero(test_labels)[1]) #get total number of correct predictions
    #accuracy_test = total_correct_test/float(test_labels.shape[0])    #get accuracy    
    #print('\nTest Accuracy ',accuracy_test)    
    
    #prior_std = 1/i**2   #from 1 to 1/100
    #print(prior_std)
    #optimized_params = adam(objective_grad, init_params, step_size=0.05, num_iters=100, callback=print_perf) 
    #predict_test = predict_regression(test_images, np.transpose(optimized_params[0])) #A3 q1/A2 code 
    #total_correct_test = np.sum(np.nonzero(predict_test)[1] == np.nonzero(test_labels)[1]) #get total number of correct predictions
    #accuracy_test = total_correct_test/float(test_labels.shape[0])    #get accuracy    
    #print('\nTest Accuracy ',accuracy_test)