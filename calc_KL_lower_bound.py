
"""
[Calculation of lower bound for the KL-divergence]
Created on Wd Jul 03 2019

Copyright (c) [2019] [Tomohiro Nishiyama]

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

"""
URL of the preprint 

https://arxiv.org/abs/1907.00288
"""

import numpy as np
import matplotlib.pyplot as plt

#The function to calculate the lower bound of the KL-divergence (Theorem 1 in the preprint)
#input ########
#EP, EQ : the expectation value of FoI
#VP, VQ : the variance of FoI

def calc_KL_lowerbound(EP, EQ, VP, VQ):
    A = (EQ - EP)**2 + VP + VQ
    D = np.sqrt(A**2 - 4 * VP * VQ)
    x = D / A
    kl_lb = 0.5 * (A - 2 * VP) / D * np.log((1 + x) / (1 - x)) + 0.5 * np.log(VP / VQ)
    return kl_lb

#The function to calculate the actual KL-divergence
#input ########
#param_P, param_Q : the parameters of the distribution
    
def calc_KL_divergence(param_P, param_Q, dist_case):

    if dist_case == 1: # normal distribution
        mu_P = param_P[:,0]
        mu_Q = param_Q[:,0]
        sigma_P = param_P[:,1]
        sigma_Q = param_Q[:,1]
        return 0.5 * (mu_Q- mu_P)**2 / sigma_P + 0.5 * sigma_P / sigma_Q - 0.5 + 0.5 * np.log(sigma_Q / sigma_P)
    elif dist_case == 2:  # exponential distribution
        nu_P = param_P[:,0]
        nu_Q = param_Q[:,0]
        return nu_P / nu_Q - 1 + np.log(nu_Q/ nu_P)
    elif dist_case == 3:  # Bernoulli distribution
        p = param_P[:,0]
        q = param_Q[:,0]
        return (1 - p) * np.log((1 - p) / (1 - q)) + p * np.log(p / q)
    else:
        print('error')
    
#case = 2 #1: normal distribution, 2: exponential distribution, 3: Bernolli distribution

#calc_case 
#1. normal distribution with sigma_P=sigma_Q=1
#2. normal distribution with mu_P = mu_Q=0
#3. exponential distribution 
#4. Bernoulli distribution 
calc_case = 3

if calc_case == 1:
    x = np.arange(0, 2.05 , 0.05)
    datanum = len(x)
    param_P = np.zeros((datanum, 2))
    param_Q = np.zeros((datanum, 2))
  
    param_P[:,0] = np.zeros(datanum)  #mean
    param_Q[:,0] = x                  #mean
    param_P[:,1] = 1                #variance
    param_Q[:,1] = 1               #variance
    
     #FoI = x       
    EP = param_P[:,0]
    EQ = param_Q[:,0]
    VP = param_P[:,1] 
    VQ = param_Q[:,1]
    
    dist_case = 1 #normal distribution
    KL = calc_KL_divergence(param_P, param_Q, dist_case)
    LB = calc_KL_lowerbound(EP, EQ, VP, VQ)
    
elif calc_case == 2:
    x = np.arange(0.5, 2.05 , 0.05)
    datanum = len(x)
    param_P = np.zeros((datanum, 2))
    param_Q = np.zeros((datanum, 2))
    
    param_P[:,0] = np.zeros(datanum) #mean
    param_Q[:,0] = param_P[:,0]        #mean
    param_P[:,1] = np.ones(datanum)  #variance
    param_Q[:,1] = x**2              #variance
    
    #FoI = x^2
    EP = param_P[:,1]
    EQ = param_Q[:,1] 
    VP = 2 * param_P[:,1]**2
    VQ = 2 * param_Q[:,1]**2
    
    dist_case = 1 #normal distribution
    KL = calc_KL_divergence(param_P, param_Q, dist_case)
    LB = calc_KL_lowerbound(EP, EQ, VP, VQ)
        
elif calc_case == 3:    
    x = np.arange(0.5, 2.05 , 0.05)
    datanum = len(x)
    param_P = np.zeros((datanum, 2))
    param_Q = np.zeros((datanum, 2))
   
    param_P[:,0] = np.ones(datanum)  #mean
    param_Q[:,0] = x                 #mean
    
    #FoI = x
    EP = param_P[:,0]
    EQ = param_Q[:,0]
    VP = param_P[:,0]**2    
    VQ = param_Q[:,0]**2   
    
    dist_case = 2 #exponential distribution                
    KL = calc_KL_divergence(param_P, param_Q, dist_case)
    LB = calc_KL_lowerbound(EP, EQ, VP, VQ)
    
elif calc_case == 4:
    x = np.arange(0.05, 1.00 , 0.05)
    datanum = len(x)
    param_P = np.zeros((datanum, 2))
    param_Q = np.zeros((datanum, 2))
    
    param_P[:,0] = 0.4 * np.ones(datanum) 
    param_Q[:,0] = x       
    
    #FoI = x       
    EP = param_P[:,0]   
    EQ = param_Q[:,0]   
    VP = param_P[:,0] * (1 - param_P[:,0])   
    VQ = param_Q[:,0] * (1 - param_Q[:,0])        
   
    dist_case = 3 #Bernoulli distribution      
    KL = calc_KL_divergence(param_P, param_Q, dist_case) 
    LB = calc_KL_lowerbound(EP, EQ, VP, VQ)

else:
    print('error')
#del plt.font_manager.weight_dict['roman']
#plt.font_manager._rebuild()        

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams["mathtext.fontset"] = "stix" 
plt.rcParams["font.size"] = 14
plt.rcParams['axes.linewidth'] = 1.0# 
plt.rcParams['axes.grid'] = True
            
ratio = LB / KL
ratio[np.abs(KL) < 1e-3] = 1 #To avoid division by zero
      
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.xlabel(r'$\beta$'+'\n (a)', fontsize = 14)
plt.plot(x, KL,label='KL-divergence', color = 'blue')
plt.plot(x, LB,label='Lower bound', color = 'red')
plt.legend()

plt.subplot(1,2,2)
plt.xlabel(r'$\beta$'+'\n (b)', fontsize = 14)

plt.plot(x, ratio, color = 'blue')

plt.show()
#plt.savefig('test.png')


