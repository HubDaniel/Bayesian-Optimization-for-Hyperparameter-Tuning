# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:15:17 2022

@author: @author: reference.  https://github.com/fmfn/BayesianOptimization
"""
import os
#os.chdir('C:\\PSTAT-232\\final')

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

from bayesian_optimization import BayesianOptimization
from util import Colours
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_data(seed):
    """Synthetic binary classification dataset."""
    data, targets = make_classification(
        n_samples=1000,
        n_features=45,
        n_informative=12,
        n_redundant=7,
        random_state=seed,
    )
    return data, targets


def svc_cv(C, gamma, data, targets):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    estimator = SVC(C=C, gamma=gamma, random_state=2)
    cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=4)
    return cval.mean()


def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )
    #cval = cross_val_score(estimator, data, targets,
    #                       scoring='neg_log_loss', cv=4)
    cval = cross_val_score(estimator, data, targets,
                           scoring='roc_auc', cv=4)
    return cval.mean()


def optimize_svc(data, targets):
    """Apply Bayesian Optimization to SVC parameters."""
    def svc_crossval(expC, expGamma):
        """Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svc_cv(C=C, gamma=gamma, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=50)

    print("Final result:", optimizer.max)
    return(optimizer._targets)


def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (0.1, 0.999),
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=200)#,init_points=1)

    print("Final result:", optimizer.max)
    return(optimizer._targets)


############################################################
############################################################
############################################################

nSim = 50
svcResults, rfcResults = np.zeros(shape=(nSim,55)), np.zeros(shape=(nSim,55))


for i in range(nSim): 
    print("Simulation:", i)
    data, targets = get_data(i)
    # bayesian hyperparameter tuning    
    svc_targets = optimize_svc(data, targets)
    rfc_targets = optimize_rfc(data, targets)
    # find max target values so far
    svc_cur_max = svc_targets[:]
    rfc_cur_max = rfc_targets[:]
    for j in range(len(rfc_cur_max)):
        #svc_cur_max[j] = max(svc_targets[:j+1])
        rfc_cur_max[j] = max(rfc_targets[:j+1])
    svcResults[i,] = svc_cur_max
    rfcResults[i,] = rfc_cur_max

np.savetxt('svc_nuinf_ucb.csv', svcResults, delimiter=',')
np.savetxt('rfc_nuinf_ucb.csv', rfcResults, delimiter=',')  

    
fig, ax=plt.subplots()
ax.plot(rfc_cur_max)


# randomized search for random forest
rfrResults = np.zeros(shape=(nSim,55))
for i in range(nSim): 
    print("Simulation:", i)
    data, targets = get_data(i)
    # randomizedSearch tuning
    rf = RFC(random_state = 2)
    param_dist = {"n_estimators": randint(10,250),
                  "max_features": uniform(0.1, 0.899),
                  "min_samples_split": randint(2,25)}
    rf_cv = RandomizedSearchCV(rf, param_dist, cv=4,scoring="roc_auc",n_jobs=14,n_iter=55)
    rf_cv.fit(data,targets)
    rfr_targets = rf_cv.cv_results_['mean_test_score']  
    # find max target values so far
    rfr_cur_max = rfr_targets[:]
    for j in range(len(rfr_cur_max)):
        rfr_cur_max[j] = max(rfr_targets[:j+1])
    rfrResults[i,] = rfr_cur_max

np.savetxt('rfr.csv', rfrResults, delimiter=',')
#fig, ax=plt.subplots()
#ax.plot(rfr_cur_max)





# randomized search for support vector classifier
svrResults = np.zeros(shape=(nSim,55))
for i in range(nSim): 
    print("Simulation:", i)
    data, targets = get_data(i)
    # randomizedSearch tuning
    sv = SVC(random_state = 2)
    param_dist = {"C": uniform(0.001, 99.999), 
                  "gamma": uniform(0.0001,0.1-0.0001)}
    sv_cv = RandomizedSearchCV(sv, param_dist, cv=4,scoring="roc_auc",n_jobs=14,n_iter=55)
    sv_cv.fit(data,targets)
    svr_targets = sv_cv.cv_results_['mean_test_score']  
    # find max target values so far
    svr_cur_max = svr_targets[:]
    for j in range(len(svr_cur_max)):
        svr_cur_max[j] = max(svr_targets[:j+1])
    svrResults[i,] = svr_cur_max

np.savetxt('svr.csv', svrResults, delimiter=',')





############################################################
############################################################
############################################################


# compare all together
fig, ax=plt.subplots(1,2)
ax[0].set_ylabel("aoc-auc")
ax[0].set_xlabel("number of evaluations") 
ax[0].set_title("support vector classifier")


colors = ["r","c","m","y"]
for count, nus in enumerate(["0.5","1.5","2.5","inf"]):
    color = colors[count]
    for im in ["ei","poi","ucb"]:        
        readin = pd.read_csv("svc_nu"+nus+"_"+im+".csv",header=None)
        readinmean = np.mean(readin, axis=0)
        readinstd = np.std(readin, axis=0)
        #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd)
        if im=="ei":
            ax[0].plot(range(55),readinmean,color=color,label="ei_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd)
        elif im=="poi":
            ax[0].plot(range(55),readinmean,linestyle="dotted",color=color,label="poi_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd,linestyle="dotted")
        else:
            ax[0].plot(range(55),readinmean,linestyle="dashdot",color=color,label="ucb_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd,linestyle="dotted")
            
readin = pd.read_csv("svr.csv",header=None)
readinmean = np.mean(readin, axis=0)
readinstd = np.std(readin, axis=0)  
ax[0].plot(range(55),readinmean,color="k",label="randSearch",linestyle="dashed")
            
ax[0].legend(loc="lower right")




ax[1].set_ylabel("aoc-auc")
ax[1].set_xlabel("number of evaluations") 
ax[1].set_title("random forest classifier")
        
for count, nus in enumerate(["0.5","1.5","2.5","inf"]):
    color = colors[count]
    for im in ["ei","poi","ucb"]:        
        readin = pd.read_csv("rfc_nu"+nus+"_"+im+".csv",header=None)
        readinmean = np.mean(readin, axis=0)
        readinstd = np.std(readin, axis=0)
        #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd)
        if im=="ei":
            ax[1].plot(range(55),readinmean,color=color,label="ei_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd)
        elif im=="poi":
            ax[1].plot(range(55),readinmean,linestyle="dotted",color=color,label="poi_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd,linestyle="dotted")
        else:
            ax[1].plot(range(55),readinmean,linestyle="dashdot",color=color,label="ucb_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd,linestyle="dotted")
            
readin = pd.read_csv("rfr.csv",header=None)
readinmean = np.mean(readin, axis=0)
readinstd = np.std(readin, axis=0)  
ax[1].plot(range(55),readinmean,color="k",label="randSearch",linestyle="dashed")   

ax[1].legend(loc="lower right")
        
fig.tight_layout()

plt.savefig('all.png', format='png', dpi=800)









# zoom in
fig, ax=plt.subplots(1,2)
ax[0].set_ylabel("aoc-auc")
ax[0].set_xlabel("number of evaluations") 
ax[0].set_title("support vector classifier")
ax[0].set_ylim([0.975,0.98])

for count, nus in enumerate(["0.5","1.5","2.5","inf"]):
    color = colors[count]
    for im in ["ei","poi","ucb"]:        
        readin = pd.read_csv("svc_nu"+nus+"_"+im+".csv",header=None)
        readinmean = np.mean(readin, axis=0)
        readinstd = np.std(readin, axis=0)
        #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd)
        if im=="ei":
            ax[0].plot(range(55),readinmean,color=color,label="ei_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd)
        elif im=="poi":
            ax[0].plot(range(55),readinmean,linestyle="dotted",color=color,label="poi_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd,linestyle="dotted")
        else:
            ax[0].plot(range(55),readinmean,linestyle="dashdot",color=color,label="ucb_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd,linestyle="dotted")
            
readin = pd.read_csv("svr.csv",header=None)
readinmean = np.mean(readin, axis=0)
readinstd = np.std(readin, axis=0)  
ax[0].plot(range(55),readinmean,color="k",label="randSearch",linestyle="dashed")
            
ax[0].legend(loc="upper left",fontsize=6)




ax[1].set_ylabel("aoc-auc")
ax[1].set_xlabel("number of evaluations") 
ax[1].set_title("random forest classifier")
ax[1].set_ylim([0.9515,0.957])
        
for count, nus in enumerate(["0.5","1.5","2.5","inf"]):
    color = colors[count]
    for im in ["ei","poi","ucb"]:        
        readin = pd.read_csv("rfc_nu"+nus+"_"+im+".csv",header=None)
        readinmean = np.mean(readin, axis=0)
        readinstd = np.std(readin, axis=0)
        #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd)
        if im=="ei":
            ax[1].plot(range(55),readinmean,color=color,label="ei_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd)
        elif im=="poi":
            ax[1].plot(range(55),readinmean,linestyle="dotted",color=color,label="poi_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd,linestyle="dotted")
        else:
            ax[1].plot(range(55),readinmean,linestyle="dashdot",color=color,label="ucb_"+nus)
            #ax[0].errorbar(range(results.shape[0]),readinmean,yerr=readinstd,linestyle="dotted")
            
readin = pd.read_csv("rfr.csv",header=None)
readinmean = np.mean(readin, axis=0)
readinstd = np.std(readin, axis=0)  
ax[1].plot(range(55),readinmean,color="k",label="randSearch",linestyle="dashed")   

ax[1].legend(loc="upper left")

ax[1].legend(loc="upper left",fontsize=6)
        
fig.tight_layout()

plt.savefig('zoomin.png', format='png', dpi=800)

#plt.show()




