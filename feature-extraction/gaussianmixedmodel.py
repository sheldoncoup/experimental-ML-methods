import numpy as np
from math import log, exp
from scipy.stats import multivariate_normal


class GaussianMixed:
  def __init__(self, class_list, threshold, mixmodel, epsilon):
    # Assign all input variables to the class and initalize empty distribution dictionary
    self.thresh = threshold
    self.mixmodel = mixmodel
    self.epsilon = epsilon
    self.distributions = { species:None for species in class_list}
      
  def fit(self, X,y):
    
    # find the shape of the X matrix (num_examples, example dimension)
    X_shape = np.array(X).shape
    num_examples = X_shape[0]
    
    # Initialize empty covariance matrix and a dictionary to store the species examples in
    avg_cov = np.zeros((X_shape[1], X_shape[1]))
    split_by_species = {}
    
    # Calculate the weighted average covariance matrix
    for species in self.distributions.keys():
      X_species = [x[0] for x in zip(X,y) if x[1] == species]
      split_by_species[species] = X_species
  
      avg_cov = avg_cov + ((len(X_species)/num_examples) * np.cov(X_species, rowvar=False))
    
    avg_cov = avg_cov + np.identity(avg_cov.shape[0]) * self.epsilon

    # Calculate the gaussian distribution for each species   
    for species in self.distributions.keys():
      X_species = split_by_species[species]

      self.distributions[species]  = multivariate_normal(mean=np.mean(X_species, axis=0), cov=avg_cov)
      
  
  def predict(self, X):
    #predicts if the examples from X are in the known distribution
    probs = []
    scores = []
    results = []
    
    # Calcualte the log of the probability density function for each example
    for species, dist in self.distributions.items():
      probs.append(dist.logpdf(X))
    
    probs = np.transpose(probs)
    
    # Threshold the logpdf either inidvidually or by mixing (averaging) 
    if self.mixmodel:
      
      def elnsum(elnlist):
        if len(elnlist) == 2:
          return elnlist[0]  + log(1 + exp(elnlist[1] - elnlist[0]) )
        elif len(elnlist) > 2:
          return elnlist[0] + log( 1 + exp(elnsum(elnlist[1:]) - elnlist[0] ) )
        else:
          raise ValueError('The list is not of an acceptable length.')
        
      for x in probs:
        x = sorted(x, reverse = True)
        mixed_prob = elnsum(x)
        scores.append(mixed_prob)
        results.append(mixed_prob >= self.thresh)
    else:
      for x in probs:
        mixed_prob = np.max(x)
        scores.append(mixed_prob)
        results.append(mixed_prob >= self.thresh)
    #print(scores[:10])
    return [1 if x else -1 for x in results], scores
