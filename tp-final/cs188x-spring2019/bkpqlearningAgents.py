#!/usr/bin/env python
# coding: utf-8

# In[28]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[29]:


#import autoreload
#?autoreload
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[30]:


# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import numpy as np
import pprint


# In[31]:


def random_argmax(v):
    """Like np.argmax(), but if there are several "best" actions,
       chooses and returns one randomly"""
    arguments = np.argwhere(v == np.amax(v)).ravel()
    return np.random.choice(arguments)


# #### Tabular Q Learning
# 
# Q(s,a) is a dictionary with each state-action value it visits  

# In[32]:


# Parent class in learningAgents.py
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        "*** YOUR CODE HERE ***"
        #self.Q = {}
        self.Q = Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # max_a(Q[state, all actions])
        legalActions = self.getLegalActions(state)
        if not legalActions:
            value=0.0
        else:
            # TODO: Find a better way
            value=max([self.getQValue(state, a) for a in legalActions])
        return value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            action=None
        else:
            # TODO: Find a better way
            action=legalActions[random_argmax([self.getQValue(state, a) for a in legalActions])]
        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # epsilon decay
        epsmin = 0.01
        eps_decay = 0.9999
        self.epsilon = max(self.epsilon*eps_decay, epsmin)
        if util.flipCoin(self.epsilon):
            # Act randomly
            action = random.choice(legalActions)
        else:
            # Act greedly
            action = self.computeActionFromQValues(state)
        
        return action
        
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        iteration = self.episodesSoFar
        self.alpha = 1/np.power((iteration+1), 1) # alpha decay
        alpha = self.alpha
        gamma = self.discount
        # -----------------------------v revisar si calculo maximo Q
        estimation = reward + gamma*self.computeValueFromQValues(nextState)
        self.Q[(state, action)] += alpha*(estimation - self.Q[(state, action)])
        #print("Q size:"+str(len(self.Q)), end="\r")

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


# In[33]:


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate/step size
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


# In[34]:


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        #extractor = 'CoordinateExtractor'
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        featureDict = self.featExtractor.getFeatures(state, action)
#         for feat in featureDict.keys():
#             self.weights[feat]*featureDict[feat]
        #print("aprox Q value: ", np.dot(self.weights, featureDict))
        return np.dot(self.weights, featureDict)
    
    def getMaxQValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # max_a(Q[state, all actions])
        legalActions = self.getLegalActions(state)
        if not legalActions:
            value=0.0
        else:
            # TODO: Find a better way
            value=max([self.getQValue(state, a) for a in legalActions])
        return value
    
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        iteration = self.episodesSoFar
        self.alpha = 1/np.power((iteration+1), 1) # alpha decay
        alpha = self.alpha
        gamma = self.discount
        #state = str(state)
        featureDict = self.featExtractor.getFeatures(state, action)
        #for key,feat in 
        estimation = reward + gamma*self.getMaxQValue(nextState)
        pastVal = self.getQValue(state, action)
        for feature in featureDict.keys():
            #print("state: ", state, " action: ", action)
            self.weights[feature] += alpha * (estimation - pastVal) * featureDict[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            
#             print("Weights:")
#             pprint.pprint(self.weights)
#             print("Features:")
#             for k in self.weights.keys():
#                 state, action = k
#                 pprint.pprint(self.featExtractor.getFeatures(state, action))

#             print(len(self.weights))
            pass


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.save_notebook()')


# In[ ]:


from subprocess import call
command = ('ipython nbconvert --to script qlearningAgents.ipynb')
call(command, shell=True)

get_ipython().run_line_magic('autoreload', '2')


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


# state:  
# %%%%%%%
# %  G> %
# % %%% %
# % %.  %
# % %%% %
# %.    %
# %%%%%%%
# Score: -8

# action:  East


# In[ ]:




