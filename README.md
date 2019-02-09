# Udacity DRLND: Collaboration and Competition

Andrzej Wodecki

February 7th, 2019



## Project details

The goal of the project is to **train two Agents to bounce a ball over a net** in the Tennis environment provided by Unity Environments. After training they should be able to play for a longer time without having the ball hit the ground or fall out of bounds. 

**The state space** has 8 variables like the position and velocity of the ball and racket, and each agent receives it's own, local observations.

**The action space** consists of 2  continuos actions: a movement (toward or away from the net) and jumping.

This is **episodic** environment. It is considered **solved** when agents get an average score of +0.5 over 100 consecutive episodes, with the score beeing the maximum of the scores of both agents.



## Getting started

First, you will need the Tennis Environment provided by Unity - the simplest way is to follow the instruction provided by Udacity and available [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).

You will also need a set of python packages installed, including jupyter, numpy and pytorch. All are provided within UDACITY "drlnd" environment: follow the instructions provided eg. [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Specifically, create and activate a new environment with Python 3.6:
`conda create --name drlnd python=3.6`
`source activate drlnd`



Finally, you should have an agent simulator  be installed: for Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip), for Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip) and for Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip).



## Instructions 

The structure of the code is the following:

1. *run.py* is the main code. Here all the parameters are read,  training procedures called and the results written to the appropriate files and folders.
2. *parameters.py* stores all the hyper parameters - the structure of this file is presented in more details  in the *Hyperparameter grid search* section of the *Report.md*.
3. all the results are stored in (see *Hyperparameter grid search* section of the *Report.md*):
   1. *results.txt* file
   2. *models/* subdirectory.

To run the code:

1. Specify hyperparameters in the *parameters.py*. Be careful: too many parameters may results with a very long computation time!
2. run the code by typing: *python run.py*
3. ... and check results: both on the screen and in the output files/folders.
