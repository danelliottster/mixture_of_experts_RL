# mixture_of_experts_RL

## Crowd ensemble

These ensembles are defined by a "crowd" of models which are only loosely coupled.  Their outputs are combined using simple operators such as means or voting.  Training is done seperately.  Each ensemble member is trained by sampling the experience replay memory.

### Required components

#### Model initialization

Set the initial parameter values for the ensemble members.

#### Update ensemble member

Inputs:
  * Experience memory
  * Optimizer
  * Model
  
 Outputs:
  

#### Combination method

### Optional components

#### Shared layers
blah blah

## Mixture of experts

Components common to all ensemble methods.

  * Initialization
  * forward through experts
  * combine experts
  

Current contents:

1) An implementation of Q-learning which solves the OpenAI gym inverted pendulum task
2) In progress: a mixture of experts implementation of Q-learning to solve the inverted pendulum task
