# Differential privacy and interpretability of machine learning models.
Our project is focused around differentially private machine learning models. Primarily, our objective was to further
our understanding of machine learning models, differential privacy, and the impact that applying differentially private 
gradient optimization techniques have on the performance and interpretatbility of training a model. 

# Defining Privacy
Privacy concerns have become a prevalent issue as machine learning and other data driven analytics become 
more popular. With this increasing use of data, understanding how a model interprets what it has 
learned about the training data it has been trained with is a key question to developing machine 
learning models that make considerations of privacy conerns. But what does it mean for a model 
to make considerations of a dataset's privacy? How would one intuitively understand if a model 
offers a level of differential privacy? One higher level way of thinking about it, is to assume 
the model is not making considerations of privacy, if an observer can reasonably gain insight 
about some data element the training set, through the model's output. However this is not always 
enough, to ensure privacy guarantees. From this need several projects have attempted to address 
these concerns. Opacus is a framework that implements Differentially Private Stochastic Gradient 
Descent, which is a technique for training a model such that it not only considers privacy 
concerns of the data it is learning from, it also is able to provide a measure such that 
a model's measure of privacy can be quantified.


# The two objectives:
This repo contains two main components spread across two Jupyter Notebooks, and python modules providing 
some utility functionality for the notebooks. 

## Performance Constraints Training with Differential Privacy
One notebook Mileston4.ipynb explores the performance concerns that arise
when training a model using DP-SDG, 

## Model Output Examination, Privacy Guarantee
The second notebook Visualization.ipynb, walks through differential privacy, and demonstrate 

## Instructions
- Make sure you have all the dependencies listed in `requirements.txt` installed using `pip`. 
        pip install -r requirements.txt 
    should install appropriate dependencies.
- Start tensorboard by running the following command in the project's root directory:
        tensorboard --logdir=./runs
- TensorBoard should now be accessible at:
        http://localhost:6006

- Once you have started tensorboard, you can use the accompanying notebook to examine the model's
training progress, and model visualizations.
- You will need to start a jupyter notebook server in the project root directory using:
        `jupyter notebook`
- This will start a jupyter notebook session you can access at:
        http://localhost:8888
- The two notebooks are named as follows:
    - `Milestone4.ipynb` Contains code to demonstrate and visualize progress of model training,
in order to observe the impacts of differential privacy techniques on model training.
    - `Visualizations.ipynb` Contains code needed to demonstrate some visualization techniques for machine learning models. These serve as a way of demonstrating how model weights can give rise
to differential privacy concerns that exist in the field of machine learning.

