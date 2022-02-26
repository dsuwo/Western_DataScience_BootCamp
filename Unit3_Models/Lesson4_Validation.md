---
title: "Validation"
---

# Learning Outcomes

- [ ] Understand when simple models are preferred over complex models (aka the overfitting problem) 
- [ ] Determine how well your model works on out-of-sample data

# Model Validation

Can you ever have a model that fits your data perfectly?

**Yes!** But that's not actually useful. 
Consider the figure below. The line hits every data point and our model will have no error, but what does a model like this *tell* us? 

<img src="figs/thisplot.png" width=480px>


```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)
x = np.random.uniform(0, 10, size = 40)
x = np.sort(x) # Sort so that the lines look nices
y = np.random.normal(loc = 0, scale = 3, size = x.size)

plt.plot(x, y) # Plot the lines
plt.plot(x, y, "o") # Plot orange points
plt.show()
```

When a model is too closely fitted to limited data points, we say that the model is guilty of overfitting.
- Overfitting is a bad thing.
    - When a model is overfitting, it performs poorly in the test set 
        - This means we can't use it effectively to make predictions!
One way to avoid overfitting is to use simpler (less complex) models. 

# K-Fold Cross-Validation 

Cross-validation is a resampling method for validating our models on limited data samples. 

In this approach, the dataset is shuffled randomly, then partitioned into K equal sized sub-samples.
*(K refers to the number of groups that a given data sample is to be split into.)*

For each unique group:
- Take the first fold as test data set.
- Take the remaining (K-1) as training set. 
- Fit the model on the training data.
- Use the model fit to the training data to find predictions for test data.
    - If the point x = 5 was in the test set, then use the linear model fit to the training set to find out the height of the line at x = 5. 
- Find out how far off the predictions are from the truth.
    - Often using the average of the squared distance between the predicted values and the actual values, called the mean squared error (MSE). The MSE will have squared units, so the Root MSE (RMSE) is often calculated instead.

The average of the values computed in the loop will be the performance measure reported by K-fold cross-validation.

<img src="figs/cross_validation.png" width=480px>

<sup> source: http://ethen8181.github.io/machine-learning/model_selection/model_selection.html </sup>

## Choosing the Number of Folds

There isn't a clear best choice for the number of folds. 5-10 folds is pretty common, but if you have large data then you can have plenty of folds and still make each fold have a meaningful amount of data in it. The only real limit is how long you're willing to wait, since the model must be re-calculated k times. Thus in general for large datasets K-fold is not the recommended method. 

### Extreme Edge: All of the Folds!

A special case of K-fold cross validation is Leave-One-Out Cross Validation (LOOCV, usually just LOO). 
- If your data have *n* observations, the LOO is the same as having *n* folds, each with one data point in it.  

For linear models, there is mathematical theory to find the RMSE *without* re-calculating the model *n* times. This is based on the **hat matrix**, but this ends up being a lot of linear algebra and is outside the scope of this course. For other models, the computational cost can be too much and you should have bigger folds.

# Hold-Out Sets

When doing cross-validation, you will likely end up trying many variations of the same model (model building is, in general, an iterative process). 
- In doing so, you're creeping back up to the problem of overfitting... 
    - You're finding the best model for those particular folds!

Hold-out sets is a kind of extension of K-fold cross validation. Essentially, one of the folds is left out *completely*. The algorithm does not *ever* use it. 

Note: The hold out set does *not* need to be connected to the K-fold CV at all. It is common to leave 20-30% of the data as a holdout set, then do however many folds you want in the remaining data; there's no restriction that each fold must be the same size as the hold-out set. 

Toward the end of the project, once you have a couple models that you think are very likely to be your final models, only then can you get your hold out set out again. Use your models to predict the hold-out data, then go with the one that has the smallest prediction error.

This process guarantees that the final data that your model is evaluated on was *not* used in any step of the modelling process. If done right, this almost completely removes the chance of overfitting. Or, if all your models are still overfitting after K-fold CV, it will show you by comparison of the hold-out RMSE to the K-fold RMSE.

# Unit 3, Lesson 4 Wrap-Up

In this lesson, we:

- [x] Discussed when simple models are preferred over complex models (aka the overfitting problem) 
- [x] Outlined techniques for determining how well your model works on out-of-sample data

Let's just take a minute to marvel at validation as a technique. 
Instead of just trying to fit the best model to the data, we take an extra step to make sure that the model isn't *too* perfect. 
- This is a fantastic way to make sure our model fits to *other* data, not just the data that we have.

Of course, no modelling technique could ever protect us against a bad sample, so this isn't complete magic.


# See you in Unit 3, Lesson 5! 
