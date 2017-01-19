# Papers 
----
-----

## Batch Normalization 2015
------------
  [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://github.com/saiprabhakar/Papers/blob/master/files/1502.03167v3.pdf)

### Problem: Internal covariance shift

* Distribution of each layer changes as the parameters of its previous layers are updated this makes training slow and needs less learning rate.

### Solution: BN

* Makes normalization a part of architecture

* Lowers training time. Higher learning rate can be used. Sometime eliminates the need for Dropout

* Fights vannishing and exploding gradients because of normalization (scale of weights doesnt matter)

### Covariance shift 

* When the distribution of input to a learning system chages (whole system as a whole)

* Usually handled by domain adaptation

* ICS is an extenstion when part of it changes

### Notes

* Training is faster in general if the inputs are whitened (line tras to have 0  mean and sigma = 1 and decorrelated)

* Ignoring the BN during gradient descent is bad idea since it leads to explosion of parameters like bias terms

* There were previous less successfull attemps on this idea

* Simply normalizing layers can constrain them. For example normalizing simoid layer would constrain them to the linear portion their nonlinearity. **So they introduced additional parameter (gamma and beta) to make sure the normalization can represent identity transformation.**

* Incase of Conv layer, we need to follow conv property. Different elements of the same feature map, at diffrent locations are notmalized the same way. We learn gamma and beta per feature map and not per activation.

* Applied BN before nonlinearity, where as standardization (2013) was after then nonlinearity.

### Further possible extensions

* Similarity between BN and standardization

* Extension to RNNs (where vanishing and exploding gradients are more severe)

* More theoritical analysis

* Application to domain adaptation

----
-----
## ModDrop 2014
-----

  [ModDrop: adaptive multi-modal gesture recognition](https://arxiv.org/abs/1501.00102)

### Notes

* Modalities (as a whole) are dropped with a probability during training
* They are trained without fusing during pretraining and are not droped at this point
* cross modal connections are introduced at training stage
* Dropping is only at the input layer
* Rescaling?

### Notes from citation

* Out Performs Dropout
* Combining with dropout gives higher performance

----
-----

## ModOut 2016
---

  [Modout: Learning to Fuse Modalities via Stochastic Regularization](http://openjournals.uwaterloo.ca/index.php/vsl/article/view/103)

### Notes

* Learns the probability of fusing modalities
* Connection between modalities btwn adjacent layers are dropped with a probability
* Dropping can be done in any layer
* No. of extra parameters to learn are small Nm*(Nm-1), where Nm is the number of modalities
* Very similar to **Blockout**
