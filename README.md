## [Deep EHR: A Survey of Recent Advances on Deep Learning Techniques for Electronic Health Record (EHR) Analysis](https://arxiv.org/pdf/1706.03446.pdf) (UofF 2017)

types of EHRs:
* basic EHRs without clinical notes
* basic EHRs with clinical notes
* comprehensice systems

EHR Information Extraction (IE): Extracting information from clinical notes which is unstructed
* Single Concept Extraction: Tag each words into categories
  - RNNs out perform CRFs
* Temporal Event Extraction: Notion of time to extracted EHR concepts
  - RNNs perform okish
* Relation Extraction: Relation between EHR concepts
  - Autoencoder generated inputs to CRFs -> sofart
* Abbreviation Expansion: 
  - custom word embedding using medical articles

EHR Representation Learning: mapping codes for medical concepts
* concept representation: learn EHR concept vetors to capture similarities and clusters in medical clusters using sparse medical codes
  - Embeddings
  - Latent Encoding: AEs, RBMs are better at encoding
* Patient Representation: Getting vector representations of patients
  - embeddings or AEs (on ordered sequences of codes)
    - can be used to predict unplanned visits
    - [Med2Vec](#med2vec)
  - LDAs on clinical notes
  - Embeddings
    - Sentence embedding on clinical notes
    - patient temporal diagnosis (better than the intervention codes)
    - intervention codes
* Outcome Prediction:
  - Static or one-time prediction: using data from single encounter
    - classification using embeddings (best)
    - embeddings learned from full EHR data is better than using diagnostic codes
  - temporal outcome prediction: over a period of time
    - CNN on temporal matrices of medical codes
    - LSTMs (target replication and auxiliary targets :punch: paperref49)
    - Predicting Doctor's behavior
    - Postoperative responses
* Computational Phenotyping: better disease descriptions from data
  - New phenotype discovery
    - AEs on raw data
    - CNN and patient representation
  - improving existing definitions: 
    - using supervise learning
    - LSTMs
* Clinical Data De-identification: removing personal data from clinical data in EHR
  - LSTM with character level + word level embeddings
  - ensemble of RBMs
  - NERs
  
Interpretability: clinical domain transparency is important  
linear models still dominate clinical informatics  
lack of interpretability is a imp limitation
* Maximum activation: in image processing
* Constraints: 
  - [Med2Vec](#med2vec)  
  - non-negativity on learned code representions then examining k most significant elements
  - non-neg on weights
  - structural smoothness by using hierachial features :punch: paperref23
* Qualitative clustering: 
  - visualization using t-SNE
* Mimic Learning
  - Train a new model using data and deep net

Summary and future work:
* Data hetrogenity: (text, codes, billing info, demographics)
* Irregular measure: varying time scale
* Clinical text: difficult to use
  - Extracting structure
    - [Medical entity identificaion](#medent), 
    - [medical event det](#medevent),
    - paperref34 Ner in clinical text :punch:
    - [Clinical temporal information extraction](#clintemp)
    - [Clinical relation extraction](#clinrel)
    - [Learning embeddings for clinical abbr expansion](#clinabbr)
* patient de-identification
* Benchmarks: Different dataset used in diff works
* Interpretability

:punch: incremental training prcedure (adding neurons to the final layer)

---
## [“Exploiting Task-Oriented Resources to Learn Word Embeddings for Clinical Abbreviation Expansion](https://nlp.cs.rpi.edu/paper/bionlp15.pdf) RPI, 2015
#### clinabbr

Abbr are ambiguous especially in intensive care  
embedding for abbr and their expansion should have similar embedding

:trollface:

---
## [Brundlefly at SemEval-2016 Task 12: Recurrent Neural Networks vs. Joint Inference for Clinical Temporal Information Extraction](https://arxiv.org/pdf/1606.01433.pdf) Stanford 2016
#### clintemp

phase 1: text span of time and event expression in clinical notes  
  - joint inference-based approach outperform naive RNN
  - timeline ordering of all events in a document
  - Using DeepDive framework (zhang 2015) :boom:
phase 2: relation btw an event and its parent document creation time
  - combination of data canonization and distance supervision rules  
:trollface:

---
## [“Clinical Relation Extraction with Deep Learning](https://pdfs.semanticscholar.org/7fac/52a9b0f96fcee6972cc6ac4687068442aee8.pdf) Harbin China 2016
#### clinrel

Relations between medical concepts  
Concept identification (NER) -> relation classification using CRFs

Relations:
* problem-treatment
  - treatment imporves problem
  - ...
* problem-test
  - test reveals problem
  - ...
* problem-problem
  - problem incdicates problem 
  - ...

:trollface:

---
## [Structured prediction models for RNN based sequence labeling in clinical text](https://arxiv.org/abs/1608.00612) UofM, 2016
#### medent

Extraction of medical entities such as medication, indication, and side-effects from EHR narratives  
RNN based feature extractors  
Model CRF pairwise potentials using NN

Usually ppl use CRFs, HMMs, NN for information extraction from unstructed text  
Graphical models predict entire label sequence jointly but require hand crafted features for good results  
NN can find patterns but predict word label in isolation

Huang et al. 2015 combined CRFs and NN for NERs :punch: :boom: (not good on exact phrase labelling)

Challenges: 
* extraction of exact medical term is important
* Long tail stuffs are also important
* Long term dependencies between text terms

* m1: embedding + Bi-LSTM + softmax (baseline)
* m2: embedding + Bi-LSTM -> CRF  
unary potential (lstm output) + binary potential (matrix)
using matrix is bad (long tail)   
* m3: embedding + Bi-LSTM -> CRF (pairwise modelling) 
1D CNN (2\*1) for modelling binary potential  
* m4: Approximate skip-chain CRF  
skip-chain to get long term dep :punch: sutton & mccallum 2006  
Exact inference is intractable -> approx sol  
every iteration of grad des need multiple Belief propagation loop iteration -> costly  
lin et al, 2015 :punch: solves it... this paper uses a variation of lin's work  
:boom: read more  

Labels
* Medical event
  - drug name
  - disease
  - ... 
* Attributes
  - severity
  - routine
  - ...
  
Skip-Chain CRF (0.8210 for strict and 0.8632 for relaxed evaluation

---
## [Bidirectional RNN for Medical Event Detection in Electronic Health Records](http://www.aclweb.org/anthology/N16-1056) UofM, 2016
#### medevent

SofArt uses CRFs  
Obj: RNNs outperform CRFs for medication, diagnosis and adverse drug event

EHRs are noisy, have incomplete sentences/phrases, and irregular use of language, have lots of abber ...   
graphical model does not use long term informations  

:trollface: :boom: 

---
## [Multi-layer Representation Learning for Medical Concepts](https://arxiv.org/abs/1602.05568)
#### Med2Vec

:boom:

---

## [Async stoc gradient descent](http://www.ijcai.org/Proceedings/16/Papers/265.pdf)
#### asyc
:boom:

---


## [Listen Attend and Spell (2015) Google Brain](https://arxiv.org/abs/1508.01211)

10.3, 14.5% WER compared to 8% state of the art [cldnn-hmm](#cldnn-hmm)

Dataet: Google voice search tasl

* Listner(PBLSTM) -> Attention (MLP + Softmax) -> Speller (MLP + Softmax) -> Characters
* No conditional independence assumption like CTC 
* No concept of phonemes
* Extra noise during training and testing
* Sampling trick for training PBLSTM
* Beam search(no dictionary was used 'cause it wasnt too useful) + LM based rescoring (very effective) 
[Know about rescoring](#rescoring-1)
* Async stoc gradient descent [aync](#asyc)

### Suggestions
* Convolution filters can improve the results [TODO](#20paper) :punch:
* Bad on short and long utterances [TODO](#15paper) :punch:

---

## [Connectionist Temporal Classification](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf) (2006) (Swiz+germany)

* RNN -> Phonemes -> prefix search decoding
* No conditional independence assumption like DNN-HMMs (Bengio 1999) :punch:

### Contributions:
* Efficient decoding
* Good training algorithm

:trollface: readmore

---

## [CLDNN-HMM](https://www.semanticscholar.org/paper/Convolutional-Long-Short-Term-Memory-fully-connect-Sainath-Vinyals/56f99610f7b144f55a511da21b74291ce11f9daf)
#### cldnn-hmm
:punch:

---
## [EFFICIENT LATTICE RESCORING USING RECURRENT NEURAL NETWORK LANGUAGE MODELS](http://mi.eng.cam.ac.uk/~mjfg/xl207_ICASSP14a.pdf) (cambridge) (2014)
#### rescoring-1

Rescoring methods:
* n-gram style clusteing of history contexts
  - data sparsity issues
  - large context leads to exponential size growth
* distance in hidden history vectors
  - [RNNLM](#rnnlm) & and FFNNLM :punch: readmore

:trollface: readmore

---

## [Prefix Tree based N-best list Re-scoring for Recurrent Neural Network Language Model used in Speech Recognition System](https://pdfs.semanticscholar.org/5f59/1b7043deefbc3f3af19b6efeb97c2a80d27c.pdf) China 2013 
#### RNNLM

RNNLM is time consuming so is used to resore only some of the n-best list

* obj: Speed up RNNLM when used to rerandk a large n-best list
* Prefix Tree based N-best list rescoring (PTNR)
  - avoid redundant computations
  - [Bunch Mode](#bunch-mode)

related:
* FFLMs -> faster paper10ref :punch:
* RNN-ME -> RNN on large dataset paper12ref :punch: 
* RNNLM -> First pass decoding by conv Weighted first pass transducer :punch:

PTNR:
* Represent hypothesis in a prefix tree thus all the LM prob for the nodes can be computed in a single forward pass preventing any redundant computation.
* Each node in the tree needs to store only hidden value and its state (if the node is not explored)

#### Bunch Mode
(block operations)
* speeding up training o0f FF-NNLM
* several words are processed at the same time using matrix\*matrix multiplcation rather than vector\*matrix multiplication
* Uses BLAS library
* 10 times faster training with slight loss of perplexity

PTRN + Bunch Mode slightly complicated using class-based RNNLM #paper11ref :punch:

ASR here uses two-pass search strategy:
* first pass: decoder uses weak LM (3-gram lm) to generate multiple recog hypothesis -> word lattice
* word lattice -> n-best hypothesis
* second pass: powerful LM used to re-score hypothesis -> best hypothesis

Acoustic modelinhg and feature settings as done in :punch: paperref25
setting training param in :punch: paperref28
Rescoring using linear combination of 4-gm lm and rnnlm -> 1.2% WER reduction using 100-best list
Much faster than standard rescoring approach. Speed up increases with n in n-best list

---

## Batch Normalization 2015
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

---
## ModDrop 2014

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

---
## ModOut 2016

  [Modout: Learning to Fuse Modalities via Stochastic Regularization](http://openjournals.uwaterloo.ca/index.php/vsl/article/view/103)
### Notes

* Learns the probability of fusing modalities
* Connection between modalities btwn adjacent layers are dropped with a probability
* Dropping can be done in any layer
* No. of extra parameters to learn are small Nm x (Nm-1), where Nm is the number of modalities
* Very similar to **Blockout**

---

## ICNN 2016
  [Input Convex Neural Networks](https://arxiv.org/abs/1609.07152)
### Notes

* Under certain condition of weights and nonlinearity a neural network will be convex in certain inputs/outputs, so we can efficiently optimize over those inputs/outputs while keeping others fixed

#### Fully input convex neural networks

* Convex interms of all the inputs
* Conditions: non-negative weights (restricts the power of the network) and non-decreasing non-linearities

#### Partially input convex neural networks

* Convex in certain inputs and not convex in others
* PICNN with k layers can represent and FICNN with k layers and any feedforward net with k layers

#### Inference

* Inference wrt to the convex variable are not done in a single pass as in feed forward network case
* Inference can be found by using optimization techniques like LP, approximate inference etc

#### Learning

* In case of q learning the fuction fitting is automatically taken care of as the goal is to fit the bellman equation
* For fitting some target output they use techniques like max-margin etc

#### Results

* Preliminary results for DRL shows faster convergence comparision to DDPG and NAF
* Can complete face (fix some inputs while solve for others)
* Classification task need more investigation

---
