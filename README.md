Papers
=================


Table of contents
=================

<!--ts-->
   * [Top](#papers)
   * [Table of contents](#table-of-contents)
   * [Speech](#speech)
      * [General](#general)
      * [Cleaning Noisy Speech Dataset](#cleaning-noisy-speech-dataset)
      * [End-to-End](#end-to-end)
<!--te-->



Speech
=================

## General
<details><summary> ... </summary>

## [Tree-Based State Tying for High Accuracy Modelling](www.aclweb.org/anthology/H94-1062) (Cambridge, 1994)

Data insufficiency occurs when using cross-word triphones. To solve this ppl use state-tying. \
Rather than using a data-driven clustering approach the work suggests tree-based state tying which can be used for unseen phones as well.

Process of building a tied-state HMM system:
- 3 state l-r monophone model with single guassian output density is trained
- using the same state output distribution a CD triphone model is trained with new and tied transition matrix.
- for all triphones from the same monophone the corresponding states are clustered and thus the parameters are tied
- number of mixture componenets in each state are incerased untill a stopping creteria

Tree-based clustering:
- for all triphones from the same monophone every state is clustered using a decision tree. 
- tree is based on increase in log-likelihood
- The questions vary from lingustics properties of the left and right context phones to set of phones

## [Subphonetic Modeling for Speech Recognition](https://core.ac.uk/download/pdf/22876656.pdf) (CMU, 1992)

Advocates for state-level (output-distribution level) parameter sharing instead of model-level and the use of state-dependent senones. \
Senones alow parameter sharing/reduction, pronunciation optimization and new word learning 

After generating all the word HMM models, cluster the senons and generate the codebook. Then replace the senones with nearest ones in the codebook. \
The clustering start by assuming all the data points are seperate clusters then a pair are merged if they are similar (If the entropy increase is small after merging then two distributions are similar). 

Explores 3, 5, 7 state triphone models and finds than 5 is the most optimal one 

## [A NOVEL LOSS FUNCTION FOR THE OVERALL RISK CRITERION BASED DISCRIMINATIVE TRAINING OF HMM](https://pdfs.semanticscholar.org/de8c/eb72bf54293959813c101c4f7ce54fbd3a20.pdf) (University of Maribor, 2000)

MBR training of ASR systems \
MBR minimizes expected loss

aim to directly max word recog accuraccy on training data

generally MAP is used for ASR argmax w P(w|o) = argmax_w p(o|w) * p(w) \
p(o|w) is AM, with HMM it becomes p(o_r | theta_r) for which MLE give best theroritically. practically they use MMI or MCEE (Min classification error estimation). \
Modification of MCEE is ORCE overall risk creterion estimation. 


In this paper they extend ORCE objective to continuous speech recognition and use a non-symmetrical loss using the number of I, S, D in WER calculation instead of 1/0 loss.

experiments on TIMIT dataset on HMM.


## [Hypothesis Spaces For Minimum Bayes Risk Training In Large Vocabulary Speech Recognition](https://pdfs.semanticscholar.org/0687/573a482d84385ddd55e708e240f3e303fab9.pdf) (University of Sheffield, 2006)

State-level MBR training

MBR training good for large vocab HMMs, implementation needs hypothesis space and loss fn. \
MMI is better than MLE training of AM (HMMs) \

minimum phone error can be interpreted as MBR when phone sequence forms hypothesis space -> better than MMI \

Lattice-based MBR -> constraining the search space to only those alignments specified by the lattice \
to do this we need l(w_reference, arc_i) is  difficult.

a solution explored here is comming up with Frame Error Rate FER.

## [CLDNN-HMM](https://www.semanticscholar.org/paper/Convolutional-Long-Short-Term-Memory-fully-connect-Sainath-Vinyals/56f99610f7b144f55a511da21b74291ce11f9daf)
#### cldnn-hmm
:punch:


## [EFFICIENT LATTICE RESCORING USING RECURRENT NEURAL NETWORK LANGUAGE MODELS](http://mi.eng.cam.ac.uk/~mjfg/xl207_ICASSP14a.pdf) (cambridge) (2014)
#### rescoring-1

Rescoring methods:
* n-gram style clusteing of history contexts
  - data sparsity issues
  - large context leads to exponential size growth
* distance in hidden history vectors
  - [RNNLM](#rnnlm) & and FFNNLM :punch: readmore

:trollface: readmore



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

</details>

## Cleaning Noisy Speech Dataset
<details><summary>CLICK ME</summary>

## [A RECURSIVE ALGORITHM FOR THE FORCED ALIGNMENT OF VERY LONG AUDIO SEGMENTS](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.649.6346&rep=rep1&type=pdf) (Cambridge, 1998)

A recursive alignment with ASR + restricting dictionary and LM \
Introduces the concept of anchors with island of confidences \
Dictionary (phonetic) is built using CMU public domain dictionary plus an algo \
A simple LM with word pair and triple model for the transcript specifically

Number of consecutive word matches needed for confidence islands is in the early point of the recursion to reduce the possibility of error in the early stage as it can affect the entire pipeline.

Used for indexing the audio using the words in the audio file. Error of 2 sec is tolerated.

General discussion:
- Viterbi is time consuming for long audio and if it gets an error it will make it completely wrong.
- increasing the beam search helps viterbi but it only scales for short audio

## [A SYSTEM FOR AUTOMATIC ALIGNMENT OF BROADCAST MEDIA CAPTIONS USING WEIGHTED FINITE-STATE TRANSDUCERS](https://homepages.inf.ed.ac.uk/srenals/pb-align-asru2015.pdf) (univ of Edinburgh, 2015)

Two pass algorithm for align speech to text

General methods:
- iterative approach to identify increasingly reliable confidence islands
- using a biased language model plus may be a background LM + DP alignment
- For low resource cases, train AM from the alignment audio and adapt it to aligned ones
- weak constraints on AM decoding
- using dynamic time warping using TTS systems
- Strong constraints on decoding using factor automaton which alows only contiguous strings from the training text (good one)

ALgo:
- First pass: use WFST based decoder to get a transducer with some modifications to allow insertions and null words
  - this alows to constraint the words but not the order (efficient)
  - but is bad in dealing with deletions, i.e. words present in text but not in audio
- Second pass: (not clear) resegment the data + extending and joining segments where there were missing words, generate factor transducer. Output from this is considered as the final output without any further text-to-text alignment.

AM training: 
- after the alignment the AM was trained using only data with word-level Matching Error rate less than 40%
- during the starting of the two pass AM was trained using MER less than 10%

Done on MGB challenge data

</details>

## End-to-End
<details><summary>CLICK ME</summary>

## [Towards End-to-End Speech Recognition with Recurrent Neural Networks](http://proceedings.mlr.press/v32/graves14.pdf) (graves, 2015) 

Modified CTC objective function. Instead of MLE, this version is trained by directly optimizing WER.
Done using samples to approximate gradients of the expected loss function (WER).

No lattice level loss here.




## [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (Graves, 2006)

First version of CTC.

b + L -> L'
prefix search decoding (works fast if the peaks at the output are around mode)
insert blanks at every pair of labels at the network output
customized forward-backward algo

MLE training of the network objective fn = - sum(x,z)_in_S ln(p(z|x))

TIMIT data + BLSTM
higher level of training noise is optimal here (guassian noise added at the input to improve generalization)

Doesnt model inter-label dependencies explicitly
Gives approximate segmentation not exact

## [Optimizing expected word error rate via sampling for speech recognition](https://arxiv.org/abs/1706.02776) (Google, 2017)

Define word-level Edit-based MBR (EMBR) on lattice generated during SMBR.\
they do it by using monte-carlo samples from the lattice to approximate the gradient of the loss function which is in the form of an expectation.\
Similar to Reinforce.

Gradient has the form (average loss - loss of state i) so cannot be used during the starting phase of the training.

Generalized version of sample based loss derived in the CTC,2015 paper. Where the CTC paper doesnt use lattice level loss function.




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


</details>


----
----

## [Generative Adversarial Network for Abstractive Text Summarization.](https://arxiv.org/pdf/1609.05473.pdf) (china, 2017)

G: attention + pointer generator network

1. Train G till -- more realistic and high quality summaries
2. Train and fix D
3. Train G

D: CNN + max-overtime pooling

G: loss = policy gradient + maximum likliehood \
pg: REINFORCE algorithm and take the estimated probability of being human generated by the discriminator D as the reward -- Since the Discriminator can only score a fully generated sequence, following (Yu et al. 2016), we use Monte Carlo Tree Search (MCTS) to evaluate the reward for an intermediate state :punch:

D: usuall loss

Added Trigram avoidance and quotation weight alleviation


## [Toward Controlled Generation of Text](https://arxiv.org/pdf/1703.00955.pdf) (CMU, 2018)
"Very few recent attempts of
using VAEs (Bowman et al., 2015; Tang et al., 2016) and
GANs (Yu et al., 2017; Zhang et al., 2016) have been made
to investigate generic text generation, while their generated
text is largely randomized and uncontrollable. -- we focus on generating realistic sentences,
whose attributes can be controlled by learning disentangled
latent representations"

Challenges:
- "A first challenge comes from the discrete nature of text
samples. The resulting non-differentiability hinders the use
of global discriminators that assess generated samples and
back-propagate gradients to guide the optimization of generators
in a holistic manner, as shown to be highly effective
in continuous image generation and representation modeling"
- "Another challenge for controllable generation relates to
learning disentangled latent representations. Interpretability
expects each part of the latent representation to govern
and only focus on one aspect of the samples. "

Contributions:
- "We base our generator on VAEs in combination
with holistic discriminators of attributes for effective imposition
of structures on the latent code."
- "End-to-end optimization
is enabled with differentiable softmax approximation"
- "The probabilistic encoder of VAE also functions
as an additional discriminator to capture variations
of implicitly modeled aspects, and guide the generator to
avoid entanglement during attribute code manipulation"
- "our method enables to use separate
datasets, one with annotated sentiment and the other with
tense labels"

## [GRAM: Graph-based Attention Model for Healthcare Representation Learning](https://arxiv.org/abs/1611.07012) (GaTech, 2016)
supplements electronic health records (EHR) with hierarchical information inherent to medical ontologies
GRAM represents a medical concept as a combination of its ancestors in the ontology via an attention mechanism.
Testing on rare disease prediction and heart failure.

medical codes as DAG, then use embedding for all leaf nodes (nodes with meaning full concepts) and visit one-hot embedding to generate a vist vector (process incoporates attention mechanism). Finally use the vist vector to predict stuff.

## [Deep Learning Approaches for Online Speaker Diarization](http://web.stanford.edu/class/cs224s/reports/Chaitanya_Asawa.pdf) (2012)

## [SPEAKER DIARIZATION WITH LSTM](https://arxiv.org/pdf/1710.10468.pdf) (google, 2018)

usually ppl use i-vector based audio embedding tech
paper explores d-vector based approach (nn based audio embedding)

usuall system:
1. speech segmentation (short speech sections of same speaker)
2. audio embedding (MFCCs, speaker factors, i-vectors)
3. clustering
2. resegmentation (refining)

recently nn based embedding's use in speaker verification outperform i-v tech (text dependent)

this paper use a lstm-based approach with non-parametric spectral clustering algo
paper also aguments spectral clustering algo :punch:
paper somewhere uses Voice Activity Detector (VAD) to find speech seg from audio

Clustering:
- online (labels for each segments as soon as they are available)
- offline (after all segments are available)

Challenges:
- non-guassian dist (imp assumption in k-means clustering)
- cluster imbalance (one speaker might speak all the time)
- Hierarchial structure (speakers in diff category, some are easy to differentiate)

evaluated using DER (diarization error rate)


## [Summarization of Spoken Language—Challenges, Methods, and Prospects](www.cs.cmu.edu/~./zechner/ezine.ps) (CMU 2002)

Types:
- extracts vs abstracts
- indicative vs informative
- generic vs query-driven
- single vs multi-document
- background vs just-the-news
- single vs multiple topic
- single vs multi-speaker
- text vs multi-modal
- selecting sentences/clauses vs condensing within sentences

challenges:
- disfluencies
- identifying units
- cross-speaker coherence and distributed information
- speech recognition errors

prosody-based emphasis detection :punch:


## [Natural Language Processing with Small Feed-Forward Networks](https://arxiv.org/pdf/1708.00214v1.pdf) (google 2017)

Shows that small shallow ffNN can achieve good results
Uses character embedding rather than word

**Explores:**  
Quantization  
Bloom Mapped word clusters  :punch:  
Selected features: character bigram features :punch:  
Pipelining(Using an auxiliary task)  

**For Diff NLP tasks**  
Language Identification  
POS tagging  
Word Segmentation  
Preordering    

## [Revealing the Structure of Medical Dictations with Conditional Random Fields](http://www.aclweb.org/anthology/D08-1001) (2008, medical univ vienna) [Identifying Segment Topics in Medical Dictations](http://www.aclweb.org/anthology/W09-0503) (2009, medical univ vienna)

Formatting the dictations considering structure and formating guidelines

related to:
* Linear text segmentation :punch: Lamprier 2008
* text classification for section detection
* dynamic programing methods for formating :punch: Matsuov 2003

mapping annotated data to dicatations need care for repeted words, punctuation, recog errors and meta instructions
hand coded features for each time step

Classifiers:
* CRFs based multiple label chains: BIO tagging without Outside label
  - Better accuraccy but high training time
* SVM based multi class
  - Lower accuraccy wuth small training time

Can Combine both the approaches by using results of SVM as input to CRFs


## [Deep EHR: A Survey of Recent Advances on Deep Learning Techniques for Electronic Health Record (EHR) Analysis](https://arxiv.org/pdf/1706.03446.pdf) (UofF 2017)

types of EHRs:
* basic EHRs without clinical notes
* basic EHRs with clinical notes
* comprehensice systems
* tagging using HMM (generative), CRFs (discriminative) and multilable classification

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
 
rel Event:
* crf for taggind and svm for recog of event attr  
rel TIMEX:
* rule based + ML  
rel TLink:
* crf, ml
rel with NN are the best
  
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
## [Structured prediction models for RNN based sequence labeling in clinical text](https://arxiv.org/abs/1608.00612) UofM, Aug 2016
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

** Private dataset?

models:
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
  
Skip-Chain CRF Precision - 0.8210 for strict and 0.8632 for relaxed evaluation

Sentence Level RNN

---
## [Bidirectional RNN for Medical Event Detection in Electronic Health Records](http://www.aclweb.org/anthology/N16-1056) UofM, June 2016
#### medevent

SofArt uses CRFs  
Obj: RNNs outperform CRFs for medication, diagnosis and adverse drug event

EHRs are noisy, have incomplete sentences/phrases, and irregular use of language, have lots of abber ...   
graphical model does not use long term informations  

** good related work ** :boom:  
** Private dataset  ?   

Labels:
* Medication
  - Drugname, Dosage, Frequency, Duration and Route
* Disease
  - ADE, Indication, Other SSD, Severity

methods:
* emb + BiLSTM
* emb + GRU
* CRF-nocontext (BIO tagging scheme :punch: :boom:)
* CRF-context(context= 2 BoW rep of sentence) (BIO tagging scheme)

Both sentence and document level RNN

RNN > CRF
Best (GRU-document) recall (0.8126), precision (0.7938) and Fscore (0.8031)

---
## [Multi-layer Representation Learning for Medical Concepts](https://arxiv.org/abs/1602.05568) Feb 2016, GaTech + children healthcare atlanta
#### Med2Vec

diagnosis, procedure, and medication codes  
EHR database with >3m visits  
** What does other papers use ?  

Other baselines:  
* GloVe
  - uses global co-occurence matrix (sparse)
  - less computationally demading than skip gram
  - uses weighting function thus but might require large tunning effort
* stacked AE
* Skip gram :punch: paperref 25 (skipgram, 2013 > word2vec (2013))
  - goal is to find a rep for word wt such that we can predict the nearby words
  - Skip-gram tries to maximize the softmax probability of the inner product of the center word’s vector and its context word’s vectors
  - ppl used hierarchial sofmax and negatice sampling to get faster training
  
health care hand eng feature rep paperref 32 16 36 :punch:  


:boom:

---

## [Async stoc gradient descent](http://www.ijcai.org/Proceedings/16/Papers/265.pdf)
#### asyc
:boom:

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
