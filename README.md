Papers
=================


## Natural Language Processing


<details><summary> General </summary>

[Natural Language Processing with Small Feed-Forward Networks](https://arxiv.org/pdf/1708.00214v1.pdf) (google 2017)

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
</details>

<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->


<details><summary> Dialog Understanding </summary>
	
Dialog dataset:

- DSTC:
	- https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/
	- both human-human/computer dialogs (microsoft, 2017?)
	- with some disfluencies
- Coached Conversational Preference Elicitation (CCPE) (google, 2019):
	- https://ai.googleblog.com/2019/09/announcing-two-new-natural-language.html?m=1
	- human - human (personal assistant)
	- conversations about people’s movie preferences
	- include errors and disfluencies
	- collected using wizard of oz method
- Taskmaster-1 English dialog datasets (google, 2019):
	-  https://ai.googleblog.com/2019/09/announcing-two-new-natural-language.html?m=1
	- 7.7k written “self-dialog” entries and ~5.5k 2-person, spoken dialogs
	- human - human (personal assistant)
	- conversations about people’s movie preferences, ride ordering ...
	- include errors and disfluencies
	- collected using wizard of oz method
	- also have single human conversation (for managing cost, diversity, volume)
	
[Modelling and Detecting Decisions in Multi-party Dialogue](https://www.aclweb.org/anthology/W08-0125) (stanford 2008)

Related work:
- action words detection (Purver et al., 2007)
- detecting decisions Hsueh and Moore (2007b)

Propose a hierachial approach to detect subclasses and combine them
"Our
scheme distinguishes among three main decision dialogue act (DDA) classes: issue (I), resolution (R),
and agreement (A). Class R is further subdivided into
resolution proposal (RP) and resolution restatement
(RR). "

1 ."We first train one independent sub-classifier for
the identification of each of our DDA classes,
using features derived from the properties of
the utterances in context"

2. "To detect decision sub-dialogues, we then train
a super-classifier, whose features are the hypothesized class labels and confidence scores from the sub-classifiers, over a suitable window." "The super-classifier is then able to “correct” the
DDA classes hypothesized by the sub-classifiers on
the basis of richer contextual information: if a DA is
classified as positive by a sub-classifier, but negative
by the super-classifier, then this sub-classification is
“corrected”, i.e. it is changed to negative"

40 meetings from AMI corpus were labelled. Used features like Utterance, Prosodic, DA (AMI dialog acts), Speaker and Context.
Assume we have information on speaker labels, general dialog acts.

</details>

<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->



<details><summary> Language Modelling </summary>

- **Types**
	- Auto Regressive
		- Use multiplicative factorization to model prob.
		- Unidirectional
		- Examples:
			- GPT
			- Transformer-XL
	- Denoising
		- Introduces noise to model prob.
		- Sometimes use conditional independence assumption when having more than one corrupted inputs
		- Examples:
			- Bert
	- AR + Denoising
		- Examples:
			- XLNet

- **Important Ones**
	- ELMO
		- Pros
		- Cons
			- concatenate forward and backward language models in a shallow manner, which is not sufficient for modeling deep interactions between the two directio


[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf) (Allen, 2018)
- The embeddings are learned functions of bi-directional LM
- Coupled LM (forward + backward) objective function is used to capture context-dependent representations
- Exposing all the internal representation of the model makes it flexible in choosing the correct level of representation for the task

Related work:
- ELMo outperforms CoVe which computes contextualized representations using neural machine translation encoder
- ELMo switches between different words ends seamlessly unlike previous work which switch by classify each word into different sense before finding the vector 
 
Working:
- Similar to previous work, we start by context independent representations via CNN over characters.  
- Then we use bi-directional LSTM layers to find the probability of predicting token k given context with softmax
- Task-specific representations are then created using hidden vector representations LSTM layers, these are concatenated with CNN representations to get task enhanced ELMo vectors
- Dropbox, residual connections and weight regulations are used during training

In some cases:
- For the ones with attention in task-RNN, when adding task specific representation to the final output layer of task-RNN helps improve the performance
- Fine tuning bidirectional LM on domain specific data leads to significant drop in perplexity and increase in downstream task performance.

Different downstream tasks prefer different LSTM layer’s output (depending on the nature of the task)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Google 2017)

Model:
- encoder: multi-attention + residual + ff
- decoder: decoder multi-attention + residual + encoder-decoder multiattention + position-wise ff
	- decoder self-attention:
		- modified to prevent attending subsequent positions (Maksing)
		- output embeddigns are offest by one => uses knowledge from output positions <i
- Position-wise FF is more like 1-d conv layer
- softmax: 
	- embedder share weight with pre-softmax layer
- scaled dot product attention
	- Q, K, V are linear projections of hidden state sequences
	- ith attention head (output value) if calculating by again liearly projecting Q, K, V to a lower dimentions space, then calculating attention
	- all the heads are concatenated and projected to get the final value
	- encoder-decoder attention uses Q from decoder and K, V from encoder
- sinusoidal position embedding (better to handle longer input during testing)
	- gives model temporal cues or bias on where to attend
	- added to the input embeedings at the bottom of the encoder-decoder stack (encoder's input)
- dropout: residual dropout and while adding position encoding with embeddings
- label smoothing: helps accuraccy but not perplexity :punch:

Testing:
- sometimes use checkpoint averaging during testing :punch:
- performs well on machine translation and english consistuency parsing 
- better than encoder-decoder and conv models
- base model dim=512 N=6
- large model dim=1024 N=6
- higher/lower number of multi-head is bad
- dropout helps

[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) (Google, CMU, 2019)

Aim:
- solve context fragmentation
- 80% longer context thatn RNNs

Notes:
- cacheing hidden states is analogous to memory networks
- freeze (stop gradient) of previous segment's hidden state at layer n-1 is used along with current segment's hidden state at layer n-1 (concat) in layer n.
	- hence maximum length of the context is propotional to the number of layers = O(NxL)
- using relative positional embedding instead of absolute to incorporate the information about different segments.
	- to do this we add the relative positional embeddings directly to the attention
	- modified with matrix algebra
	
Questions:
- use cached hidden states for both encoder and decoder?


Related work:
- speeding up softmax graves 2016a :punch:
- enridhing ouptut distribution 
- improving regularization and optimization algorithm
- LSTM uses 200 context words on average



</details>


<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->




<details><summary> Relation Extraction </summary>
https://arxiv.org/pdf/1606.09370.pdf
https://github.com/thunlp/NREPapers
</details>



<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->



<details><summary> Adversarial methods </summary>

[ADVERSARIAL TRAINING METHODS FOR SEMI-SUPERVISED TEXT CLASSIFICATION](https://openreview.net/pdf?id=r1X3g2_xl) (2017, Google Brain and Kyoto Univ)

Notes:
- use adversatial sample and virtual adversarial samples to regularize.
- adversatial sample: adding a peturbation r, that max p(y|x+r;theta)
	- making model robust to worst case peturbation at x
	- use linearization to solve for x using gradient descent
- vir. adversatial sample: adding a peturbation r, that max KL [p(.|x;theta) || p(.|x+r;theta)]
	- make the model resistant to perturbations in directions to which it is most sensitive on the current model
	- solve with approximation using back prop.
- applied on word embeddings instead of words since they are discrete


Analysis:
- The embeddinds are able to push the distance between workds like good and bad
- increases the model's performance marginally
- better than applying random noise 
	- random noise on average are orthogaonal to cost which hwere we maximize the cost with peturbations
- Adversarial is better than vir. Adversarial if the input is noisy/small
- vir. Adversarial is better on other cases (observed in a secret work) calculated in an unsupervised way
- combining both improves better than individual increases

Related work:
- "Adversarial and virtual adversarial training resemble some semi-supervised or transductive SVM
approaches (Joachims, 1999; Chapelle & Zien, 2005; Collobert et al., 2006; Belkin et al., 2006) in
that both families of methods push the decision boundary far from training examples (or in the case
of transductive SVMs, test examples). However, adversarial training methods insist on margins on
the input space , while SVMs insist on margins on the feature space defined by the kernel function.
This property allows adversarial training methods to achieve the models with a more flexible function
on the space where the margins are imposed"


</details>
<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->


<details><summary> Question Answering </summary>

[DYNAMIC COATTENTION NETWORKS FOR QUESTION ANSWERING](https://arxiv.org/pdf/1611.01604.pdf) (salesforce research, 2018)

The paper claims that previously proposed single-pass nature models does not recover from local maximas.

Network:
- Dynamic attention Network fuse codependent representation of question and document 
- Dynamic Point decoder iterate over potential answer spans  
- has a big improvement on squad dataset  
 
Implementation:
- The question and passage share the same lstm encoder  
- the question encodings are non-linearly projected to account for variation in the encoding spaces
- Coattention mechanism attends over the question and document simultaneously then finally fuse their attention contexts  
- Coattention was previously proposed in 2016 
- Dynamic decoder is similar to a state machine whose state is maintained by a lstm-based encoder 
- “During each iteration, the decoder updates its state taking into account the coattention encoding corresponding to current estimates of the start and end positions, and produces, via a multilayer neural network, new estimates of the start and end positions” 
- Highway maxout network which is a combination of previously proposed MaxOut networks in 2013 and Highway networks in 2015.
- “The intuition behind using such model is that the QA task consists of multiple question types and document topics. These variations may require different models to estimate the answer span. Maxout provides a simple and effective way to pool across multiple model variations.”

Notes:
- Sentinel vectors are randomly initialized and optimized during training :punch:
- Dropouts where used 
- “There is no notable performance degradation for longer documents and questions contrary to our expectations. “
- “The DCN, like other models, is adept at “when” questions and struggles with the more complex “why” questions.”
- “we observe that when the model is wrong, its mistakes tend to have the correct “answer type” (eg. person for a “who” question, method for a “how” question) and the answer boundaries encapsulate a well-defined phrase.”

Related work:
- Dynamic chunk reader (2016):
	- “A neural reading comprehension model that extracts a set of answer candidates of variable lengths from the document and ranks them to answer the question.”
	- 71.0 % in squad


	
[MACHINE COMPREHENSION USING MATCH-LSTM AND ANSWER POINTER](https://arxiv.org/pdf/1608.07905.pdf) (singapore management univ. 2017)

 64.7 exact match score, word level F1 73.7% with single model on SQuAD dataset

- Match lstm model was originally proposed for textual entailment problem.
- Here we choose the question as the premise and the  passage has the hypothesis.
- After the match lstm, they use pointers via a sequence model or boundary model for the question answering problem

The network consists of three layers
- The first layer is the lstm pre-processing layer. It has two unidirectional lstm models each operating on the passage and the question.
- Match lstm has two unidirectional lstm Networks
	- The forward unidirectional lstm takes the hidden state representation of the passage at position i concatenated with the weighted version of the question as input. During the attention mechanics it uses the hidden State representation of i - 1. In Backward lstm we do similar processing.
	- At the end of the second model, we concatenate the hidden state vectors from the forward and backward lstm Network into a matrix.
- Pointer network:
	- In the sequence model part, we have a special value at p + 1 which indicate the stopping of answer generation  
	- The pointer network has an lstm followed by an attention mechanism. The attention mechanism takes the last hidden state from the lstm, hidden state matrix from the previous model, to generate the probability of position i from the passage to be the answer 
	- The boundary model produces start and the end position of the answer

Loss: the negative log-likelihood function

- used word embeddings from glove to initialize the model (not updated during training)
- Boundary method works better than the sequence method
- During the prediction phase, they limit the size of the span. Using bi-directional network the pre-processing face as well as the answer generator part helps
- They further extend the boundary method by introducing a global search algorithm which looks at all the probabilities for the start and end word and selects the one with the highest product.
- Longer answers are hard to predict 
- Performance trend: When> (what= which = where) > why, because of the diverse possible answers for ‘why’ questions.
- " Note that in the development set and the test set each question has
around three ground truth answers. F1 scores with the best matching answers are used to compute
the average F1 score."

	
[Text Understanding with the Attention Sum Reader Network](https://arxiv.org/pdf/1603.01547.pdf) (IBM Watson, 2016)

Uses CNN/Daily Mail and Children's Book Test. Generate lots of one word QA data from summaries.

Intuitively, our model is structured as follows:
1. We compute a vector embedding of the query.
2. We compute a vector embedding of each individual word in the context of the whole document (contextual embedding).
3. Using a dot product between the question embedding and the contextual embedding of each occurrence of a candidate answer in the document, we select the most likely answer.

Notes:
- Very simple model based on pointers idea. Does not compute a fixed length representation of the document like usual models. They claim blending is bad when there are multiple similar candidates.
- Also accounts for same word occuring multiple times in the input which pointer network does not.
- Log likelihood loss
- Masks named entities with unique tags per example which are randomly shuffled.

Results:
- Performance decreases as the input lenght and the number of candidates increase.
- Performance increase if the correct answer is likely to occur frequenctly. this is because we sum the scores for each occurance.

Related work:
- 2015 Attentive reader: 
	- compute a fixed length embedding of the document
	- computes a joint query m, and doc representation with a non-linear fn
	- m is compared against condidates and scored

- 2015 Impatient Readers:
	- :punch:
	- Impatient Reader computes attention over the document after reading every word
	of the query. 

- chen 2016:
	- modified version of attentive reader
	- performs significantly better than the original

- memNNs 2015:
	- window memory + self supervision - similar performance
	
	


</details>




<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->



<details><summary> Information Extraction </summary>

Types:
- Event Extraction:
	- that it can have several event
types and multiple events per input.
- Information Extraction:
	- In general we assume, we only have a single event type and assume
there is zero or one event mentioned in the input,
which is an easier task

Datasets
- IE Datasets:
	- ATIS:
	  - ~5k training, ~900 testing
	  - natural language requests to a simulated airline booking system
	  - Each word is labeled with one of several classes, e.g. departure city, arrival city, cost, etc.
	- MIT restaurant:
		- ~7600 train, ~1500 testing
		- Ratings and amenities
		- 10 fields
	- Movie corpus:
		- ~9700 train, ~2400 testing
		- Actors and plots
		- 8 fields
	- Above 3 contains token-level labels in BIO format
- EE datasets:
	- ACE 2005 dataset:
		- :punch:

Approaches:
- Closed or Traditional IE:
	- purely supervised learning with engineered word-level and syntactic features
	- weakly supervised multiple-instance learning:
		- where negative examples are automatically generated from non-annotated entity pairs within a
	sentence. 
		- small size of many annotated datasets: bootstrapping supervised systems from a high-precision
	seed patterns
		- Some contributions brought this approach to the extreme, with
	**self-training methods** that automatically generate their own training data
		- One of the major issues with semi-supervised approaches, both bootstrapped
	and self-supervised, is **semantic drift**, which occurs when erroneous patterns are
	learnt and lead to erroneous triples which, in turn, generate problematic patterns
	where the meaning of the original pattern is substantially altered.
		- NELL “never-ending learning” paradigm.
	- distant supervision paradigm:
		- distantly supervised systems generate a lot of noisy pattern-based features using triples from (possibly human-contributed) knowledge resources, and then combine all these features using supervised classifiers.
	- Statistical Relational Learning paradigm:
		- to couple actual IE with relational inference over knowledge
	bases (Wang and Cohen, 2015), or leverage end-to-end deep neural network models
	to frame the relation extraction task
- Open IE:
	- not only is it fully unsupervised, but it does not even rely on a
	predefined entity or relation inventory at all. 
	- open and unconstrained extraction of an unspecified set of relations, which is not
	given as input, but rather obtained as a by-product of the extraction process. The
	sole input of an OIE system is a large, usually Web-scale, textual corpus.
	:punch:
- Universal schemas:
	- combination of open and closed IE
	- :punch:

[End-to-End Information Extraction without Token-Level Supervision](https://aclweb.org/anthology/W17-4606) (TUDenmark, Tradeshift 2017)

Code: https://github.com/rasmusbergpalm/e2e-ie-release

IE without token level labels using pointers \
Achieve results close to baseline which is uses token-level labels

Baseline:
- 2 layer, Bi-LSTM -> LSTM (128 hidden, 128 emb, Adam)
- BIO labels
- AITS F1: 0.9456

Data:
- Joined multiple output for single lable with commas (multiple diestination)
- Used frequent 10 labesl for ATIS, and all the labels from MIT and Movie corpus.
- prepend inputs with commas to get in the output, LOL

Proposed model:
- Different implementaiton that the originial pointers
- Output is content rather than the position :punch:
- 1 shared encoder
- K decoders one for each type of information to be  extracted
- The output at each time step is a probability distribution over one-hot encoded input.

Modifications:
- For restaraunt data:
	- the parametes were doubles and droupout was used
	- Added "summarizer LSTM" to each decoder ? :punch:
	- last hidden state of summ LSTM is appended to each input of the summarizer

Related work:
- EE model:
	- :punch: Nguyen et al. (2016)
- Generate word level tokens using searching similar words

Cons:
- can only produce words in the input, shouldnt normalize the input data (dates)


[Attend, Copy, Parse End-to-end information extraction from documents](https://arxiv.org/pdf/1812.07248.pdf) (Tradeshift 2017)

Extract information from images of business documents, invoices \ 
Uses images, words and the word's position to extract output strings \
Some modification in loss function and regularization which might be interesting :punch:


</details>




<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->




<details><summary> Summarization </summary>

http://nlpprogress.com/english/summarization.html

Datasets:
- Coversation Dataset: AMI corpus
- The Argumentative Dialogue Summary Corpus
(Misra et al. 2015) consist of 225 summaries, 5
different summaries produced by trained summarizers, of 45 dialogue excerpts on topics like gun
control, gay marriage, the death penalty and abortion. This was used for experiments and evaluations.
- CNN/Daily Mail Dataset
The dataset used by See et al. (2017) was the
CNN/Daily Mail dataset (Hermann et al., 2015;
Nallapati et al., 2016), which contains online
news articles (781 tokens on average) paired with
multi-sentence summaries (3.75 sentences or 56
tokens on average).
- The DUC corpus8
comes in two parts: the 2003
corpus consisting of 624 document, summary
pairs and the 2004 corpus consisting of 500 pairs.
- Gigaword corpus contains about 3.8M training examples

Compared to MT, here the target is shorter than the input, we want a lossy translation and one-to-one word level alignemnt is less obvious here

Two types of repetition avoidance:
- Intra-decoder attention as used in the above-mentioned paper, to let the decoder access its history (attending over all past decoder states).
- Coverage mechanism, which discourages repeatedly attending to the same area of the input sequence: See Get To The Point: Summarization with Pointer-Generator Networks by See and Manning for the coverage loss (note that the attention here incorporates the coverage vector in a different way).

Trends:
- Extractive sentence selection
- RL loss + ML loss
- Pointer generator
- coverage mechanism
- Intra-decoder attention
- Embedding sharing across encoder, decoder input, and decoder output.
- Initialization with pre-trained word embeddings.
- Teacher forcing ratio.


[Automatic Community Creation for Abstractive Spoken Conversation Summarization](https://www.aclweb.org/anthology/W17-4506) (Italy, 2017)

Poorly written paper. \
This paper focuses on Template based summarization which needs links between summary and conversation (we need this anyway). \
Describes a way to find links from human generated summary and conversation which can be used for training.

Pipeline: Community creation, template generation, ranker training, and summary generation components.

Template generation: Generate templates from summaries by (POS tagging -> dependency parsing -> wordnet -> clustering -> word graph algorithm

Community creation: Similar to topic extraction. Here they explore different way to cluster sentences

Summary generation: topic segmentation, template identification (for each topic I guess), extract slot fillers, fill the template with fillers

Sentence Ranking: Ranking filled template sentences with n-grams pos and tokens. This is dont to prevent repetetion of information.


[A Neural Attention Model for Sentence Summarization](https://aclweb.org/anthology/D15-1044) (FB, 2015)

Dataset: headline generation in Gigaword 4 million articles and DUC-2004,2003 shared task \
One of the first good deep learning based abstractive summarization paper

The model shows significant performance gains on the DUC-2004 shared task compared with several strong baselines.

attention-based encoder + beam-search decoder \
Fixed vocabulary \
Output length is fixed \
Abstractive summarization = finding optimal sequence of N words from vocaublary \
Extractive summarization = finding optimal sequence of N words from input (this can be sentence compression if we place constrains on the output sequence order) \
Here they generate yi+1 using input x, and previous c window summary yc by using conditional log prob and markov assumption. \
Modelling the local conditional distribution. -> conditional language model (neutal machine translation) \

Neural machin translation: models the distribution directly instead of spliting and estimating individually. \

Here the encoder takes yc and x as input to produce prob of yi+1.

They consider:
- bag-of-words enc
- conv encoder
- attention enc

Decoding:
- viterbi decoding, is tractable but takes a lot of time.
- replace argmax with greedy/deterministic approaches- although bad is effective and fast.
- beam serch is an comprimise between the two (here it is simpler than phrase-based MT)

Extension:
- this is bad for unseen proper nouns 
- To solve this they add additional feature to the final word probability and combine them with weight to get the final score. 
- these features encourage using a word from the input.

"The minibatches
are grouped by input length. After each epoch, we
renormalize the embedding tables"


[Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://arxiv.org/pdf/1602.06023.pdf) (IBM, 2016)

Gigaword, DUC, CNN daily mail

Attentional EncoderDecoder Recurrent Neural Networks \
We
propose several novel models that address
critical problems in summarization that
are not adequately modeled by the basic
architecture, such as modeling key-words,
capturing the hierarchy of sentence-toword structure, and emitting words that
are rare or unseen at training time.

Basic model:
- The encoder consists of a bidirectional GRU-RNN
- decoder
consists of a uni-directional GRU-RNN with the
same hidden-state size as that of the encoder
-  attention mechanism over the source-hidden
states and a soft-max layer over target vocabulary to generate word
- Large vocabulary trick: decoder-vocabulary of each mini-batch is restricted to words in the source documents of that
batch.
- In addition, the most frequent words in the
target dictionary are added until the vocabulary
reaches a fixed size.
- reduces softmax size (computational bottle-neck) and helps modelling by restricting vocab

Extensions:
- Keyword capturing: word-embeddings-based representation of the input document and capture additional linguistic features for encoder
- Switching Generator-Pointer
- Hierarchical Document
Structure with Hierarchical Attention: if source is long: bi-dir RNNs on the source side, one at the word level
and the other at the sentence level. The attention
mechanism operates at both levels simultaneously.
- sentence positional embedding
- If the summary is long there is a repetition problem- Use temporal attention to solve it "keeps track of past attentional weights
of the decoder and expliticly discourages it from
attending to the same parts of the document in future time steps"

[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf) (brain, 2017)

A subset of IBM, 2016 paper. Explores pointer generator and coverage mechanism. \
Scores are better than their paper though: 
"Those works train their pointer components to activate only for out-of-vocabulary words
or named entities (whereas we allow our model to
freely learn when to use the pointer), and they do
not mix the probabilities from the copy distribution and the vocabulary distribution. We believe
the mixture approach described here is better for
abstractive summarization"


[Controlling Decoding for More Abstractive Summaries with Copy-Based Networks](https://arxiv.org/abs/1803.07038) (stonybrook ,2018)

:punch:

analysis on pointer-generators

[A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION](https://arxiv.org/pdf/1705.04304.pdf) (salesforce 2017)

RL loss + ML loss \
uses pointers too

:punch:

[Generative Adversarial Network for Abstractive Text Summarization.](https://arxiv.org/pdf/1711.09357.pdf) (china, 2017)

G: attention + pointer generator network

1. Train G till -- more realistic and high quality summaries
2. Train and fix D
3. Train G

D: CNN + max-overtime pooling

G: loss = policy gradient + maximum likliehood \
pg: REINFORCE algorithm and take the estimated probability of being human generated by the discriminator D as the reward -- Since the Discriminator can only score a fully generated sequence, following (Yu et al. 2016), we use Monte Carlo Tree Search (MCTS) to evaluate the reward for an intermediate state :punch:

D: usuall loss

Added Trigram avoidance and quotation weight alleviation


[Toward Controlled Generation of Text](https://arxiv.org/pdf/1703.00955.pdf) (CMU, 2018)

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



[Summarization of Spoken Language—Challenges, Methods, and Prospects](www.cs.cmu.edu/~./zechner/ezine.ps) (CMU 2002)

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

</details>






<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->






<details><summary> Clinical NLP </summary>

[GRAM: Graph-based Attention Model for Healthcare Representation Learning](https://arxiv.org/abs/1611.07012) (GaTech, 2016)

supplements electronic health records (EHR) with hierarchical information inherent to medical ontologies
GRAM represents a medical concept as a combination of its ancestors in the ontology via an attention mechanism.
Testing on rare disease prediction and heart failure.

medical codes as DAG, then use embedding for all leaf nodes (nodes with meaning full concepts) and visit one-hot embedding to generate a vist vector (process incoporates attention mechanism). Finally use the vist vector to predict stuff.


[Revealing the Structure of Medical Dictations with Conditional Random Fields](http://www.aclweb.org/anthology/D08-1001) (2008, medical univ vienna) [Identifying Segment Topics in Medical Dictations](http://www.aclweb.org/anthology/W09-0503) (2009, medical univ vienna)

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


[Deep EHR: A Survey of Recent Advances on Deep Learning Techniques for Electronic Health Record (EHR) Analysis](https://arxiv.org/pdf/1706.03446.pdf) (UofF 2017)

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


[“Exploiting Task-Oriented Resources to Learn Word Embeddings for Clinical Abbreviation Expansion](https://nlp.cs.rpi.edu/paper/bionlp15.pdf) RPI, 2015

Abbr are ambiguous especially in intensive care  
embedding for abbr and their expansion should have similar embedding

:trollface:


[Brundlefly at SemEval-2016 Task 12: Recurrent Neural Networks vs. Joint Inference for Clinical Temporal Information Extraction](https://arxiv.org/pdf/1606.01433.pdf) Stanford 2016

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


[“Clinical Relation Extraction with Deep Learning](https://pdfs.semanticscholar.org/7fac/52a9b0f96fcee6972cc6ac4687068442aee8.pdf) Harbin China 2016

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


[Structured prediction models for RNN based sequence labeling in clinical text](https://arxiv.org/abs/1608.00612) UofM, Aug 2016

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


[Bidirectional RNN for Medical Event Detection in Electronic Health Records](http://www.aclweb.org/anthology/N16-1056) UofM, June 2016

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


[Multi-layer Representation Learning for Medical Concepts](https://arxiv.org/abs/1602.05568) Feb 2016, GaTech + children healthcare atlanta

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
</details>




<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->
<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->
<!--- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --------------------------------- --->




## Speech



<details><summary> General </summary>


[A time delay neural network architecture for efficient modeling of long temporal contexts](speak.clsp.jhu.edu/uploads/publications/papers/1048_pdf.pdf) (Povey, 2015)    +
[JHU ASPIRE SYSTEM : ROBUST LVCSR WITH TDNNS, IVECTOR adaptation and RNN-LMs](https://www.danielpovey.com/files/2015_asru_aspire.pdf) (Povey, 2015)

3-fold reduced frame rate \
TDNN faster than rnns because of parallizations and subsampling \
data augumentation using reverberation, speed peturbation (not helpful) and volume peturbation (multi-condition training is very important)

iVector features: 
- normalize test with training stats
- iVector extraction in this dataset doesnt do well if the speech segment contains even a small part of silence (use VAD or two-pass decoding to remove it)

TDBB trained with greedy layer-wise supervised training on 18 gpus with model averaging techniques \
Trained using sMBR with MPE + insertion penality error \
GMM-HMM AM model used to generate CD state alignments

Used CMUdict for training lexicons with multiple pronunciations also modelling inter-word silences \
3-gram LM used for decoding with 4-gram used for rescoring the lattice \
N-gram LMs trainined by using 3M words of the training transcripts later interpolated using the 22M words of the Fisher English transcripts ? :punch: \
RNN-LM lattice rescoring using context vector instead of words

6 layers TDNN with unsymmetric context window

Modified sMBR better than sMBR \
Modified sMBR still prone to insertion errors \
70% of the test data had modified sMBR better than cross-enrtopy \
for 30% cross-entropy was much better than modified sMBR




[CLDNN-HMM](https://www.semanticscholar.org/paper/Convolutional-Long-Short-Term-Memory-fully-connect-Sainath-Vinyals/56f99610f7b144f55a511da21b74291ce11f9daf)
:punch:


[EFFICIENT LATTICE RESCORING USING RECURRENT NEURAL NETWORK LANGUAGE MODELS](http://mi.eng.cam.ac.uk/~mjfg/xl207_ICASSP14a.pdf) (cambridge) (2014)

Rescoring methods:
* n-gram style clusteing of history contexts
  - data sparsity issues
  - large context leads to exponential size growth
* distance in hidden history vectors
  - [RNNLM](#rnnlm) & and FFNNLM :punch: readmore

:trollface: readmore



[Prefix Tree based N-best list Re-scoring for Recurrent Neural Network Language Model used in Speech Recognition System](https://pdfs.semanticscholar.org/5f59/1b7043deefbc3f3af19b6efeb97c2a80d27c.pdf) China 2013 

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

- Bunch Mode
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










<details><summary> Loss functions </summary>

[Purely sequence-trained neural networks for ASR based on lattice-free MMI](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf) (Povey, 2016)

MMI denomenator computation without using lattice ie on all possible word labellings \
3-fold reduced frame rate \
Phone-level LM for speed \
On GPU

:punch:


[BOOSTED MMI FOR MODEL AND FEATURE-SPACE](https://www.danielpovey.com/files/icassp08_mmi.pdf) (Povey, 2008)

MMI - maximize the posterior prob of correct utter given our model/all other utter (discriminative)

modify the objective funtion to take the accuraccy of the sent3ence into consideration -> this makes BMMI very similar to MPE.
Accuraccy for all the sentences are computed per phone. And similat to MMI we compute statistics using forward-backward algo to train it.

Also uses I vector smiootheninig on statistics accumulates. We back of to ML estimates


[A NOVEL LOSS FUNCTION FOR THE OVERALL RISK CRITERION BASED DISCRIMINATIVE TRAINING OF HMM](https://pdfs.semanticscholar.org/de8c/eb72bf54293959813c101c4f7ce54fbd3a20.pdf) (University of Maribor, 2000)

MBR training of ASR systems \
MBR minimizes expected loss

aim to directly max word recog accuraccy on training data

generally MAP is used for ASR argmax w P(w|o) = argmax_w p(o|w) * p(w) \
p(o|w) is AM, with HMM it becomes p(o_r | theta_r) for which MLE give best theroritically. practically they use MMI or MCEE (Min classification error estimation). \
Modification of MCEE is ORCE overall risk creterion estimation. 


In this paper they extend ORCE objective to continuous speech recognition and use a non-symmetrical loss using the number of I, S, D in WER calculation instead of 1/0 loss.

experiments on TIMIT dataset on HMM.


[Hypothesis Spaces For Minimum Bayes Risk Training In Large Vocabulary Speech Recognition](https://pdfs.semanticscholar.org/0687/573a482d84385ddd55e708e240f3e303fab9.pdf) (University of Sheffield, 2006)

State-level MBR training

MBR training good for large vocab HMMs, implementation needs hypothesis space and loss fn. \
MMI is better than MLE training of AM (HMMs) \

minimum phone error can be interpreted as MBR when phone sequence forms hypothesis space -> better than MMI \

Lattice-based MBR -> constraining the search space to only those alignments specified by the lattice \
to do this we need l(w_reference, arc_i) is  difficult.

a solution explored here is comming up with Frame Error Rate FER.

[Tree-Based State Tying for High Accuracy Modelling](www.aclweb.org/anthology/H94-1062) (Cambridge, 1994)

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

[Subphonetic Modeling for Speech Recognition](https://core.ac.uk/download/pdf/22876656.pdf) (CMU, 1992)

Advocates for state-level (output-distribution level) parameter sharing instead of model-level and the use of state-dependent senones. \
Senones alow parameter sharing/reduction, pronunciation optimization and new word learning 

After generating all the word HMM models, cluster the senons and generate the codebook. Then replace the senones with nearest ones in the codebook. \
The clustering start by assuming all the data points are seperate clusters then a pair are merged if they are similar (If the entropy increase is small after merging then two distributions are similar). 

Explores 3, 5, 7 state triphone models and finds than 5 is the most optimal one 

</details>











<details><summary> Cleaning Noisy Speech Dataset </summary>

[A RECURSIVE ALGORITHM FOR THE FORCED ALIGNMENT OF VERY LONG AUDIO SEGMENTS](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.649.6346&rep=rep1&type=pdf) (Cambridge, 1998)

A recursive alignment with ASR + restricting dictionary and LM \
Introduces the concept of anchors with island of confidences \
Dictionary (phonetic) is built using CMU public domain dictionary plus an algo \
A simple LM with word pair and triple model for the transcript specifically

Number of consecutive word matches needed for confidence islands is in the early point of the recursion to reduce the possibility of error in the early stage as it can affect the entire pipeline.

Used for indexing the audio using the words in the audio file. Error of 2 sec is tolerated.

General discussion:
- Viterbi is time consuming for long audio and if it gets an error it will make it completely wrong.
- increasing the beam search helps viterbi but it only scales for short audio

[A SYSTEM FOR AUTOMATIC ALIGNMENT OF BROADCAST MEDIA CAPTIONS USING WEIGHTED FINITE-STATE TRANSDUCERS](https://homepages.inf.ed.ac.uk/srenals/pb-align-asru2015.pdf) (univ of Edinburgh, 2015)

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












<details><summary> End-to-End </summary>
	
[Joint Speech Recognition and Speaker Diarization via Sequence Transduction](https://arxiv.org/pdf/1907.05337.pdf) (Google, 2019)
	
The authors explore combining speaker diarization and ASR into a single sequence to sequence model.The combined model achieves dramatically low diarization (speaker role detections) rate.

The model predicts the speaker using lexical cues and acoustic features. Recurrent neural network transducer model is used for the end to end joint diarization cum speech recognition. The loss function used dynamic programming with forward and backward algorithm. 15k hrs of doctor patient conversation was used for training and testing the model.

The model has 3 parts, 1. transcription network or the encoder which converts acoustic frames to higher-level representation, 2. prediction network which predicts next labels based on the previous non-blank symbol, 3. joint network which combines the above two outputs to produce logit which is converted to probability distribution using softmax.

The acoustic model uses morphemes instead of graphemes since it is a higher duration model, to achieve this time-delay neural networks were used, it reduces the output time resolution from 10 to 80 millisecond.

The architecture took 2 days to train on 128 TPU. 4k morphemes (data driven) were used. The network uses 1D temporal convolution layers, max pooling, uni and bi-directional lstms.

Observation:
The model sadly rarely misses a speaker change, but when it does, it does not correct it later, this is a side-effect of the training approach.
The speaker roles of non doctor and patients were inferred to be the closest of the two.



[Towards End-to-End Speech Recognition with Recurrent Neural Networks](http://proceedings.mlr.press/v32/graves14.pdf) (graves, 2015) 

Modified CTC objective function. Instead of MLE, this version is trained by directly optimizing WER.
Done using samples to approximate gradients of the expected loss function (WER).

No lattice level loss here.




[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (Graves, 2006)

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

[Optimizing expected word error rate via sampling for speech recognition](https://arxiv.org/abs/1706.02776) (Google, 2017)

Define word-level Edit-based MBR (EMBR) on lattice generated during SMBR.\
they do it by using monte-carlo samples from the lattice to approximate the gradient of the loss function which is in the form of an expectation.\
Similar to Reinforce.

Gradient has the form (average loss - loss of state i) so cannot be used during the starting phase of the training.

Generalized version of sample based loss derived in the CTC,2015 paper. Where the CTC paper doesnt use lattice level loss function.




[Listen Attend and Spell (2015) Google Brain](https://arxiv.org/abs/1508.01211)

10.3, 14.5% WER compared to 8% state of the art [cldnn-hmm](#cldnn-hmm)

Dataset: Google voice search tasl

* Listner(PBLSTM) -> Attention (MLP + Softmax) -> Speller (MLP + Softmax) -> Characters
* No conditional independence assumption like CTC 
* No concept of phonemes
* Extra noise during training and testing
* Sampling trick for training PBLSTM
* Beam search(no dictionary was used 'cause it wasnt too useful) + LM based rescoring (very effective) 
[Know about rescoring](#rescoring-1)
* Async stoc gradient descent [aync](#asyc)

- Suggestions
	* Convolution filters can improve the results [TODO](#20paper) :punch:
	* Bad on short and long utterances [TODO](#15paper) :punch:


</details>











<details><summary> Diarization </summary>
  
[Deep Learning Approaches for Online Speaker Diarization](http://web.stanford.edu/class/cs224s/reports/Chaitanya_Asawa.pdf) (2012)

[SPEAKER DIARIZATION WITH LSTM](https://arxiv.org/pdf/1710.10468.pdf) (google, 2018)

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

</details>






## General Deep Learning

<details><summary> Theory </summary>

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.7873&rep=rep1&type=pdf

[Multilayer Feedforward Networks are Universal Approximators ](http://cognitivemedium.com/magic_paper/assets/Hornik.pdf) (1988 UofC)

Rigorously establishes thut standard multiluyer feedforward networks with one
hidden layer using arbitrary squashing functions are capable of approximating any Borel measurable function
from one finite dimensional space to another to any desired degree of auccuracy.

The main idea behind the proof is to show that Stone-Weierstrass Theorem can be applied to the networks and the space.

- \sum\prod^{r} (G) functions/networks is universal approximator for any continuous nonconstant function G.
- \sum{r} (G) functions/networks is universal approximator for any squashing function G.


https://papers.nips.cc/paper/7203-the-expressive-power-of-neural-networks-a-view-from-the-width

https://arxiv.org/abs/1708.02691

</details>






<details><summary> General </summary>

The noisy channel model is a framework used in spell checkers, question answering, speech recognition, and machine translation. In this model, the goal is to find the intended word given a word where the letters have been scrambled in some manner.

[Async stoc gradient descent](http://www.ijcai.org/Proceedings/16/Papers/265.pdf)
:boom:



[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://github.com/saiprabhakar/Papers/blob/master/files/1502.03167v3.pdf)
- Problem: Internal covariance shift
	* Distribution of each layer changes as the parameters of its previous layers are updated this makes training slow and needs less learning rate.

- Solution: BN
	* Makes normalization a part of architecture
	* Lowers training time. Higher learning rate can be used. Sometime eliminates the need for Dropout
	* Fights vannishing and exploding gradients because of normalization (scale of weights doesnt matter)

- Covariance shift 
	* When the distribution of input to a learning system chages (whole system as a whole)
	* Usually handled by domain adaptation
	* ICS is an extenstion when part of it changes

- Notes
	* Training is faster in general if the inputs are whitened (line tras to have 0  mean and sigma = 1 and decorrelated)
	* Ignoring the BN during gradient descent is bad idea since it leads to explosion of parameters like bias terms
	* There were previous less successfull attemps on this idea
	* Simply normalizing layers can constrain them. For example normalizing simoid layer would constrain them to the linear portion their nonlinearity. **So they introduced additional parameter (gamma and beta) to make sure the normalization can represent identity transformation.**
	* Incase of Conv layer, we need to follow conv property. Different elements of the same feature map, at diffrent locations are notmalized the same way. We learn gamma and beta per feature map and not per activation.
	* Applied BN before nonlinearity, where as standardization (2013) was after then nonlinearity.

- Further possible extensions
	* Similarity between BN and standardization
	* Extension to RNNs (where vanishing and exploding gradients are more severe)
	* More theoritical analysis
	* Application to domain adaptation

---

[ModDrop: adaptive multi-modal gesture recognition](https://arxiv.org/abs/1501.00102)
- Notes
	* Modalities (as a whole) are dropped with a probability during training
	* They are trained without fusing during pretraining and are not droped at this point
	* cross modal connections are introduced at training stage
	* Dropping is only at the input layer
	* Rescaling?

- Notes from citation
	* Out Performs Dropout
	* Combining with dropout gives higher performance


[Modout: Learning to Fuse Modalities via Stochastic Regularization](http://openjournals.uwaterloo.ca/index.php/vsl/article/view/103)
- Notes
	* Learns the probability of fusing modalities
	* Connection between modalities btwn adjacent layers are dropped with a probability
	* Dropping can be done in any layer
	* No. of extra parameters to learn are small Nm x (Nm-1), where Nm is the number of modalities
	* Very similar to **Blockout**


[Input Convex Neural Networks](https://arxiv.org/abs/1609.07152)
- Notes
	* Under certain condition of weights and nonlinearity a neural network will be convex in certain inputs/outputs, so we can efficiently optimize over those inputs/outputs while keeping others fixed

- Fully input convex neural networks
	* Convex interms of all the inputs
	* Conditions: non-negative weights (restricts the power of the network) and non-decreasing non-linearities

- Partially input convex neural networks
	* Convex in certain inputs and not convex in others
	* PICNN with k layers can represent and FICNN with k layers and any feedforward net with k layers

- Inference
	* Inference wrt to the convex variable are not done in a single pass as in feed forward network case
	* Inference can be found by using optimization techniques like LP, approximate inference etc

- Learning
	* In case of q learning the fuction fitting is automatically taken care of as the goal is to fit the bellman equation
	* For fitting some target output they use techniques like max-margin etc

- Results
	* Preliminary results for DRL shows faster convergence comparision to DDPG and NAF
	* Can complete face (fix some inputs while solve for others)
	* Classification task need more investigation

---
</details>

