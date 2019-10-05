Talks_and_Blogs
=================


## Natural Language Processing


<details><summary>
DAVID BAMMAN on applying natural language processing and machine learning to empirical questions in the humanities and social sciences
</summary>

CMU 2019, https://lti.cs.cmu.edu/lti-colloquium

We usually represent meaning of fatual text formally using Neo David Sone representation,
- Doesnt work on literature with metaphors and figurative language

Works in modeling literary phenomena focus on:
- character type
- Relationship
- sentiment or plot
- Character psychology

Works in computational Humanities focus on: 
 - time passing, loudness (verb)
 - emotions

Gender bias analysis on literary text:
- screen time or attention is given to a character can be analyzed using POS, NER, DP, Name clustering

Plot decomposition:
- character:  entity recognition  
- Event: Event detection
- Settings:  entity recognition / setting coreference
- objects: object detection / coreference
- time: temporal processing / event ordering

Accuracy of the pipeline goes down:
- named entity recognition 100% - > coreference resolution 75% on WSJ


News -> Non-News corpus  performance goes down
- Domain adaptation
- Contextualized word representation
- Data annotation (training on new data bring the performance to as expected during WSJ)

NER:
- nested named entity recognition
- metaphor detection
- personification

Events:
- Detection, slot filling
- factuality and belief of an event
- temporal grounding
- Event chain scheme

realist event protection ( event happen or not):
- positive polarity
- Tense
- specific or generic 
- Mortality:
    - Accepted or not accepted
    - Belief 
    - Hypotheticals
    - Command
    - Threats 
    - Desires

Performance:
- 80 % of the events are verbs
- contextual embeddings perform better than non-contextual embeddings

Using events we can find:
- The abstractness of a novel
- prestige
- Event timeline
