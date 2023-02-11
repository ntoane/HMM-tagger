## First-order Hidden Markov Model POS tagger
- This supervised learning Hidden Markov Model Part-of-Speech Tagger is trained on the Sesotho language given as input in the training CSV file of rows of words. The end of a sentence is marked by a full stop and paragraphs are separated by blank rows. 
- The model computes emission and transition probability tables and uses the Viterbi algorithm to tag unseen words given in a testing CSV file with the same structure as the training file. 
- The tagged words are written to the output text file named TestOutput.txt.

### Install these packages
- python 3x