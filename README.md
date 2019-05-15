# Experimental machine learning and computer vision methods in Keras

This repoisitory consists of a number of Keras implementations of machine learning algorithms.
Any algorithm that is not original/ my creation will have a link to the paper that proposed it (or proposes a similar idea).

##Feature Extraction
These scripts use pre-trained convolutional architectures to extract features from images.
The extracted features can then be used in a range of classical classifiers.

##Greedy Layerwise Training
These scripts trains convolutional architectures in a sequential layerwise fashion.
The early stages of the network are selected and trained as an independent classifier.
After training those layers are frozen, the next layers (unfrozen) are added and training resumes.
Inspired by "Effective training of convolutional neural ntworks with small, specialized datasets" - Plata, Diego Ruedaa, Ramos-Pollán, Raúla, González, Fabio ( https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs169131 )


##Progressive Networks
Similar to the networks presented in : Progressive Neural Networks Andrei A. Rusu*, Neil C. Rabinowitz*, Guillaume Desjardins, Hubert Soyer,James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, Raia Hadsell (https://arxiv.org/pdf/1606.04671.pdf)

