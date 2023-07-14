# Grapheme To Phoneme Conversion

This repository holds code written in a process of a my graduation project (Matura) in the school year 2019/2020. 
The aim of the project was to train a neural network to deduce phonetic transcriptions of English or English sounding words. This was done with the help of a Carnegie Mellon University pronouncing dictionary, which provided a long list of English words with their phonetic transcriptions in the ARPABET phonetic alphabet. The dictionary is stored in index.json. Please note, that the dictionary used here may not be up to date as it is from autumn 2019.

The model used for the neural network was an LSTM (long short term memory) sequence-to-sequence model.
Libraries used: sklearn, keras, pandas, numpy
