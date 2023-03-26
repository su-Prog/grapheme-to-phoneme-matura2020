import time
import string
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Input, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#unboxing all of the df from cmu_df.pickle
df = pd.read_pickle('cmu_df.pickle')

# delete words over 20 characters long
df = df[~(df.Word.str.len() > 20)]

# data info
data_len = 116503
test_size = 0.2
num_phones = 39
num_phonemes_with_stress = 84

#find out the maximum length of inputs and outputs
def max_len(list):
    lengths = []
    for i in list:
        length = len(i)
        lengths.append(length)
    return max(lengths)

#import the phonemes list
phonemes_txt = open("phonemes.txt", "r")
phonemes_with_stress = []
start_symbol = '\t'
end_symbol = '\n'

#create a list of phonemes with and without stresses (all phonemes used in transcrptions)
def txt_to_list(file, list):
    for i in file:
        list.append(i)

    #get rid of \n at the end
    for i,s in enumerate(list):
         list[i] = s.strip()
    return list

phonemes_with_stress = txt_to_list(phonemes_txt, phonemes_with_stress)
phonemes_with_stress.append(start_symbol)
phonemes_with_stress.append(end_symbol)

#create the list of all used characters in the words
character_list = list(string.ascii_uppercase)

#mapping for each character/phoneme so they are mapped to a number (str_to_id) and back (id_to_str) => dictionary
def id_mappings_from_list(str_list):
    str_to_id = {s: i for i, s in enumerate(str_list)}
    id_to_str = {i: s for i, s in enumerate(str_list)}
    return str_to_id, id_to_str

# Create character to ID ( token) mappings
char_to_id, id_to_char = id_mappings_from_list(character_list)

# Load phonetic symbols and create ID (token) mappings
phone_to_id, id_to_phone = id_mappings_from_list(phonemes_with_stress)

#number of the character/phoneme options

n_character_tokens = len(char_to_id)
n_phonemes_tokens = len(phone_to_id)

#each character to a 1-hot vector

def char_to_1_hot(char):
    char_id = char_to_id[char]
    hot_vec = np.zeros((n_character_tokens))
    hot_vec[char_id] = 1.
    return hot_vec

#each phoneme to a 1-hot vector

def phone_to_1_hot(phone):
    phone_id = phone_to_id[phone]
    hot_vec = np.zeros((n_phonemes_tokens))
    hot_vec[phone_id] = 1.
    return hot_vec

# max length of the words and the phoneme sequences (for phoneme the starting and ending sequences added)
phone_seq_lens = []
for index, row in df.iterrows():
    length = len(str.split(row["Phonemes"]))
    phone_seq_lens.append(length)

max_char_seq = 20
max_phone_seq = max(phone_seq_lens) + 2

#pandas dataframe to dictionary, row 0 to values, row 1 to keys
dataset_dictionary = {row[0]: row[1] for row in df.values}


'''
#make lists of matrices that represent words and phonetic transcriptions
char_seqs = []
phone_seqs = []

#use one hot encoding mappings on items in the dictionary and append to lists
for word, pronunciation in dataset_dictionary.items():
#fixed dimensions of word matrices
    word_matrix = np.zeros((max_char_seq, n_character_tokens))
    for t, char in enumerate(word):
        word_matrix[t, :] = char_to_1_hot(char)
    char_seqs.append(word_matrix)
#fixed dimensions of transcrription matrices
    pronun_matrix = np.zeros((max_phone_seq, n_phonemes_tokens))
#add start and end symbol for training and predicting
    phones = [start_symbol] + pronunciation.split() + [end_symbol]
    for t, phone in enumerate(phones):
        pronun_matrix[t,:] = phone_to_1_hot(phone)
    phone_seqs.append(pronun_matrix)
'''

#put them in a numpy document with numpy.save() but first turn into a numpy array
#char_seq_matrix, phone_seq_matrix = np.array(char_seqs), np.array(phone_seqs)

#np.save('char_seq_matrix.npy', char_seq_matrix) # save
char_seq_matrix = np.load('char_seq_matrix.npy') # load

#np.save('phone_seq_matrix.npy', phone_seq_matrix) # save
phone_seq_matrix = np.load('phone_seq_matrix.npy') # load

#print('Pronunciation Matrix Shape: ', phone_seq_matrix.shape)
#print('Word Matrix Shape: ', char_seq_matrix.shape)
Word_Matrix_Shape = (116503, 20, 26)
Pronunciation_Matrix_Shape = (116503, 21, 86)

#shift by one step in the decoder output so that the start symbol is not in output
phone_seq_matrix_decoder_output = np.pad(phone_seq_matrix,((0,0),(0,1),(0,0)), mode='constant')[:,1:,:]

#encoder, decoder architecture, model
def seq2seq_model(hidden_nodes = 256):

    # Shared Components - Encoder (LSTM NN, states passed on, dropout optional)
    # Input layer initialization for the encoder
    char_inputs = Input(shape=(None, n_character_tokens))
    encoder = LSTM(hidden_nodes, return_state=True, recurrent_dropout=0.3) #recurrent_dropout=0.2, none and 0.5, same for decoder

    # Shared Components - Decoder (LSTM NN, FNN layer, states passed on, dropout optional)
    # Input layer initialization for the decoder
    phone_inputs = Input(shape=(None, n_phonemes_tokens))
    decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True, recurrent_dropout=0.3)
    decoder_dense = Dense(n_phonemes_tokens, activation='softmax')  #probability distribution due to softmax

    # Training Model, encoder states are used for training
    _, state_h, state_c = encoder(char_inputs)      #outputs of encoder
    encoder_states = [state_h, state_c]
    #inputs of decoder are the encoder output states and previous phones
    decoder_outputs, _, _ = decoder(phone_inputs, initial_state=encoder_states)
    phone_prediction = decoder_dense(decoder_outputs)

    training_model = Model([char_inputs, phone_inputs], phone_prediction)

    # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_states)

    # Testing Model - Decoder
    #Input layer initialization for the decoder
    decoder_state_input_h = Input(shape=(hidden_nodes,))
    decoder_state_input_c = Input(shape=(hidden_nodes,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    phone_prediction = decoder_dense(decoder_outputs)

    testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [phone_prediction] + decoder_states)

    return training_model, testing_encoder_model, testing_decoder_model

#split into train and test data
#specific random state, so that the split is always same ( same data in test data each time the code is executed in the same order)
(char_input_train, char_input_test,
 phone_input_train, phone_input_test,
 phone_output_train, phone_output_test) = train_test_split(
    char_seq_matrix, phone_seq_matrix, phone_seq_matrix_decoder_output,
    test_size=test_size, random_state=42)

#length of test dataset
TEST_EXAMPLE_COUNT = char_input_test.shape[0]

#ask path for the model saving before training or which model to use if done for predicting/testing
weights_path = input("Where should the model be saved? / What model would you want to use? ")

#training model, hyperparameters
def train(model, weights_path, encoder_input, decoder_input, decoder_output):
    #saves the model, only the best one
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True, save_weights_only= False)
    #stops if the model starts overfitting, monitors the validation data loss
    stopper = EarlyStopping(monitor='val_loss', patience=3)
    #time the process
    start = time.time()
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  #optimizer=Adam(lr=0.001)
    history = model.fit([encoder_input, decoder_input], decoder_output,
          batch_size=256, #I kept it fixed, smaller gave errors
          epochs=100,       #never made it to 100
          validation_split=0.2, # validation data split from the 80% of initial amount of data to evaluate models during training for new data
          callbacks=[checkpointer, stopper])
#trainiing time
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    # Plot validation loss values
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

training_model, testing_encoder_model, testing_decoder_model = seq2seq_model()

#Time the process

#train(training_model, weights_path, char_input_train, phone_input_train, phone_output_train)

def predict_baseline(input_char_seq, encoder, decoder):
    #get hidden state and cell state vectors
    state_vectors = encoder.predict(input_char_seq)
    #start symbol always used as the initial phoneme
    prev_phone = np.zeros((1, 1, n_phonemes_tokens))
    prev_phone[0, 0, phone_to_id[start_symbol]] = 1.
    # look for the end symbol in prediction and cut it and everything after off by stopping transcribing
    end_found = False
    pronunciation = ''
    while not end_found:
        #concatenate the previous phoneme and the state vectors from the encoder and apply the decoder
        decoder_output, h, c = decoder.predict([prev_phone] + state_vectors)

        # Give back the index of the vector component with the highest probability with argmax
        #the index can be converted back to the phoneme
        predicted_phone_idx = np.argmax(decoder_output[0, -1, :])
        predicted_phone = id_to_phone[predicted_phone_idx]
        #Add all the phonemes to a phonetic transcription
        pronunciation += predicted_phone + ' '

        #stop at the end symbol
        if predicted_phone == end_symbol or len(pronunciation.split()) > max_phone_seq:
            end_found = True

        # define the new previous phoneme based on the last phoneme prediction, use new state vectors
        prev_phone = np.zeros((1, 1, n_phonemes_tokens))
        prev_phone[0, 0, predicted_phone_idx] = 1.
        state_vectors = [h, c]

    return pronunciation.strip()

# vector back to word with a mapping using the previous indices
def one_hot_matrix_to_word(char_seq):
    word = ''
    for char_vec in char_seq[0]:
        if np.count_nonzero(char_vec) == 0:
            break
        hot_bit_idx = np.argmax(char_vec)
        char = id_to_char[hot_bit_idx]
        word += char
    return word


#Check if the prediction matches the real transcription
def is_correct(word,test_pronunciation):
    correct_pronun = dataset_dictionary[word]
    #info used as a list including two pieces of information, whether the model predicted correctly and what the correct transcription is
    if test_pronunciation == correct_pronun:
        info = [True, correct_pronun]
        return info
    else:
        info = [False, correct_pronun]
        return info


#testing on data and showing the results
def sample_baseline_predictions(sample_count, word_decoder):
    sample_indices = range(sample_count)
    a = 0
    for example_idx in sample_indices:
        example_char_seq = char_input_test[example_idx:example_idx+1]
        predicted_pronun = predict_baseline(example_char_seq, testing_encoder_model, testing_decoder_model)
        example_word = word_decoder(example_char_seq)
        pred_is_correct = is_correct(example_word, predicted_pronun)
        state = pred_is_correct[0]
        correct_transc = pred_is_correct[1]
        print('✅ ' if state else '❌ ', example_word,'-->', predicted_pronun," correct: ", correct_transc)
        if state:
            a = a+1
    print(a, "out of ", sample_count, " samples were predicted correctly.")
    #percentage of correctly predicted transcriptions for words
    print("Perfect Accuracy : ", a*100/sample_count, "%")

training_model.load_weights(weights_path)# also loads weights for testing models

#predict for all test samples
sample_baseline_predictions(TEST_EXAMPLE_COUNT, one_hot_matrix_to_word)

#conversion from arpabet to ipa, stress markers 1, 2 stayed as numbers
def arpa_ipa (word):
    #removes ARPABet version and replaces with IPA symbols
    #order matters, all the double letter phonemes first, otherwise wrong things would be replaced
    word = re.sub(r'AA', "ɑ" , word)
    word = re.sub(r'AE', "æ", word)
    word = re.sub(r'AH', "ʌ", word)
    word = re.sub(r'AW', "aʊ", word)
    word = re.sub(r'AO', "ɔ", word)
    word = re.sub(r'AY', "aɪ", word)
    word = re.sub(r'EH', "ɛ", word)
    word = re.sub(r'ER', "ɝ", word)
    word = re.sub(r'EY', "eɪ", word)
    word = re.sub(r'IH', "ɪ", word)
    word = re.sub(r'IX', "ɨ", word)
    word = re.sub(r'IY', "i", word)
    word = re.sub(r'OW', "oʊ", word)
    word = re.sub(r'OY', "ɔɪ", word)
    word = re.sub(r'UH', "ʊ", word)
    word = re.sub(r'UW', "u", word)
    word = re.sub(r'CH', "tʃ", word)
    word = re.sub(r'DH', "ð", word)
    word = re.sub(r'DX', "ɾ", word)
    word = re.sub(r'EL', "l̩", word)
    word = re.sub(r'EM', "m̩", word)
    word = re.sub(r'SH', "ʃ", word)
    word = re.sub(r'TH', "θ", word)
    word = re.sub(r'ZH', "ʒ", word)
    word = re.sub(r'WH', "ʍ", word)
    word = re.sub(r'EN', "n̩", word)
    word = re.sub(r'NG', "ŋ", word)
    word = re.sub(r'HH', "h", word)
    word = re.sub(r'JH', "dʒ", word)
    word = re.sub(r'K', "k", word)
    word = re.sub(r'L', "l", word)
    word = re.sub(r'M', "m", word)
    word = re.sub(r'N', "n", word)
    word = re.sub(r'P', "p", word)
    word = re.sub(r'Q', "ʔ", word)
    word = re.sub(r'R', "ɹ", word)
    word = re.sub(r'S', "s", word)
    word = re.sub(r'T', "t", word)
    word = re.sub(r'F', "f", word)
    word = re.sub(r'G', "ɡ", word)
    word = re.sub(r'V', "v", word)
    word = re.sub(r'W', "w", word)
    word = re.sub(r'Y', "j", word)
    word = re.sub(r'Z', "z", word)
    word = re.sub(r'B', "b", word)
    word = re.sub(r'D', "d", word)
    word = re.sub(r"0", "", word)
    word = re.sub(r" ", "", word)
    return word

#set the primary and secondary stress markers before the phoneme
def stress_markers( word):
    if "1" in word:
        #find where the stress marker was, delete it and place the sign before the phoneme it was after
        position = word.find("1")
        word = re.sub( r"1", "", word)
        word = word[0:position - 1] + "'" + word[(position - 1):]
    if "2" in word:
        position = word.find("2")
        word = re.sub( r"2", "", word)
        word = word[0:position - 1] + "ˌ" + word[(position - 1):]
    return word

#taking inputs and making predictions by using the testing model and the saved weight matrices
def predict_based_on_input():
    word = input("What is your input word? ")
    word = word.upper()
    word_matrix = np.zeros((max_char_seq, n_character_tokens))
    for t, char in enumerate(word):
        word_matrix[t, :] = char_to_1_hot(char)
    #reshape input to process (batch, word length, classes) so that the prediction works
    word_matrix = word_matrix.reshape(1, 20, 26)
    pred_phon_seq = predict_baseline(word_matrix, testing_encoder_model, testing_decoder_model)
    print(word, "-->", pred_phon_seq)
    print("Phonetic transcription in IPA : ", stress_markers(arpa_ipa(pred_phon_seq)))


#keep on asking for input unless declined
def ask_for_more():
    state = True
    while state== True:
        answer = input("Do you want to transcribe a word? y/n ")
        if answer == "y":
            predict_based_on_input()
            state = True
        elif answer == "n":
            print("Goodbye")
            state = False
        else:
            print("You can only answer with y or n.\nTry again. ")
            state = True

ask_for_more()

'''
Perfect Accuracy :  49.10948027981632 %  for "model_en_de_plot.hdf5", num_epochs : 30, time: ~ 1 h 
Perfect Accuracy :  60.07038324535428 % for model_en_de_dropout.hdf5, time: 02:21:14.79, min_loss: 0.09145, num_epochs: 64 
Perfect Accuracy :  48.63310587528432 %  for en_de_model3.hdf5, time : 00:55:51.94, min_loss:0.13560, num_epochs = 30 (no dropout, learning rate= 0.001)
Perfect Accuracy :  59.697008712072446 % for en_de_model4_onlydropout.hdf5, time:01:59:59.58, epochs:57 (lr default, dropout 0.2)
Perfect Accuracy:   58.41380198274752 %, for en_de_model5_05.hdf5, loss = 0.09267, epochs = 94, time :03:28:34.03, (dropout 0.5, lr default)
Perfect Accuracy :  60.11759151967727 % for model6_ende_dropout03.hdf5, loss = 0.08894, time = 02:28:52.73, epochs = 75 (dropout 0.3, lr default)
Perfect Accuracy :  59.62405046993691 % for model7_en_de03_0001.hdf5 , loss = 0.08940 , time : 02:23:28.62 , epochs = 67 (dropout 0.3, lr 0.001)
Perfect Accuracy : 0%, model_ende8.hdf5 , loss=  1.57055, 00:24:28.27, epochs 12, (dropout 0.3, lr 0.1)
'''

