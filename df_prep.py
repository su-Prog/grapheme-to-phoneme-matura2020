import pandas as pd
import json
import numpy as np
from numpy.core.defchararray import isalpha

data = open("index.json", "r")
read_data = data.read()
# from json to dictionary

dict_obj = json.loads(read_data)

# from dictionary to pandas dataframe
df = pd.DataFrame()
df['Word'] = dict_obj.keys()
df['Phonemes'] = dict_obj.values()

#check if words only contain letters
info = []
def not_only_letters(df, info):
   for index, row in df.iterrows():
        status = isalpha(row['Word'])
        info.append(status)
   return info

not_only_letters(df, info)
#new column indicates if words only contain letters
df['Only_letters_in_words'] = info

# delete rows with words containing other characters than letters
df = df.query("Only_letters_in_words != False")

# lose the Only_letters_in _words column
df = df.drop(['Only_letters_in_words'], axis = 1)

# make the words uppercase => easier for alignment
df['Word'] = df['Word'].str.upper()

#make ascii versions of word and phonemes
ascii_word = []
ascii_phonemes = []

def ascii_columns(df,old_column, new_column):
   for index, row in df.iterrows():
        ascii_values = [ord(i) for i in row[old_column]]
        new_column.append(ascii_values)
   return new_column

df['ASCII Word'] = ascii_columns(df, 'Word', ascii_word)
df['ASCII Phonemes'] = ascii_columns(df, 'Phonemes', ascii_phonemes)

# shuffle data
df = df.reindex(np.random.permutation(df.index))

#put the data in pickle
df.to_pickle('cmu_df.pickle')