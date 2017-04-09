model_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/word_weights.pkl"

import sys
import pickle
import pandas
from collections import Counter
from nlp import get_word_weights, remove_punc, get_weight, count_grams_full


if __name__ == "__main__":
    input_path = sys.argv[1]
    df = pandas.read_csv(input_path)
    
    # Get word and n-gram weights based on counts
    questions = pandas.Series(df['question1'].tolist() + df['question2'].tolist()).astype(str)
    questions = [remove_punc(q).lower() for q in questions]
    eps = 500 
    words = (" ".join(questions)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count, eps=eps) for word, count in counts.items()}
    weights_2gram = count_grams_full(df, 2)
    weights_3gram = count_grams_full(df, 3)

    ngram_weights = {1: weights, 2: weights_2gram, 3: weights_3gram}
    with open(model_path, 'wb') as model_file:
        pickle.dump(ngram_weights, model_file, protocol=pickle.HIGHEST_PROTOCOL)
