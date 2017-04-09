train_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train.csv"
model_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/lda_model.pkl"


import pickle
import pandas
from functools import partial
from nlp import train_lda, construct_doc_list


if __name__ == "__main__":
    df = pandas.read_csv(train_path)

    n_lda_topics = 20
    print "Starting LDA modelling!"

    doc_list_lda_train = list(construct_doc_list(df))
    lda_model, id2word_dictionary, word2idx_dictionary, topics = \
        train_lda(n_lda_topics,
                  documents=doc_list_lda_train)

    print "Finished model training!"
    print "Saving output"

    model = {
        "lda_model": lda_model,
        "id2word_dict": id2word_dictionary,
        "word2idx_dict": word2idx_dictionary,
        "topics": topics
    }

    # df.to_csv(train_pred_path, index=False)
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
