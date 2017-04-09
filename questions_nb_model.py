uillinois_q_set_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/uillinois_labelled_question_set.txt"


import pandas
from nlp import train_naive_bayes


def read_uillinois_q_set(path, granular_target=True):
    questions = []
    with open(path, 'rb') as f:
        for rec in f.readlines():
            if granular_target:
                questions.append({
                    'target': rec.split(' ')[0],
                    'text': " ".join(rec.split(' ')[1:]).lower().rstrip()})
            else:
                questions.append({
                    'target': rec.split(' ')[0].split(":")[0],
                    'text': " ".join(rec.split(' ')[1:]).lower().rstrip()})

    return pandas.DataFrame(questions)


if __name__ == "__main__":
    # Train Naive Bayes models for classifying the U-Illinois
    # labelled question set
    uillinois_q_set_granular_target = read_uillinois_q_set(uillinois_q_set_path)
    uillinois_q_set_coarse_target = read_uillinois_q_set(uillinois_q_set_path, granular_target=False)

    nb_uillinois_fine_grained = train_naive_bayes(uillinois_q_set_granular_target)
    nb_uillinois_coarse_grained = train_naive_bayes(uillinois_q_set_coarse_target)

    # TODO: Save
    