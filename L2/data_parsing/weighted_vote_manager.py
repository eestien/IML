from collections import defaultdict

# write validation accuracy of each model
POSTER_ACC = 0.8
TITLE_ACC = 0.8
SINOPSIS_ACC = 0.8


# takes probs and classes_ (p,c) of related model for one sample and defines final class
def classify(title_probs_classes, sin_probs_classes, poster_probs_classes):
    # need to align classes with classes_
    classes = title_probs_classes[1]
    title_dict = _create_dict(title_probs_classes)
    sin_probs_dict = _create_dict(sin_probs_classes)
    poster_probs_dict = _create_dict(poster_probs_classes)
    voting_bucket = defaultdict()
    for c in classes:
        voting_bucket[c] += title_dict[c] * TITLE_ACC
        voting_bucket[c] += sin_probs_dict[c] * SINOPSIS_ACC
        voting_bucket[c] += poster_probs_dict[c] * POSTER_ACC

        votes_to_classes = {v: k for k, v in voting_bucket.items()}
        max_class_value = max(votes_to_classes.keys())
        resulting_class = votes_to_classes[max_class_value]

    return resulting_class

def _create_dict(pc_tuple):
    d = {}
    for item in pc_tuple:
        d[item[1]] = item[0]
    return d
