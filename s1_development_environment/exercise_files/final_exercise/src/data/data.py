import numpy as np

def mnist():
    data = np.load("/home/glob/Documents/github/dtu_mlops/s1_development_environment/exercise_files/final_exercise/data/corruptmnist/train_0.npz")
    train = data["images"]
    train_labels = data["labels"]
    _ = data["allow_pickle"]

    data = np.load("/home/glob/Documents/github/dtu_mlops/s1_development_environment/exercise_files/final_exercise/data/corruptmnist/train_0.npz")
    test = data["images"]
    test_labels = data["labels"]
    _ = data["allow_pickle"]

    return train, train_labels, test, test_labels
