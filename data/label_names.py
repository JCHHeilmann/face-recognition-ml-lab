import pickle

class label_names:

    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as pickle_file:
            self.dictionary = pickle.load(pickle_file)

    def read_from_pickle(self, folder_tag):
        #returns a name from dictionary
        #for example read_from_pickle("0000045") returns Bruce_Lee
        return self.dictionary[folder_tag]