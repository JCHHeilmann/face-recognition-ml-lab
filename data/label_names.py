import pickle


class LabelNames:
    def __init__(self, pickle_file):
        with open(pickle_file, "rb") as pickle_file:
            self.dictionary = pickle.load(pickle_file)

    def read_from_pickle(self, folder_tag):
        # returns a name from dictionary
        # for example read_from_pickle("45") returns Bruce_Lee
        folder_tag = str(int(folder_tag))
        folder_tag = folder_tag.zfill(7)
        return self.dictionary[folder_tag]

    def add_name(self, name, label):
        self.dictionary[str(label)] = name
