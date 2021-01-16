import pickle

def map_to_name(folder_tag):
    with open('data.p', 'rb') as fp:
        d = pickle.load(fp)

        return (d[folder_tag])