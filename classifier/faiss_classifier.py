class FaissClassifer:
    def __init__(self) -> None:
        super().__init__()

        index = load_index()

    def classify(image):
        # align
        # calc embedding
        # use index to find nearest 50 embeddings
        # pick only embeddings within a threshold
        # from those -> choose label for with the most embeddings
        # get persons name
        return  # label

    def classify_with_surroundings():
        return  # label_name, surrounding_embeddings

    def add_person(image, label: str):
        pass

    def load_index():
        pass
