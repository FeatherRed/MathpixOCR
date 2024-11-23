class Basic_Model:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def train_batch(self, image, caption, length):
        raise NotImplementedError