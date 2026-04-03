import numpy as np
import regex
from collections import defaultdict

def count_pairs(word_counts):
    pair_counts = defaultdict(int)
    for word in word_counts:
        for i in range(len(word)-1):
            pair_key = word[i], word[i+1]
            pair_counts[pair_key] += 1
    return(pair_counts)

def merge_max(word_counts, max_pair, new_idx):
    new_counts = defaultdict(int)
    for word in word_counts:
        new_word = word
        for i in range(len(word)-1):
            if word[i] == max_pair[0] and word[i+1] == max_pair[1]:
                # pair in word
                new_word = word[:i] + (new_idx,) + word[i+2:]
        new_counts[new_word] = word_counts[word]
    return new_counts

class BPETokenizer:
    def __init__(self):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = {x: bytes([x]) for x in range(256)}
        self.merges = []
        pass
        
    def train(self, input_path: str, 
              vocab_size: int, 
              special_tokens: list[str],
              **kwargs) -> tuple:
        with open(input_path, 'r') as file:
            doc = file.read()

        if type(doc) == list:
            doc = '\n'.join(doc) # merge if list

        # expand vocab with special tokens
        for sp in special_tokens:
            self.vocab[max(self.vocab) + 1] = sp

        # break str into words
        words = regex.finditer(self.PAT, doc)

        # count words
        word_counts = defaultdict(int)
        for word in words:
            sub_word = tuple(word.group().encode())
            word_counts[sub_word] += 1

        while len(self.vocab) <= vocab_size:
            pair_counts = count_pairs(word_counts)
            max_pair = max(pair_counts, key=pair_counts.get)
            self.vocab[max(self.vocab) + 1] = max_pair
            self.merges.append(max_pair)
            word_counts = merge_max(word_counts, max_pair, 
                                    new_idx=max(self.vocab))

        return((self.vocab, self.merges))

# doc = ['low low low low low', 
#        'lower lower widest widest widest',
#        'newest newest newest newest newest newest', 
#        'hello! こんにちは!']

# bpe_tokenizer = BPETokenizer()
# print(bpe_tokenizer.train('./tests/fixtures/corpus.en', 1000, ['<|endoftext|>']))