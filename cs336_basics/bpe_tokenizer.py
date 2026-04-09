import numpy as np
import regex
from collections import defaultdict

def count_pairs(word_counts: dict[list[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word in word_counts:
        for i in range(len(word)-1):
            pair_key = (word[i], word[i+1])
            pair_counts[pair_key] += 1
    return pair_counts 

def merge_max(word_counts: dict[tuple[bytes, ...], int], max_pair: tuple[bytes, bytes]) -> dict[list[bytes], int]:
    new_counts: dict[list[bytes], int] = defaultdict(int)
    for word in word_counts:
        occurances = []
        for i in range(len(word)-1):
            if word[i] == max_pair[0] and word[i+1] == max_pair[1]:
                    # pair in word
                    occurances.append(i)
        new_word = []
        for i in range(len(word)):
            if i in occurances:
                new_word.append(b''.join(max_pair))
            elif i-1 in occurances: continue
            else:
                new_word.append(word[i])
        new_counts[tuple(new_word)] = word_counts[word]
    return new_counts

# count words
def get_word_counts(words: iter) -> dict[tuple[bytes, ...], int]:
    word_counts: dict[list[bytes], int] = defaultdict(int)
    for word in words:
        sub_word = tuple(bytes([x]) for x in word.group().encode())
        word_counts[sub_word] += 1
    return word_counts
        
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

        # expand vocab with special tokens
        for sp in special_tokens:
            self.vocab[max(self.vocab) + 1] = sp.encode()

        # break str into words
        words = regex.finditer(self.PAT, doc)
        word_counts = get_word_counts(words)

        sorted_counts = lambda x: sorted(x.items(), key=lambda item: item[1])
        # print(sorted_counts(word_counts))
        # print("============================")
        while len(self.vocab) < vocab_size:
            pair_counts = count_pairs(word_counts)
            # print(sorted_counts(pair_counts))
            # print("-----------------")
            max_pair = max(pair_counts, key=pair_counts.get)
            # print(max_pair)
            self.vocab[max(self.vocab) + 1] = max_pair
            self.merges.append(max_pair)
            word_counts = merge_max(word_counts, max_pair)
            
        return (self.vocab, self.merges)



bpe_tokenizer = BPETokenizer()
print(bpe_tokenizer.train('./tests/fixtures/corpus.en', 260, ['<|endoftext|>']))
# print(bpe_tokenizer.train('./tests/fixtures/test_doc.txt', 260, ['<|endoftext|>']))