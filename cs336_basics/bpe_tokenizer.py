import numpy as np
import regex
from collections import defaultdict
from itertools import chain
import pickle

def count_pairs(word_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word in word_counts:
        for i in range(len(word)-1):
            pair_key = (word[i], word[i+1])
            pair_counts[pair_key] += word_counts[word]
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
def get_word_counts(words: iter) -> dict[tuple[bytes], int]:
    word_counts: dict[list[bytes], int] = defaultdict(int)
    for word in words:
        sub_word = tuple(bytes([x]) for x in word.group().encode())
        word_counts[sub_word] += 1
    return word_counts
        
def get_words(pattern: str, docs: list[str]):
    return chain.from_iterable(regex.finditer(pattern, doc) for doc in docs)

sorted_counts = lambda x: sorted(x.items(), key=lambda item: item[1],  reverse=True)
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

        docs = [doc]
        # expand vocab with special tokens
        if len(special_tokens) > 0: 
            for sp in special_tokens:
                self.vocab[max(self.vocab) + 1] = sp.encode()
                docs = list(chain.from_iterable([doc.split(sp) for doc in docs]))

        # break str into words
        words = get_words(self.PAT, docs)
        word_counts = get_word_counts(words)

        # print(sorted_counts(word_counts))
        # print("============================")
        while len(self.vocab) < vocab_size:
            pair_counts = count_pairs(word_counts)
            # print(sorted_counts(pair_counts))
            # print("-----------------")
            # max_pair = max(pair_counts, key=pair_counts.get)
            max_value = max(pair_counts.values())
            max_pair = [k for k, v in pair_counts.items() if v == max_value]
            max_pair = max(max_pair)
            # print(max_pair)
            self.vocab[max(self.vocab) + 1] = b''.join(max_pair)
            self.merges.append(max_pair)
            word_counts = merge_max(word_counts, max_pair)
            
        return (self.vocab, self.merges)
    
    def save_params(self, file_name: str) -> None:
        with open(file_name, 'wb') as f:
            meta = "(vocab, merges)"
            data = (self.vocab, self.merges)
            payload = (meta, data)
            pickle.dump(payload, f)

    def load_params(self, file_name: str) -> None: 
        with open(file_name, 'rb') as f:
            payload = pickle.load(f)
            meta, data = payload
            if meta == "(vocab, merges)":
                return data
            else:
                raise ValueError("`file_name` metadata invalid")


def main() -> None: 
    import pprint
    import cProfile

    fixture_name = 'test_doc'
    max_vocab_size = 260

    with cProfile.Profile() as pr:
        bpe_tokenizer = BPETokenizer()
        # v, m = bpe_tokenizer.train('./tests/fixtures/german.txt', 1000, ['<|endoftext|>'])
        print(f'./tests/fixtures/{fixture_name}.txt')
        v, m = bpe_tokenizer.train(f'./tests/fixtures/{fixture_name}.txt', max_vocab_size, ['<|endoftext|>', '<|endofsentense|>'])

    bpe_tokenizer.save_params(f'./tests/fixtures/params/{fixture_name}_params_{max_vocab_size}.bin')

    pprint.pprint(v)
    pprint.pprint(m)
    pr.dump_stats(f'./tests/fixtures/cprofile_reports/{fixture_name}_{max_vocab_size}.perf')

if __name__  == "__main__":
    main()