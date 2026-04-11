import numpy as np
import regex
from collections import defaultdict
from itertools import chain
import pickle
from loguru import logger
from pretokenization_example import find_chunk_boundaries

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
        self.open_parallel = False
        self.n_parallel = None
        pass

    def open_file(self, input_path: str, n_chunks:int=4, special_tokens:bytes=b''):
        chunks = []
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, desired_num_chunks=4,
                                                    split_special_token=special_tokens)
        
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)

        return chunks

    def train(self, input_path: str, 
              vocab_size: int, 
              special_tokens: list[str],
              **kwargs) -> tuple:
        
        if not self.open_parallel: 
            with open(input_path, 'r') as file:
                doc = file.read()
            docs = [doc]
        else: 
            docs = self.open_file(input_path, 
                                  n_chunks=4,
                                  special_tokens=special_tokens[0].encode())

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

    def encode(self) -> None:
        raise NotImplementedError


def main() -> None: 
    import pprint
    import cProfile

    path: str = './tests/fixtures'
    param_path = './tests/fixtures/params'
    cprofile_path = './tests/fixtures/cprofile_reports'

    fixture_name = 'test_doc.txt'
    max_vocab_size = 260

    _train_file: str = f'{path}/{fixture_name}'
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.open_parallel = True
    v, m = bpe_tokenizer.train(_train_file, max_vocab_size, ['<|endoftext|>', '<|endofsentence|>'])
    pprint.pprint(v)
    pprint.pprint(m)

    # with cProfile.Profile() as pr:
    #     bpe_tokenizer = BPETokenizer()
    #     logger.info(f'Training on file {_train_file} ...')
    #     v, m = bpe_tokenizer.train(_train_file, max_vocab_size, ['<|endoftext|>', '<|endofsentence|>'])
                            # special_tokens=[b'<|endoftext|>', b'<|endofsentence|>'])

    # _param_path = f'{param_path}/{fixture_name}_params_{max_vocab_size}.bin'
    # logger.info(f'Save params {_param_path}')
    # bpe_tokenizer.save_params(_param_path)

    # logger.info(f"Vocab size: {len(v)}")
    # logger.info(f"Merges size: {len(m)}")
    # # pprint.pprint(v)
    # # pprint.pprint(m)


    # _cprofile_path = f'{cprofile_path}/{fixture_name}_{max_vocab_size}.perf'
    # logger.info(f'Saving cprofile stats to {_cprofile_path} ...')
    # pr.dump_stats(_cprofile_path)

if __name__  == "__main__":
    main()
    logger.info("Goodbye.")