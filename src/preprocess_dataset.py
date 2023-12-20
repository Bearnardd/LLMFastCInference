import argparse
import os
import requests
from tqdm import tqdm
import json
import glob
import sentencepiece as spm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# TODO: change that with package
import sys
sys.path.append('..')
from tokenizer import Tokenizer

STEPS_SPLITTER = "|"
ARGS_SPLITTER = ":="


def download_file(url: str, fpath: str, chunk_size: int = 1024) -> None:
    """Helper function to download a file from a given url"""
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(fpath, "wb") as file, tqdm(
        desc=fpath,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_dataset(data_url: str, cache_dir: str) -> None:
    """Downloads dataset to cache_dir"""
    os.makedirs(cache_dir, exist_ok=True)
    # # TODO: fix nameing finding
    fpath = os.path.join(cache_dir, data_url.split("/")[-1])
    if not os.path.exists(fpath):
        download_file(url=data_url, fpath=fpath)
        os.system(f"tar -xzf {fpath} -C {cache_dir}")
    shard_filenames = glob.glob(os.path.join(cache_dir, "*.json"))
    print(f"Number of shards: {len(shard_filenames)}")
    # if os.path.exists(cache_dir):
    #     print(f"{cache_dir} already exists, skipping download...")
    #     # os.makedirs(cache_dir, exist_ok=True)
    # # if data_target_dir_name is None:
    #     # data_target_dir_name = data_url.split("/")[-1]
    # # data_filename = os.path.join(cache_dir, data_target_dir_name)
    # else:
        # os.makedirs(cache_dir, exist_ok=True)
        # print(f"Downloading {data_url} to {cache_dir}...")
        # download_file(data_url, cache_dir)
    # else:
    #     print(f"{cache_dir} already exists, skipping download...")
    #     return
    # unpack the tar.gz file into all the data shards (json files)
    # TODO support mort than just tar.gz
    # data_dir = os.path.join(cache_dir, "TinyStories_all_data")
    # if not os.path.exists(data_dir):
        # os.makedirs(data_dir, exist_ok=True)
    # print(f"Unpacking {cache_dir}...")
    # os.system(f"tar -xzf {cache_dir} -C {cache_dir}")
    # else:
        # print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such


# def download_dataset(data_url, cache_dir):
#     # TODO: change args
#     os.makedirs(cache_dir, exist_ok=True)
#     ds = load_dataset(data_url, cache_dir=cache_dir)

def train_vocab(vocab_size, cache_dir):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"
    tokenizer_path = os.path.join(cache_dir, "tokenizers")
    os.makedirs(os.path.join(tokenizer_path), exist_ok=True)
    # output file prefix path for sentencepiece
    prefix = os.path.join(tokenizer_path, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 1

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(cache_dir, "tiny.txt")
    # TODO: make it more general
    # data_dir = os.path.join(cache_dir, src_dir)
    # TODO: support noy only json
    shard_filenames = sorted(glob.glob(os.path.join(cache_dir, "*.json")))
    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")

    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")




def process_shard(args, vocab_size, cache_dir):
    shard_id, shard = args
    # TODO: make it work with default tokenizer
    model_path = os.path.join(cache_dir, f"tokenizers/tok{vocab_size}.model")
    enc = Tokenizer(model_path)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    pretokenized_data_target_dir = os.path.join(cache_dir, f"pretokenized_data/tok{vocab_size}")
    shard_basename = os.path.basename(shard)
    bin_basename = shard_basename.replace(".json", ".bin")
    tokenized_filename = os.path.join(pretokenized_data_target_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size, cache_dir):
    # TODO: support noy only json
    shard_filenames = sorted(glob.glob(os.path.join(cache_dir, "*.json")))
    # TODO: change that
    assert vocab_size > 0, "Vocab size must be positive!"
    # .bin files will be saved into tok{N} directory, create it once here
    bin_dir = os.path.join(cache_dir, f"pretokenized_data/tok{vocab_size}")
    # Should be already created right?
    os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size, cache_dir=cache_dir)
    # fun((0, shard_filenames[0]))
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Data has been pretokenized!.")


def process_args(args):
    args_dict = vars(args)
    processed_args = {}
    for arg in args_dict:
        # TODO: change that
        if arg == "steps" and args_dict[arg] is not None:
            steps = args_dict[arg].split(STEPS_SPLITTER)
            for step in steps:
                if ":" in step:
                    step, arg = step.strip().split(ARGS_SPLITTER)
                    processed_args[step] = arg 
                else:
                    processed_args[step] = None
        else:
            processed_args[arg] = args_dict[arg]
    return processed_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=str)
    parser.add_argument("--data_cache_dir", type=str, default="data")
    parser.add_argument("--data_target_dir_name", type=str, default=None)
    args = parser.parse_args()
    processed_args = process_args(args)
    if processed_args.get("download"):
        download_dataset(data_url=processed_args["download"], cache_dir=processed_args["data_cache_dir"])
    if processed_args.get("train_vocab"):
        train_vocab(vocab_size=int(processed_args.get("train_vocab")), cache_dir=processed_args["data_cache_dir"])
    # TODO: make it work without specifying vocab size
    if processed_args.get("pretokenize"):
        vocab_size = int(processed_args.get("pretokenize"))
        pretokenize(vocab_size=vocab_size, cache_dir=processed_args["data_cache_dir"])