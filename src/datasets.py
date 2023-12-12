import torch
import glob 
import random
import torch.distributed as dist
import os
import numpy as np

class PreTokenizedDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, cache_dir):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        # self.vocab_source = vocab_source
        self.cache_dir = cache_dir

    def _get_shard_filenames(self) -> list[str]:
        pretokenized_data_relative_path = f"pretokenized_data/tok{self.vocab_size}"
        bin_dir = os.path.join(self.cache_dir, pretokenized_data_relative_path)
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir}!"
        return shard_filenames

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        shard_filenames = self._get_shard_filenames()
        # train/test split. let's use only shard 0 for test split, rest train
        # TODO: make it configurable
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                # drop the last partial batch
                num_batches = (len(m) // self.max_seq_len) - 1
                assert num_batches > 0, "this shard is way too small? investigate."
                idxs = list(range(num_batches))
                rng.shuffle(idxs)
                for idx in idxs:
                    start = idx * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PreTokenizedDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y