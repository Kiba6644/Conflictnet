import random
import bisect
from torch.utils.data import Sampler, ConcatDataset
import torch.distributed as dist

def _get_item_meta(dataset, idx):
    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect.bisect_right(dataset.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - dataset.cumulative_sizes[dataset_idx - 1]
        return _get_item_meta(dataset.datasets[dataset_idx], sample_idx)
    else:
        if hasattr(dataset, "items"):
            item = dataset.items[idx]
            conv_id = item.get("conversation_id", item.get("dialogue_id", f"uid_{idx}"))
            turn_index = item.get("turn_index", 0)
            return conv_id, turn_index
        else:
            return f"uid_{idx}", 0

class DialogueDistributedBatchSampler(Sampler):
    """
    A custom BatchSampler that streams turns from conversations sequentially.
    
    Instead of randomly shuffling all utterances (which destroys the dialogue history
    in the ContextCache), this sampler groups utterances by conversation, shuffles
    the conversations, and streams them in parallel.
    
    In DDP environments, it coordinates across replicas so each rank receives
    a dedicated slice of the active conversations, and pads automatically to
    prevent synchronization hangs.
    """
    def __init__(self, dataset, batch_size, num_replicas=1, rank=0, drop_last=False, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Group by conversation_id
        self.conv_to_indices = {}
        for idx in range(len(dataset)):
            conv_id, turn_index = _get_item_meta(dataset, idx)
            if conv_id not in self.conv_to_indices:
                self.conv_to_indices[conv_id] = []
            self.conv_to_indices[conv_id].append((idx, turn_index))
            
        # Sort each conversation's turns chronologically
        for conv_id in self.conv_to_indices:
            self.conv_to_indices[conv_id].sort(key=lambda x: x[1])
            self.conv_to_indices[conv_id] = [x[0] for x in self.conv_to_indices[conv_id]]
            
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        conv_ids = list(self.conv_to_indices.keys())
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(conv_ids)
            
        # We maintain `global_batch_size` active conversations at any time
        global_batch_size = self.batch_size * self.num_replicas
        active_convs = []
        conv_idx = 0
        
        while len(active_convs) < global_batch_size and conv_idx < len(conv_ids):
            active_convs.append({
                "id": conv_ids[conv_idx],
                "turns": self.conv_to_indices[conv_ids[conv_idx]],
                "ptr": 0
            })
            conv_idx += 1
            
        while active_convs:
            global_batch = []
            next_active = []
            for conv in active_convs:
                global_batch.append(conv["turns"][conv["ptr"]])
                conv["ptr"] += 1
                
                # Keep conversation if it has more turns, otherwise queue a new one
                if conv["ptr"] < len(conv["turns"]):
                    next_active.append(conv)
                else:
                    if conv_idx < len(conv_ids):
                        next_active.append({
                            "id": conv_ids[conv_idx],
                            "turns": self.conv_to_indices[conv_ids[conv_idx]],
                            "ptr": 0
                        })
                        conv_idx += 1
                        
            # Pad global_batch if it's smaller than global_batch_size 
            # (Ensures all DDP replicas receive exactly `batch_size` samples)
            if len(global_batch) < global_batch_size:
                if self.drop_last:
                    break
                else:
                    pad_size = global_batch_size - len(global_batch)
                    if len(global_batch) > 0:
                        global_batch.extend([global_batch[-1]] * pad_size)
                    else:
                        break
                        
            # Slice the global batch for the current rank
            start_idx = self.rank * self.batch_size
            end_idx = start_idx + self.batch_size
            replica_batch = global_batch[start_idx:end_idx]
            
            if not replica_batch:
                break
                
            yield replica_batch
            active_convs = next_active
            
    def __len__(self):
        total_turns = sum(len(v) for v in self.conv_to_indices.values())
        global_batch_size = self.batch_size * self.num_replicas
        if self.drop_last:
            return total_turns // global_batch_size
        else:
            return (total_turns + global_batch_size - 1) // global_batch_size

