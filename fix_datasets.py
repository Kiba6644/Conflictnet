import torch
import os

with open("data/datasets.py", "r") as f:
    content = f.read()

target = "\"audio_np\": audio.numpy(),"
replacement = "\"audio_np\": audio.numpy() if isinstance(audio, torch.Tensor) else None,"
content = content.replace(target, replacement)

collate_target = """        for b in batch:
            aug_np = augmentor(b["audio_np"])
            # Create a shallow copy and update so we don't mutate the original dict
            b_new = {**b, "audio_np": aug_np, "audio": torch.tensor(aug_np, dtype=torch.float32)}
            batch_out.append(b_new)"""
collate_replacement = """        for b in batch:
            if b.get("audio_np") is not None:
                aug_np = augmentor(b["audio_np"])
                b_new = {**b, "audio_np": aug_np, "audio": torch.tensor(aug_np, dtype=torch.float32)}
                batch_out.append(b_new)
            else:
                batch_out.append(b)"""
                
content = content.replace(collate_target, collate_replacement)

with open("data/datasets.py", "w") as f:
    f.write(content)
print("Fix applied")

