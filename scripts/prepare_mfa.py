"""
Prepare MELD Subset for Montreal Forced Aligner (MFA).

Creates a mirrored directory structure containing ONLY the .wav files 
selected by the dataset subsampling (e.g. 1500 train, 200 dev) along 
with the matching .txt transcript files required by MFA.
"""
import os
import argparse
import shutil
from pathlib import Path
import sys

# Add project root to sys.path so we can import ConflictNet's dataset class
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.datasets import MELDDataset

def prepare_subset(split, root, max_samples, output_dir):
    print(f"Preparing {split} split (max_samples={max_samples})...")
    # Initializing MELDDataset automatically runs the stratified subsampling algorithm
    dataset = MELDDataset(root=root, split=split, max_samples=max_samples)
    
    out_dir = Path(output_dir)
    count = 0
    
    # We iterate over the final subsampled items
    for item in dataset.items:
        wav_path = Path(item["wav_path"])
        text = item["text"]
        
        # Calculate relative path to maintain directory structure
        # (ConflictNet's _textgrid_path_from_wav expects the textgrid folder to mirror the audio folder)
        try:
            rel_path = wav_path.relative_to(Path(root))
        except ValueError:
            rel_path = Path(split) / wav_path.name
            
        target_wav = (out_dir / rel_path).with_suffix(".wav")
        target_txt = target_wav.with_suffix(".txt")
        
        # Create parent directories
        target_wav.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy or convert file
        if wav_path.exists():
            if wav_path.suffix.lower() == ".mp4":
                try:
                    import torchaudio
                    from data.datasets import load_audio
                    # load_audio handles torchaudio -> soundfile -> ffmpeg fallback
                    waveform = load_audio(str(wav_path), target_sr=16000)
                    if isinstance(waveform, dict):
                        # if for some reason we hit the .pt feature cache, this shouldn't happen 
                        # for MFA preparation, but guard just in case
                        print(f"Skipping {wav_path} because load_audio returned precomputed dict")
                    else:
                        torchaudio.save(str(target_wav), waveform, 16000)
                except Exception as e:
                    print(f"Error converting {wav_path}: {e}")
            else:
                shutil.copy2(wav_path, target_wav)
        else:
            print(f"Warning: audio file not found: {wav_path}")
            
        # Write transcript txt file
        with open(target_txt, "w", encoding="utf-8") as f:
            f.write(text.strip())
            
        count += 1
        
    print(f"Done! Prepared {count} files for {split} split in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MELD subset for MFA alignment")
    parser.add_argument("--meld_root", type=str, required=True, help="Path to MELD.Raw")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/mfa_input", help="Output directory for MFA data")
    parser.add_argument("--train_samples", type=int, default=1500)
    parser.add_argument("--dev_samples", type=int, default=200)
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prepare_subset("train", args.meld_root, args.train_samples, args.output_dir)
    prepare_subset("dev", args.meld_root, args.dev_samples, args.output_dir)
    
    print("\n" + "="*60)
    print("✅ MFA PREPARATION COMPLETE")
    print("="*60)
    print("\nNext steps on Kaggle (in a CPU notebook):")
    print("1. Install MFA (takes a few minutes):")
    print("   !conda install -c conda-forge montreal-forced-aligner -y")
    print("\n2. Download pretrained English models:")
    print("   !mfa model download dictionary english_us_arpa")
    print("   !mfa model download acoustic english_us_arpa")
    print("\n3. Run the alignment:")
    print(f"   !mfa align {args.output_dir} english_us_arpa english_us_arpa /kaggle/working/meld_textgrids --clean --jobs 4")
    print("\n4. Zip the results and upload as a Kaggle Dataset:")
    print("   !zip -r meld_textgrids.zip /kaggle/working/meld_textgrids")
