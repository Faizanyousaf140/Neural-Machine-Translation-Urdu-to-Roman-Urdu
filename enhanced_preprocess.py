import os
import re
import pickle
from collections import Counter
from tqdm import tqdm
import unicodedata

# Special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

def normalize_urdu_text(text):
    """
    Clean and normalize Urdu text according to project requirements.
    - Remove diacritics (tashkeel)
    - Normalize characters
    - Remove extraneous punctuation
    """
    if not text:
        return ""
    
    # Step 1: Remove diacritics (tashkeel) - critical for Urdu
    TASHKEEL = 'ًٌٍَُِّْٰۖۗۘۙۚۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩۪ۭ۫۬ۮۯ'
    text = ''.join([c for c in text if c not in TASHKEEL])
    
    # Step 2: Normalize Unicode characters
    text = unicodedata.normalize('NFC', text)
    
    # Step 3: Remove extraneous punctuation but keep essential ones
    # Keep: space, period, comma, question mark, exclamation
    # Remove: brackets, quotes, special symbols
    text = re.sub(r'[\[\](){}""''«»‹›]', '', text)
    text = re.sub(r'[،؛]', '،', text)  # Normalize Arabic punctuation
    
    # Step 4: Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 5: Remove lines that are too short or contain only numbers/symbols
    if len(text) < 3 or text.isdigit() or not any(c.isalpha() for c in text):
        return None
    
    return text

def load_tsv_dataset(tsv_path):
    """Load and clean dataset from TSV file: urdu<tab>roman_urdu"""
    pairs = []
    skipped = 0
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading and cleaning data"):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            urdu, roman = parts
            
            # Clean Urdu text
            urdu_clean = normalize_urdu_text(urdu)
            if urdu_clean is None:
                skipped += 1
                continue
            
            # Clean Roman text (basic)
            roman_clean = roman.strip()
            if not roman_clean or len(roman_clean) < 3:
                skipped += 1
                continue
            
            # Filter by length
            if 3 <= len(urdu_clean) <= 100 and 3 <= len(roman_clean) <= 100:
                pairs.append((urdu_clean, roman_clean))
            else:
                skipped += 1
    
    print(f"✅ Loaded {len(pairs)} pairs after cleaning.")
    print(f"❌ Skipped {skipped} pairs due to quality issues.")
    return pairs

def build_char_vocab(texts, min_freq=1):
    """Build character vocabulary from list of strings."""
    counter = Counter()
    for text in texts:
        counter.update(list(text))
    
    # Keep chars with frequency >= min_freq
    chars = [c for c, freq in counter.items() if freq >= min_freq]
    
    # Add special tokens
    vocab = {
        PAD_TOKEN: 0,
        SOS_TOKEN: 1,
        EOS_TOKEN: 2,
        UNK_TOKEN: 3
    }
    
    # Sort characters for consistent vocabulary
    for idx, char in enumerate(sorted(chars), start=4):
        vocab[char] = idx
    
    # Inverse vocab
    idx2char = {idx: char for char, idx in vocab.items()}
    return vocab, idx2char

def tokenize_char(text, vocab):
    """Convert text to list of indices using char vocab."""
    return [vocab.get(c, vocab[UNK_TOKEN]) for c in text]

def numericalize_dataset(pairs, src_vocab, tgt_vocab):
    """Convert (urdu, roman) pairs to numerical sequences."""
    src_seqs = []
    tgt_seqs = []
    
    for urdu, roman in tqdm(pairs, desc="Numericalizing"):
        # Source: Urdu (no special tokens for encoder)
        src_seq = tokenize_char(urdu, src_vocab)
        
        # Target: Roman Urdu (add SOS and EOS for decoder)
        tgt_seq = (
            [tgt_vocab[SOS_TOKEN]] +
            tokenize_char(roman, tgt_vocab) +
            [tgt_vocab[EOS_TOKEN]]
        )
        
        src_seqs.append(src_seq)
        tgt_seqs.append(tgt_seq)
    
    return src_seqs, tgt_seqs

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def analyze_dataset_quality(pairs, sample_size=100):
    """Analyze dataset quality and provide statistics."""
    print("\n📊 Dataset Quality Analysis:")
    print("=" * 50)
    
    # Length statistics
    urdu_lengths = [len(pair[0]) for pair in pairs]
    roman_lengths = [len(pair[1]) for pair in pairs]
    
    print(f"Total pairs: {len(pairs)}")
    print(f"Avg Urdu length: {sum(urdu_lengths)/len(urdu_lengths):.1f} chars")
    print(f"Avg Roman length: {sum(roman_lengths)/len(roman_lengths):.1f} chars")
    print(f"Max Urdu length: {max(urdu_lengths)} chars")
    print(f"Max Roman length: {max(roman_lengths)} chars")
    
    # Sample quality check
    print(f"\n🔍 Sample pairs (first {min(sample_size, len(pairs))}):")
    for i, (urdu, roman) in enumerate(pairs[:sample_size]):
        print(f"{i+1:3d}. UR: {urdu}")
        print(f"     RO: {roman}")
        print()

def main():
    # Paths
    TSV_PATH = "urdu_to_roman.tsv"
    OUTPUT_DIR = "processed_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Enhanced Urdu-Roman Preprocessing")
    print("=" * 50)
    
    # Load and clean data
    pairs = load_tsv_dataset(TSV_PATH)
    
    # Analyze quality
    analyze_dataset_quality(pairs)
    
    # Split: 50% train, 25% val, 25% test
    total = len(pairs)
    train_end = int(0.5 * total)
    val_end = int(0.75 * total)
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    print(f"\n📊 Data Split:")
    print(f"Train: {len(train_pairs)} pairs ({len(train_pairs)/total*100:.1f}%)")
    print(f"Val:   {len(val_pairs)} pairs ({len(val_pairs)/total*100:.1f}%)")
    print(f"Test:  {len(test_pairs)} pairs ({len(test_pairs)/total*100:.1f}%)")
    
    # Extract texts for vocab building (use only train set to avoid leakage)
    train_urdu_texts = [pair[0] for pair in train_pairs]
    train_roman_texts = [pair[1] for pair in train_pairs]
    
    # Build vocabularies
    print(f"\n🔤 Building vocabularies...")
    src_vocab, src_idx2char = build_char_vocab(train_urdu_texts)
    tgt_vocab, tgt_idx2char = build_char_vocab(train_roman_texts)
    
    print(f"Source (Urdu) vocab size: {len(src_vocab)}")
    print(f"Target (Roman Urdu) vocab size: {len(tgt_vocab)}")
    
    # Show vocabulary samples
    print(f"\n🔤 Sample Urdu characters: {list(src_vocab.keys())[:20]}")
    print(f"🔤 Sample Roman characters: {list(tgt_vocab.keys())[:20]}")
    
    # Numericalize all splits
    print(f"\n🔢 Processing datasets...")
    train_src, train_tgt = numericalize_dataset(train_pairs, src_vocab, tgt_vocab)
    val_src, val_tgt = numericalize_dataset(val_pairs, src_vocab, tgt_vocab)
    test_src, test_tgt = numericalize_dataset(test_pairs, src_vocab, tgt_vocab)
    
    # Save everything
    data = {
        'train': (train_src, train_tgt),
        'val': (val_src, val_tgt),
        'test': (test_src, test_tgt)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    save_vocab(src_vocab, os.path.join(OUTPUT_DIR, 'src_vocab.pkl'))
    save_vocab(tgt_vocab, os.path.join(OUTPUT_DIR, 'tgt_vocab.pkl'))
    save_vocab(src_idx2char, os.path.join(OUTPUT_DIR, 'src_idx2char.pkl'))
    save_vocab(tgt_idx2char, os.path.join(OUTPUT_DIR, 'tgt_idx2char.pkl'))
    
    print(f"\n💾 All processed data saved to '{OUTPUT_DIR}/'")
    
    # Show sample
    print(f"\n🔍 Sample numericalized pair (train):")
    print("UR (chars):", list(train_pairs[0][0]))
    print("UR (ids)  :", train_src[0][:20], "...")
    print("RO (chars):", ['<sos>'] + list(train_pairs[0][1]) + ['<eos>'])
    print("RO (ids)  :", train_tgt[0][:20], "...")

if __name__ == "__main__":
    main()
