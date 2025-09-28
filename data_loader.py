import os
import re
from pathlib import Path
from tqdm import tqdm

# ================================
# URDU TO ROMAN URDU TRANSLITERATOR (Rekhta.org Style)
# ================================

def urdu_to_roman_urdu(urdu_text):
    """
    Convert Urdu string to Roman Urdu using Rekhta.org conventions.
    Handles diphthongs, common words, and poetic conventions.
    """
    if not urdu_text:
        return ""
    
    # Step 1: Normalize text
    text = urdu_text.strip()
    
    # Remove diacritics (tashkeel)
    TASHKEEL = 'ًٌٍَُِّْٰ'
    text = ''.join([c for c in text if c not in TASHKEEL])
    
    # Step 2: Handle diphthongs FIRST (critical for accuracy)
    # Replace common diphthong patterns
    text = re.sub(r'آ', 'aa', text)
    text = re.sub(r'ائی|ئی|یٗ', 'ai', text)  # Various 'ai' forms
    text = re.sub(r'اؤ|ؤ', 'au', text)
    text = re.sub(r'وی|وٗ', 'oi', text)  # Sometimes used in poetry
    
    # Step 3: Character-by-character mapping
    mapping = {
        'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's',
        'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
        'ڈ': 'd', 'ذ': 'z', 'ر': 'r', 'ڑ': 'rr', 'ز': 'z',
        'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z',
        'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f',
        'ق': 'q', 'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm',
        'ن': 'n', 'و': 'o', 'ہ': 'h', 'ھ': 'h', 'ء': '',
        'ی': 'i', 'ے': 'e', 'ئ': 'y', 'ا': 'a'
    }
    
    roman = []
    i = 0
    while i < len(text):
        char = text[i]
        
        # Handle special cases with context
        if char == 'ا':
            # Check for diphthongs (already handled above, but double-check)
            if i + 1 < len(text):
                next_char = text[i + 1]
                if next_char in 'وؤ':
                    roman.append('au')
                    i += 2
                    continue
                elif next_char in 'یئ':
                    roman.append('ai')
                    i += 2
                    continue
            roman.append('a')
        elif char == 'و':
            # In poetry, 'و' is usually 'o' (not 'v' or 'u')
            roman.append('o')
        elif char == 'ی':
            # Final 'ی' is usually 'i', medial can be 'y' or 'i'
            if i == len(text) - 1:
                roman.append('i')
            elif i == 0 or (i > 0 and text[i - 1] in ' '):
                roman.append('y')
            else:
                roman.append('i')
        elif char == 'ہ':
            # Final 'ہ' often silent in Roman Urdu
            if i == len(text) - 1:
                roman.append('')  # Silent
            else:
                roman.append('h')
        else:
            # Use mapping, fallback to original if not found
            roman.append(mapping.get(char, char.lower()))
        
        i += 1
    
    roman_str = ''.join(roman)
    
    # Step 4: Post-process common words and errors
    # Fix frequent transliteration errors
    corrections = {
        'dl': 'dil', 'jae': 'jay', 'nh': 'nah', 'krna': 'karna',
        'hain': 'hain', 'hai': 'hai', 'thaa': 'tha', 'gaii': 'gayi',
        'rahaa': 'raha', 'chlaa': 'chala', 'milaa': 'mila'
    }
    
    for wrong, correct in corrections.items():
        roman_str = roman_str.replace(wrong, correct)
    
    # Clean up multiple spaces and leading/trailing spaces
    roman_str = re.sub(r'\s+', ' ', roman_str).strip()
    
    # Remove leading/trailing non-alphanumeric chars (except spaces)
    roman_str = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', roman_str)
    
    return roman_str if roman_str else "a"  # Avoid empty strings

# ================================
# DATA LOADER
# ================================

def load_urdu_roman_dataset(root_dir: str):
    """
    Load Urdu → Roman Urdu pairs from the urdu_ghazals_rekhta dataset.
    
    Args:
        root_dir (str): Path to the root dataset folder (e.g., 'C:/Users/HP-OMEN/Desktop/dataset/dataset')
    
    Returns:
        List[Tuple[str, str]]: List of (urdu_line, roman_urdu_line)
    """
    dataset = []
    
    # Look for poet directories (ahmad-faraz, allama-iqbal, etc.)
    poet_dirs = Path(root_dir).glob('*')
    
    for poet_dir in tqdm(list(poet_dirs), desc="Processing poet folders"):
        if not poet_dir.is_dir() or poet_dir.name.startswith('.'):
            continue
        
        # Look for 'ur' subdirectory in each poet folder
        urdu_dir = poet_dir / 'ur'
        if not urdu_dir.exists() or not urdu_dir.is_dir():
            print(f"⚠️ No 'ur' folder found in {poet_dir.name}")
            continue
            
        print(f"📁 Processing {poet_dir.name}...")
        
        # Process all files in the 'ur' directory
        urdu_files = list(urdu_dir.glob('*'))
        if not urdu_files:
            print(f"⚠️ No files found in {urdu_dir}")
            continue
            
        for urdu_file in urdu_files:
            if urdu_file.is_file() and not urdu_file.name.startswith('.'):
                try:
                    # Read Urdu file (UTF-8)
                    with open(urdu_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            continue
                        urdu_lines = content.split('\n')
                    
                    # Process each line
                    for line in urdu_lines:
                        line = line.strip()
                        if not line or line.startswith('#') or len(line) < 3:
                            continue
                            
                        roman_line = urdu_to_roman_urdu(line)
                        # Skip if transliteration failed or too short
                        if roman_line and len(roman_line) > 2 and not roman_line.isdigit():
                            dataset.append((line, roman_line))
                            
                except Exception as e:
                    print(f"❌ Error reading {urdu_file}: {e}")
                    continue
    
    print(f"✅ Loaded {len(dataset)} Urdu → Roman Urdu pairs.")
    return dataset

# ================================
# SAVE DATASET (Optional)
# ================================

def save_dataset(dataset, output_path: str):
    """Save as TSV: urdu<tab>roman_urdu"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for urdu, roman in dataset:
            # Skip pairs where roman is suspiciously short compared to urdu
            if len(roman) >= len(urdu) * 0.3:  # Basic sanity check
                f.write(f"{urdu}\t{roman}\n")
    print(f"💾 Dataset saved to {output_path}")

# ================================
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    # Update this path to your actual dataset root
    ROOT_DIR = "E:/NLP ASSIGNMENT/dataset"  # ← Fixed path for your system
    
    print(f"🔍 Looking for data in: {ROOT_DIR}")
    
    # Check if directory exists
    if not Path(ROOT_DIR).exists():
        print(f"❌ Directory not found: {ROOT_DIR}")
        print("Please check the path and try again.")
        exit(1)
    
    # Load data
    data = load_urdu_roman_dataset(ROOT_DIR)
    
    # Show sample
    print("\n🔍 Sample pairs:")
    for i in range(min(5, len(data))):
        print(f"UR: {data[i][0]}")
        print(f"RO: {data[i][1]}\n")
    
    # Save (optional)
    if data:
        save_dataset(data, "urdu_to_roman.tsv")
        print(f"💾 Saved {len(data)} pairs to urdu_to_roman.tsv")
    else:
        print("❌ No data loaded. Please check your dataset structure.")