#!/usr/bin/env python3
"""
Clean training data by removing @xmath placeholders from assistant responses (abstracts).

The training data has 59.4% of examples with @xmath placeholders in the abstracts,
which causes the model to learn to generate them. This script removes those placeholders.
"""

import json
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(desc + "...")
        return iterable

def clean_abstract(abstract):
    """
    Remove all placeholder patterns from abstract text.
    
    Removes:
    - @xmath (with digits): Math placeholders (e.g., @xmath0, @xmath1)
    - @xcite (with optional digits): Citation placeholders (e.g., @xcite, @xcite1)
    - @xref, @xeq, @xfig, @xtab, @xsec: Other cross-reference placeholders
    - Broken figure/table/section references: fig.[fig:...], tab.[tab:...], sec.[sec:...]
    
    All placeholders are removed entirely to keep abstracts clean.
    """
    # Remove @xmath followed by digits
    cleaned = re.sub(r'@xmath\d+', '', abstract, flags=re.IGNORECASE)
    # Remove @xcite (with optional digits)
    cleaned = re.sub(r'@xcite\d*', '', cleaned, flags=re.IGNORECASE)
    # Remove other @x* placeholders
    cleaned = re.sub(r'@x(?:ref|eq|fig|tab|sec)\d*', '', cleaned, flags=re.IGNORECASE)
    
    # Remove broken figure/table/section references
    # Pattern: fig.[fig:name] or fig [fig:name] or similar
    cleaned = re.sub(r'fig\.?\s*\[?\s*fig\s*:\s*\w+\s*\]?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'tab\.?\s*\[?\s*tab\s*:\s*\w+\s*\]?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'sec\.?\s*\[?\s*sec\s*:\s*\w+\s*\]?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\(eq\s*:\s*\w+\)', '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra spaces that might result
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Clean up spaces around punctuation
    cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
    cleaned = re.sub(r'([.,;:!?])\s+', r'\1 ', cleaned)
    # Clean up spaces at start/end of sentences
    cleaned = re.sub(r'\.\s+\.', '.', cleaned)  # Remove double periods
    return cleaned.strip()

def clean_training_data(input_file, output_file, dry_run=False):
    """
    Clean training data file by removing @xmath from assistant responses.
    
    Args:
        input_file: Path to input training data JSONL file
        output_file: Path to output cleaned training data JSONL file
        dry_run: If True, only report statistics without writing output
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    stats = {
        'total': 0,
        'cleaned': 0,
        'removed': 0,
        'xmath_count_before': 0,
        'xmath_count_after': 0,
        'skipped_empty': 0
    }
    
    print(f"Reading from: {input_path}")
    if not dry_run:
        print(f"Writing to: {output_path}")
    print()
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in tqdm(infile, desc="Processing"):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    stats['total'] += 1
                    
                    if 'messages' not in data:
                        # Keep non-message format as-is
                        if not dry_run:
                            outfile.write(line)
                        continue
                    
                    # Process messages
                    cleaned_messages = []
                    needs_cleaning = False
                    
                    for msg in data['messages']:
                        if msg.get('role') == 'assistant':
                            original = msg.get('content', '')
                            # Count all placeholder types
                            placeholder_patterns = [
                                r'@xmath\d+',
                                r'@xcite\d*',
                                r'@x(?:ref|eq|fig|tab|sec)\d*',
                                r'fig\.?\s*\[?\s*fig\s*:\s*\w+\s*\]?',
                                r'tab\.?\s*\[?\s*tab\s*:\s*\w+\s*\]?',
                                r'sec\.?\s*\[?\s*sec\s*:\s*\w+\s*\]?',
                                r'\(eq\s*:\s*\w+\)',
                            ]
                            
                            total_placeholders = sum(len(re.findall(pattern, original, re.IGNORECASE)) 
                                                    for pattern in placeholder_patterns)
                            xmath_count = len(re.findall(r'@xmath\d+', original, re.IGNORECASE))
                            
                            if total_placeholders > 0:
                                stats['xmath_count_before'] += xmath_count
                                cleaned_content = clean_abstract(original)
                                # Check if any placeholders remain
                                remaining = sum(len(re.findall(pattern, cleaned_content, re.IGNORECASE)) 
                                              for pattern in placeholder_patterns)
                                stats['xmath_count_after'] += remaining
                                
                                if len(cleaned_content.strip()) == 0:
                                    stats['skipped_empty'] += 1
                                    # Skip this example entirely if abstract becomes empty
                                    needs_cleaning = False
                                    break
                                
                                msg['content'] = cleaned_content
                                stats['cleaned'] += 1
                                stats['removed'] += total_placeholders
                                needs_cleaning = True
                        cleaned_messages.append(msg)
                    
                    # Only write if we didn't skip due to empty abstract
                    if needs_cleaning or stats['total'] <= stats['cleaned']:
                        data['messages'] = cleaned_messages
                        if not dry_run:
                            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    elif not dry_run:
                        # Write unchanged if no cleaning needed
                        outfile.write(line)
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    
    # Print statistics
    print("\n" + "="*80)
    print("CLEANING STATISTICS")
    print("="*80)
    print(f"Total examples processed: {stats['total']}")
    print(f"Examples cleaned (had placeholders): {stats['cleaned']} ({stats['cleaned']/stats['total']*100:.1f}%)")
    print(f"Total placeholders removed: {stats['removed']:,}")
    print(f"Average placeholders per cleaned example: {stats['removed']/stats['cleaned']:.1f}" if stats['cleaned'] > 0 else "N/A")
    print(f"Examples skipped (became empty after cleaning): {stats['skipped_empty']}")
    print(f"Remaining placeholders in output: {stats['xmath_count_after']}")
    print(f"\nPlaceholders cleaned include:")
    print(f"  - @xmath, @xcite, @xref, @xeq, @xfig, @xtab, @xsec")
    print(f"  - Broken figure/table/section references")
    
    if dry_run:
        print("\nDRY RUN - No files were modified")
    else:
        print(f"\nCleaned data written to: {output_path}")
        print(f"Output file size: {output_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean @xmath placeholders from training data")
    parser.add_argument(
        '--input',
        type=str,
        default='arxiv_abstract/data/arxiv_summarization_train_instruct.jsonl',
        help='Input training data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='arxiv_abstract/data/arxiv_summarization_train_instruct_cleaned.jsonl',
        help='Output cleaned training data file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only report statistics without writing output'
    )
    
    args = parser.parse_args()
    
    clean_training_data(args.input, args.output, dry_run=args.dry_run)

