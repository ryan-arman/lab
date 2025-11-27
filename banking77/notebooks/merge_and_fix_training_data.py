#!/usr/bin/env python3
"""
Merge banking77_train.jsonl and evaluation_results.jsonl, then fix format.
This combines the two files and converts evaluation_results entries to messages format.
"""

import json
import sys
import re
from pathlib import Path
from utils import SYSTEM_PROMPT


def convert_evaluation_entry(entry):
    """Convert evaluation_results entry to messages format."""
    if "messages" in entry:
        # Already in correct format
        return entry
    
    # Convert from {"request": "...", "response": "..."} format
    request = entry.get("request", "")
    response = entry.get("response", "").strip()
    
    # Extract label from response (should be ONLY a number, not embedded in text)
    try:
        # Check if the entire response is just a number (with optional whitespace)
        if re.match(r'^\s*\d+\s*$', response):
            label = int(response.strip())
            if not (0 <= label <= 76):
                return None
        else:
            # Response is not just a number (e.g., "2.9%", "The answer is 5")
            return None
    except Exception:
        return None
    
    # Create messages format
    converted = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
            {"role": "assistant", "content": str(label)}
        ],
        "metadata": {
            "label": label,
            "label_name": f"label_{label}"  # We don't have label names from evaluation_results
        }
    }
    
    return converted


def is_valid_label(content):
    """
    Check if the assistant response is a valid integer label (0-76).
    The response must be ONLY a number (with optional whitespace), not embedded in other text.
    Returns the label if valid, None otherwise.
    """
    if not content:
        return None
    
    # Strip whitespace
    content = content.strip()
    
    # Check if the entire content is just a number (possibly with whitespace)
    # This ensures "2.9%" or "The answer is 5" won't pass
    if re.match(r'^\s*\d+\s*$', content):
        try:
            label = int(content.strip())
            if 0 <= label <= 76:
                return label
        except ValueError:
            pass
    
    return None


def has_valid_assistant_response(messages):
    """
    Check if the last assistant message contains a valid label (0-76).
    """
    # Find the last assistant message
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return is_valid_label(content) is not None
    return False


def merge_and_fix_files(train_file, eval_file, output_file):
    """Merge training and evaluation files, converting format as needed."""
    train_path = Path(train_file)
    eval_path = Path(eval_file)
    output_path = Path(output_file)
    
    if not train_path.exists():
        print(f"Error: Training file not found: {train_file}")
        sys.exit(1)
    
    if not eval_path.exists():
        print(f"Error: Evaluation file not found: {eval_file}")
        sys.exit(1)
    
    train_count = 0
    train_filtered = 0
    eval_count = 0
    eval_written = 0
    eval_filtered = 0
    converted_count = 0
    skipped_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # First, copy all entries from training file
        print(f"Reading training data from {train_path}...")
        with open(train_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "messages" in entry:
                        # Validate assistant response is a valid label (0-76)
                        if has_valid_assistant_response(entry["messages"]):
                            outfile.write(line + '\n')
                            train_count += 1
                        else:
                            train_filtered += 1
                    else:
                        print(f"Warning: Training entry missing 'messages' key, skipping")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error in training file: {e}")
        
        # Then, convert and add entries from evaluation file
        print(f"Reading evaluation data from {eval_path}...")
        with open(eval_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    eval_count += 1
                    
                    if "messages" in entry:
                        # Check if system prompt is present
                        messages = entry["messages"]
                        has_system = any(msg.get("role") == "system" for msg in messages)
                        
                        if not has_system:
                            # Add system prompt at the beginning
                            entry["messages"] = [
                                {"role": "system", "content": SYSTEM_PROMPT}
                            ] + messages
                            converted_count += 1
                        
                        # Validate assistant response is a valid label (0-76)
                        if has_valid_assistant_response(entry["messages"]):
                            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            eval_written += 1
                        else:
                            eval_filtered += 1
                            print(f"Line {line_num}: Filtered entry (invalid assistant response)")
                    else:
                        # Convert from evaluation_results format
                        converted = convert_evaluation_entry(entry)
                        if converted:
                            # Validate assistant response is a valid label (0-76)
                            if has_valid_assistant_response(converted["messages"]):
                                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                                converted_count += 1
                                eval_written += 1
                            else:
                                eval_filtered += 1
                                print(f"Line {line_num}: Filtered entry (invalid assistant response after conversion)")
                        else:
                            skipped_count += 1
                            print(f"Line {line_num}: Skipped entry (could not convert)")
                            
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: JSON decode error: {e}")
                    skipped_count += 1
                except Exception as e:
                    print(f"Line {line_num}: Error: {e}")
                    skipped_count += 1
    
    print(f"\nMerge and conversion complete!")
    print(f"  Training entries: {train_count} (filtered: {train_filtered})")
    print(f"  Evaluation entries: {eval_written} written, {eval_filtered} filtered, {skipped_count} skipped (total read: {eval_count})")
    print(f"  Converted: {converted_count}")
    print(f"  Total output: {train_count + eval_written}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    train_file = "data/banking77_train.jsonl"
    eval_file = "data/evaluation_results.jsonl"
    output_file = "data/banking77_train_improved_fixed.jsonl"
    
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
    if len(sys.argv) > 2:
        eval_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    merge_and_fix_files(train_file, eval_file, output_file)

