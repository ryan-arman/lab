#!/usr/bin/env python3
"""
Convert banking77 test.jsonl format to arxiv_abstract_test.jsonl format.

Input format (banking77):
{
    "messages": [
        {"content": "...", "role": "system"},
        {"content": "...", "role": "user"},
        {"content": "...", "role": "assistant"}  # Optional
    ]
}

Output format (arxiv_abstract):
{
    "content": {
        "request": "...",
        "response": ""
    }
}
"""

import json
import sys
from pathlib import Path


def convert_banking77_to_dict_format(input_file, output_file):
    """
    Convert banking77 format to dict format.
    
    Args:
        input_file: Path to input banking77 JSONL file
        output_file: Path to output dict JSONL file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    converted_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the banking77 format
                data = json.loads(line)
                
                if 'messages' not in data:
                    print(f"Warning: Line {line_num} missing 'messages' field, skipping", file=sys.stderr)
                    continue
                
                messages = data['messages']
                
                # Extract system, user, and assistant messages
                system_content = None
                user_content = None
                assistant_content = None
                
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    if role == 'system':
                        system_content = content
                    elif role == 'user':
                        user_content = content
                    elif role == 'assistant':
                        assistant_content = content
                
                if system_content is None:
                    print(f"Warning: Line {line_num} missing system message, skipping", file=sys.stderr)
                    continue
                
                if user_content is None:
                    print(f"Warning: Line {line_num} missing user message, skipping", file=sys.stderr)
                    continue
                
                # Combine system and user content for the request field
                # Format: system prompt + "\n\n" + user query
                request = f"{system_content}\n\n{user_content}"
                
                # Use assistant content for response, or empty string if not present
                response = assistant_content if assistant_content is not None else ""
                
                # Create arxiv_abstract format
                output_data = {
                    "content": {
                        "request": request,
                    },
                    "metadata": {"label": response}

                }
                
                # Write to output file
                outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Error: Line {line_num} processing error: {e}", file=sys.stderr)
                continue
    
    print(f"Successfully converted {converted_count} entries", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_format.py <input_file> <output_file>", file=sys.stderr)
        print("Example: python convert_format.py data/test.jsonl data/test_arxiv_format.jsonl", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_banking77_to_dict_format(input_file, output_file)

