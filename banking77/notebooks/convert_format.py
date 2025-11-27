#!/usr/bin/env python3
"""
Convert between banking77 test.jsonl format and arxiv_abstract_test.jsonl format.

Banking77 format:
{
    "messages": [
        {"content": "...", "role": "system"},
        {"content": "...", "role": "user"},
        {"content": "...", "role": "assistant"}  # Optional
    ]
}

Arxiv_abstract format:
{
    "content": {
        "request": "...",
        "response": ""
    },
    "metadata": {"label": "..."}
}
"""

import json
import sys
from pathlib import Path
from utils import SYSTEM_PROMPT, SYSTEM_PROMPT_BASIC


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


def convert_dict_to_banking77_format(input_file, output_file, system_prompt=None):
    """
    Convert dict format (arxiv_abstract) back to banking77 format.
    
    Args:
        input_file: Path to input dict JSONL file
        output_file: Path to output banking77 JSONL file
        system_prompt: System prompt to use (defaults to SYSTEM_PROMPT from utils)
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
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
                # Parse the dict format
                data = json.loads(line)
                
                if 'content' not in data:
                    print(f"Warning: Line {line_num} missing 'content' field, skipping", file=sys.stderr)
                    continue
                
                content = data['content']
                
                if 'request' not in content:
                    print(f"Warning: Line {line_num} missing 'request' field in content, skipping", file=sys.stderr)
                    continue
                
                request = content['request']
                
                # Extract user content from request
                # The request field may contain system prompt + "\n\n" + user query
                # We'll use the specified system prompt for the system message
                # and extract just the user content
                parts = request.split('\n\n', 1)
                
                if len(parts) == 1:
                    # If no "\n\n" separator found, treat entire request as user message
                    user_content = parts[0]
                else:
                    # Request contains system prompt + user query
                    # Use the second part as user content (first part was the system prompt)
                    user_content = parts[1]
                
                # Get response from content.response or metadata.label
                response = ""
                if 'response' in content and content['response']:
                    response = content['response']
                elif 'metadata' in data and 'label' in data['metadata']:
                    response = data['metadata']['label']
                
                # Reconstruct messages array using the specified system prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                
                # Add assistant message if response is not empty
                if response:
                    messages.append({"role": "assistant", "content": response})
                
                # Create banking77 format
                output_data = {
                    "messages": messages
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
    if len(sys.argv) < 3:
        print("Usage: python convert_format.py <input_file> <output_file> [--reverse] [--system-prompt <prompt_type>]", file=sys.stderr)
        print("  Forward (default): Convert banking77 format to arxiv_abstract format", file=sys.stderr)
        print("  Reverse (--reverse): Convert arxiv_abstract format to banking77 format", file=sys.stderr)
        print("  System prompt options: 'full' (default) or 'basic'", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  python convert_format.py data/test.jsonl data/test_arxiv_format.jsonl", file=sys.stderr)
        print("  python convert_format.py data/test_arxiv_format.jsonl data/test.jsonl --reverse", file=sys.stderr)
        print("  python convert_format.py data/test_arxiv_format.jsonl data/test.jsonl --reverse --system-prompt basic", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse arguments
    reverse = False
    system_prompt_type = 'full'
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--reverse':
            reverse = True
            i += 1
        elif sys.argv[i] == '--system-prompt' and i + 1 < len(sys.argv):
            system_prompt_type = sys.argv[i + 1]
            i += 2
        else:
            print(f"Warning: Unknown argument '{sys.argv[i]}', ignoring", file=sys.stderr)
            i += 1
    
    # Select system prompt based on type
    if system_prompt_type == 'basic':
        selected_prompt = SYSTEM_PROMPT_BASIC
    elif system_prompt_type == 'full':
        selected_prompt = SYSTEM_PROMPT
    else:
        print(f"Error: Unknown system prompt type '{system_prompt_type}'. Use 'full' or 'basic'.", file=sys.stderr)
        sys.exit(1)
    
    if reverse:
        convert_dict_to_banking77_format(input_file, output_file, system_prompt=selected_prompt)
    else:
        convert_banking77_to_dict_format(input_file, output_file)

