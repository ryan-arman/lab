"""Utility functions for Banking77 evaluation."""

import json
import re


def measure_accuracy(inference_data):
    """
    Measure accuracy for all items in inference_data.
    
    Args:
        inference_data: List of dictionaries, each containing:
            - 'messages': List with last message containing predicted label in 'content'
            - 'metadata': Dict with 'label' containing ground truth label
    
    Returns:
        Tuple of (accuracy, correct_count, total_count, errors)
    """
    correct = 0
    total = 0
    errors = []
    
    for idx, row in enumerate(inference_data):
        try:
            # Extract predicted label from the last message content
            content = row['messages'][-1]['content'].strip()
            # Extract first integer from the response (handles cases where model adds reasoning)
            match = re.search(r'\b(\d+)\b', content)
            if match:
                inference_class_label = int(match.group(1))
            else:
                raise ValueError(f"No integer found in response: '{content[:50]}...'")
            
            # Extract ground truth label
            gt_label = row['metadata']['label']
            if len(content) > 5:
                print(f"idx: {idx}, gt_label: {gt_label}, inference_class_label: {inference_class_label}, content: {content}")
            
            # Compare and count
            if inference_class_label == gt_label:
                correct += 1
            total += 1
            
        except (KeyError, ValueError, IndexError) as e:
            errors.append((idx, str(e)))
            total += 1  # Still count as total, but mark as incorrect
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        for idx, error_msg in errors[:5]:  # Show first 5 errors
            print(f"  Row {idx}: {error_msg}")
    
    return accuracy, correct, total, errors


def measure_accuracy_thinking(inference_data):
    """
    Measure accuracy for thinking model outputs.
    
    Thinking models output reasoning followed by the answer. This function
    extracts the final answer, which is typically the last integer in the response
    or appears after reasoning markers.
    
    Args:
        inference_data: List of dictionaries, each containing:
            - 'messages': List with last message containing predicted label in 'content'
            - 'metadata': Dict with 'label' containing ground truth label
    
    Returns:
        Tuple of (accuracy, correct_count, total_count, errors)
    """
    correct = 0
    total = 0
    errors = []
    
    for idx, row in enumerate(inference_data):
        try:
            # Extract predicted label from the last message content
            content = row['messages'][-1]['content'].strip()
            
            # For thinking models, try multiple strategies to extract the answer:
            inference_class_label = None
            
            # Strategy 1: Check for think block markers and extract everything after them
            # Then find the integer in that portion
            # This handles cases like: </think>\n\n41 or </think>\n41
            think_block_match = re.search(r'</think>\s*\n+', content, re.IGNORECASE)
            if think_block_match:
                content_after_think = content[think_block_match.end():].strip()
                # Find integer in the content after think block
                integer_match = re.search(r'\b(\d+)\b', content_after_think)
                if integer_match:
                    inference_class_label = int(integer_match.group(1))
            
            # Strategy 2: Look for patterns like "Therefore, we output 41" or "the answer is 41"
            if inference_class_label is None:
                patterns = [
                    r'(?:therefore|thus|hence|so|answer|output|result|conclusion).*?(\d+)',
                    r'(?:we output|the answer is|the result is|output|answer).*?(\d+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        inference_class_label = int(match.group(1))
                        break
            
            # Strategy 3: Extract integer from the very end of the content
            # (thinking models often put the answer at the very end, e.g., "...\n\n41")
            if inference_class_label is None:
                # Strip whitespace and try to extract number from the end
                content_stripped = content.rstrip()
                # Look for integer at the end (handles both single and multi-digit)
                end_match = re.search(r'(\d+)\s*$', content_stripped)
                if end_match:
                    inference_class_label = int(end_match.group(1))
            
            # Strategy 4: If still no match, get the last integer anywhere in the response
            if inference_class_label is None:
                # Find all integers in the content
                all_integers = re.findall(r'\b(\d+)\b', content)
                if all_integers:
                    # Use the last integer (most likely to be the final answer)
                    inference_class_label = int(all_integers[-1])
                else:
                    raise ValueError(f"No integer found in response: '{content[:100]}...'")
            
            # Extract ground truth label
            gt_label = row['metadata']['label']
            
            # Only print if content is long (thinking model output) or if there's a mismatch
            if len(content) > 200 or inference_class_label != gt_label:
                # Truncate content for display
                content_preview = content[:200] + "..." if len(content) > 200 else content
                print(f"idx: {idx}, gt_label: {gt_label}, inference_class_label: {inference_class_label}, content_preview: {content_preview}")
            
            # Compare and count
            if inference_class_label == gt_label:
                correct += 1
            total += 1
            
        except (KeyError, ValueError, IndexError) as e:
            errors.append((idx, str(e)))
            total += 1  # Still count as total, but mark as incorrect
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        for idx, error_msg in errors[:5]:  # Show first 5 errors
            print(f"  Row {idx}: {error_msg}")
    
    return accuracy, correct, total, errors


def read_jsonl(file_path: str):
    """
    Read a JSONL file and return a list of parsed JSON objects.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line in the file
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
                continue
            except Exception as e:
                print(f"Error: Line {line_num} processing error: {e}")
                continue
    
    return data

