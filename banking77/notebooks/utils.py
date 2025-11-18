"""Utility functions for Banking77 evaluation."""

import json
import os
import re
import openai


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
    incorrect_list = []
    
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
            else:
                incorrect_list.append((idx, inference_class_label, gt_label))
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
    
    return accuracy, correct, total, errors, incorrect_list


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


# System instruction for evaluating incorrect classifications
EVALUATION_SYSTEM_INSTRUCTION = """You are an expert at analyzing banking intent classification errors. Your task is to explain why a model incorrectly classified a user's banking query.

You will be given:
1. The user's query
2. The ground truth (correct) intent label and name
3. The predicted (incorrect) intent label and name
4. The system prompt that was used (which contains all available intent labels)

Your goal is to provide a clear, insightful explanation of why the model made this mistake. Consider:
- Semantic similarity between the two intents
- Ambiguous wording in the user query
- Missing context that would clarify the intent
- Whether the intents are genuinely difficult to distinguish
- Any patterns in the query that might have led to the confusion

Provide a detailed explanation that would help improve the classification system."""

# Evaluation prompt template
EVALUATION_PROMPT_TEMPLATE = """Here is the misclassification data:

[BEGIN DATA]
***
[User Query]:
{user_query}
***
[Ground Truth]:
Label ID: {gt_label_id}
Label Name: {gt_label_name}
***
[Predicted (Incorrect)]:
Label ID: {predicted_label_id}
Label Name: {predicted_label_name}
***
[System Prompt with All Labels]:
{system_prompt}
***
[END DATA]

Please analyze why the model incorrectly classified this query and provide a detailed explanation."""


def get_openai_client():
    """Initialize and return OpenAI client using API key from environment."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return openai.OpenAI(api_key=api_key)


# Initialize client (can be re-initialized if needed)
_client = None

def get_client():
    """Get or create OpenAI client instance."""
    global _client
    if _client is None:
        _client = get_openai_client()
    return _client


def extract_label_name(label_id, system_message):
    """Extract label name for a given ID from system message."""
    pattern = rf'{label_id}:\s*([^\n]+)'
    match = re.search(pattern, system_message)
    if match:
        return match.group(1).strip()
    return f"Unknown_{label_id}"


def evaluate_incorrect_classification(inference_row, model="gpt-4o", temperature=0.7, client_instance=None):
    """
    Evaluate why a classification was incorrect using OpenAI.
    
    Args:
        inference_row: Dictionary from inference_data containing:
            - 'messages': List with system message, user message, and assistant response
            - 'metadata': Dict with 'label' (ground truth) and 'label_name'
        model: OpenAI model to use (default: "gpt-4o")
        temperature: Temperature for the API call (default: 0.7)
        client_instance: Optional OpenAI client instance (uses module-level client if not provided)
    
    Returns:
        dict with 'explanation' (str) and 'full_response' (str)
    """
    if client_instance is None:
        client_instance = get_client()
    
    # Extract data from inference_row
    messages = inference_row['messages']
    system_message = messages[0]['content']  # System prompt with all labels
    user_query = messages[1]['content']  # User query
    model_response = messages[2]['content'].strip()  # Model's response
    
    # Extract predicted label (first integer in response)
    predicted_match = re.search(r'\b(\d+)\b', model_response)
    if not predicted_match:
        raise ValueError(f"Could not extract predicted label from response: '{model_response}'")
    predicted_label_id = int(predicted_match.group(1))
    predicted_label_name = extract_label_name(predicted_label_id, system_message)
    
    # Get ground truth
    gt_label_id = inference_row['metadata']['label']
    gt_label_name = inference_row['metadata'].get('label_name', extract_label_name(gt_label_id, system_message))
    
    # Format the evaluation prompt
    evaluation_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        user_query=user_query,
        gt_label_id=gt_label_id,
        gt_label_name=gt_label_name,
        predicted_label_id=predicted_label_id,
        predicted_label_name=predicted_label_name,
        system_prompt=system_message
    )
    
    # Call OpenAI to evaluate
    response = client_instance.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EVALUATION_SYSTEM_INSTRUCTION},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=temperature
    )
    
    full_response = response.choices[0].message.content.strip()
    
    return {
        'explanation': full_response,
        'full_response': full_response,
        'gt_label_id': gt_label_id,
        'gt_label_name': gt_label_name,
        'predicted_label_id': predicted_label_id,
        'predicted_label_name': predicted_label_name,
        'user_query': user_query
    }


def evaluate_incorrect_classifications_batch(inference_data, incorrect_list, model="gpt-4o", 
                                             temperature=1.0, max_workers=5,
                                             client_instance=None, show_progress=True):
    """
    Evaluate multiple incorrect classifications in parallel using batch processing.
    
    Args:
        inference_data: List of inference data dictionaries
        incorrect_list: List of tuples (row_idx, predicted_label, gt_label) from measure_accuracy
        model: OpenAI model to use (default: "gpt-4o")
        temperature: Temperature for the API call (default: 0.7)
        max_workers: Maximum number of parallel workers (default: 5)
        client_instance: Optional OpenAI client instance (uses module-level client if not provided)
        show_progress: If True, print progress updates (default: True)
    
    Returns:
        List of tuples: (row_idx, result_dict, error) for each incorrect classification
        Results are returned in the order they complete (may not match input order)
        If you need results in input order, sort by row_idx after receiving results.
    """
    if client_instance is None:
        client_instance = get_client()
    
    results = []
    errors = []
    
    def evaluate_single(row_idx, predicted_label, gt_label):
        """Evaluate a single incorrect classification and return result."""
        try:
            inference_row = inference_data[row_idx]
            result = evaluate_incorrect_classification(
                inference_row,
                model=model,
                temperature=temperature,
                client_instance=client_instance
            )
            return (row_idx, result, None)
        except Exception as e:
            return (row_idx, None, str(e))
    
    # Process in parallel
    if show_progress:
        print(f"Evaluating {len(incorrect_list)} incorrect classifications with {max_workers} workers...")
        print(f"Using model: {model}")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(evaluate_single, row_idx, pred_label, gt_label): row_idx
            for row_idx, pred_label, gt_label in incorrect_list
        }
        
        # Collect results as they complete
        completed = 0
        pbar = None
        if show_progress and has_tqdm:
            pbar = tqdm(total=len(incorrect_list), desc="Evaluating misclassifications", unit="item")
        
        try:
            for future in as_completed(future_to_idx):
                row_idx, result, error = future.result()
                if error:
                    errors.append((row_idx, error))
                    if show_progress and pbar:
                        pbar.write(f"  Error on row {row_idx}: {error}")
                else:
                    results.append((row_idx, result))
                    completed += 1
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': len(results),
                        'errors': len(errors)
                    })
                elif show_progress and not has_tqdm:
                    print(f"  Completed {completed}/{len(incorrect_list)}")
        finally:
            if pbar:
                pbar.close()
    
    if show_progress:
        print(f"\n✓ Completed: {len(results)} successful, {len(errors)} errors")
    
    # Sort results by row_idx to maintain input order
    results.sort(key=lambda x: x[0])
    
    return results, errors


def save_evaluation_results(results, inference_data, output_path):
    """
    Save evaluation results to a JSONL file in the specified format.
    
    Args:
        results: List of tuples (row_idx, result_dict) from evaluate_incorrect_classifications_batch
        inference_data: List of inference data dictionaries (to get model responses)
        output_path: Path to the output JSONL file
    
    Format:
        {"request": "...", "response": "...", "judgment": false, "explanation": "..."}
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for row_idx, result in results:
            # Get the model's response from the inference data
            inference_row = inference_data[row_idx]
            model_response = inference_row['messages'][2]['content'].strip()
            
            # Format according to specification
            output_entry = {
                "request": result['user_query'],
                "response": model_response,
                "judgment": False,
                "explanation": result['explanation']
            }
            
            # Write as JSON line
            f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} evaluation results to {output_path}")

