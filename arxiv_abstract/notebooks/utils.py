"""Utility functions for arXiv abstract evaluation and display."""

import os
import sys
import json
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import display, HTML
import openai

# Try to import tqdm for progress bars, with fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    # Fallback: create a simple tqdm-like class that does nothing
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
        
        def __iter__(self):
            return iter(self.iterable) if self.iterable else self
        
        def __next__(self):
            if self.iterable:
                return next(self.iterable)
            raise StopIteration
        
        def update(self, n=1):
            self.n += n
        
        def close(self):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

# Judge system instruction (matches API server structure)
JUDGE_SYSTEM_INSTRUCTION = """You are an expert academic reviewer evaluating whether a summary of an arXiv paper is of good quality, according to the five dimensions below.

Your task is to determine whether the LLM summary (contained in the "response" field), which summarizes the article (contained in the "request" field), meets the minimum quality standards across all five dimensions. A summary should receive 'Yes' if it meets the basic requirements for each dimension (typically scores of 70 or above), even if it is not perfect. Respond with 'No' only if the summary has significant flaws that violate the core criteria in one or more dimensions (typically scores below 70).

IMPORTANT: Academic abstracts are expected to be concise and may omit detailed technical specifics while still being high-quality summaries. Do not penalize brevity if the summary covers the essential aspects (main problem, approach, and at least one key result).

CRITICAL FORMATTING REQUIREMENT: You MUST begin your response with either "Yes" or "No" on the first line, followed by a blank line, then your detailed explanation.

You must provide a detailed explanation of your reasoning, addressing each of the five dimensions (Faithfulness, Coverage, Clarity, Conciseness, and Coherence). Your explanation should clearly justify your judgment and give a 0-100 score for each dimension and a 0-100 score for the overall summary.

Dimensions:

1. Faithfulness: The summary must accurately reflect the paper's content.
- Yes if all of the following are met:
  - Every factual statement in the summary is supported by the paper text.
  - No new claims, methods, results, or datasets are introduced that are absent from the paper.
  - The described relationships between ideas (e.g., 'the method improves X using Y') match the paper.
- No if any of the following occur:
  - The summary includes hallucinated or incorrect claims, numbers, results, or conclusions.
  - The summary attributes methods, datasets, or results to the paper that it does not contain.
  - The summary confuses the paper's contribution (e.g., swaps method/result, changes task).
  - The paper and summary are on unrelated topics.
  - The summary contradicts the paper's stated findings.
- Corner cases:
  - If the summary is off-topic or refers to content outside the paper, output 'No'.
  - If the summary omits details but makes no incorrect statements, do not penalize here (evaluate under Coverage instead).

2. Coverage: The summary must include the essential aspects of the paper.
- Yes if all of the following are met:
  - Mentions the paper's main problem or goal.
  - Describes the core approach or method (at a high level is acceptable for abstracts).
  - Includes at least one main result, finding, or contribution.
- No if any of the following occur:
  - Omits ALL of the key aspects above (missing problem, method, AND results).
  - Focuses only on background or context without describing the paper's work.
  - Only restates the title or abstract fragment without substance.
  - Includes unrelated or irrelevant information instead of summarizing the paper.
- Corner cases:
  - For abstracts: High-level descriptions of methods and results are acceptable. Detailed technical specifics are not required.
  - If the summary is extremely short and fails to mention results or contributions, output 'No'.
  - If the paper section provided is incomplete (e.g., abstract only), evaluate coverage relative to the provided text.

3. Clarity: The summary must be understandable and readable.
- Yes if all of the following are met:
  - Sentences are grammatically correct and well-formed.
  - The meaning is unambiguous and understandable to a technically literate reader.
  - Technical terms are used correctly and not misapplied.
- No if any of the following occur:
  - Contains incoherent, incomplete, or nonsensical sentences.
  - Ambiguity or unclear phrasing prevents comprehension.
  - Significant grammatical errors make meaning unclear.
  - Includes non-academic, personal, or irrelevant commentary.
- Corner cases:
  - Minor typos that don't affect meaning are acceptable.
  - If meaning is mostly clear despite small syntax issues, do not penalize.

4. Conciseness: The summary must be focused and not verbose.
- Yes if all of the following are met:
  - Contains only information directly related to the paper.
  - Avoids unnecessary background, repetition, or filler phrases.
  - Length is appropriate for the type of summary (abstracts are expected to be brief; detailed summaries can be longer).
- No if any of the following occur:
  - Includes unrelated context (e.g., general field discussion or subjective opinions).
  - Repeats the same ideas multiple times.
  - Is excessively wordy or much longer than needed to ACTUALLY convey the core content.
  - Contains filler text, self-referential commentary, or promotional tone.
- Corner cases:
  - Short but fully informative summaries (especially abstracts): respond 'Yes'. Brevity is a virtue for abstracts, not a flaw.
  - Do not penalize concise abstracts for being brief if they cover essential information.

5. Coherence: The summary must be logically structured and flow naturally.
- Yes if all of the following are met:
  - The order of information follows a logical progression (e.g., problem → method → results → implications).
  - Ideas connect smoothly without abrupt topic shifts.
  - No contradictions or circular statements.
- No if any of the following occur:
  - The text jumps between unrelated points without transitions.
  - The sequence of ideas makes understanding difficult.
  - Sentences contradict one another or present disjointed fragments.
- Corner cases:
  - A one-sentence summary that is internally consistent is a 'Yes'.
  - Fragmented bullet points or unordered ideas are 'No'.
"""

# Judge prompt template (matches API server format)
JUDGE_PROMPT_TEMPLATE_WITH_REQUEST_AND_RESPONSE = """Here is the data:
[BEGIN DATA]
***
[user request]:
{request}
***
[response]:
{response}
***
[END DATA]"""


def get_openai_client():
    """Initialize and return OpenAI client using API key from environment."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return openai.OpenAI(api_key=api_key)


# Initialize client (can be re-initialized if needed)
client = get_openai_client()


def evaluate_summary(messages, model="gpt-4o", temperature=1.0, return_full=False, client_instance=None):
    """
    Evaluate a summary using the judge prompt.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                 Expected format: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        model: OpenAI model to use (default: "gpt-4o")
        temperature: Temperature for the API call (default: 0)
        return_full: If True, return full response dict; if False, return just judgment and explanation
        client_instance: Optional OpenAI client instance (uses module-level client if not provided)
    
    Returns:
        If return_full=False: dict with 'judgment' (Yes/No) and 'explanation' (str)
        If return_full=True: dict with 'judgment', 'explanation', and 'full_response'
    """
    if client_instance is None:
        client_instance = client
    
    # Extract user and assistant content from messages
    user_content = None
    assistant_content = None
    
    for msg in messages:
        if msg['role'] == 'user':
            user_content = msg['content']
        elif msg['role'] == 'assistant':
            assistant_content = msg['content']
    
    if user_content is None or assistant_content is None:
        raise ValueError("Messages must contain both 'user' and 'assistant' roles")
    
    # Format the judge prompt template with the actual content (matches API server format)
    evaluation_prompt = JUDGE_PROMPT_TEMPLATE_WITH_REQUEST_AND_RESPONSE.format(
        request=user_content,
        response=assistant_content
    )

    # Call OpenAI to evaluate (matches API server structure: system instruction + prompt template)
    response = client_instance.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_INSTRUCTION},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=temperature
    )

    full_response = response.choices[0].message.content.strip()

    # Try to extract Yes/No judgment
    judgment = None
    explanation = full_response
    
    # First, check if the first line or first word is Yes/No
    first_line = full_response.split('\n', 1)[0].strip()
    if first_line.upper() in ['YES', 'NO']:
        judgment = first_line
        if '\n' in full_response:
            explanation = '\n'.join(full_response.split('\n')[1:]).strip()
    else:
        # Look for "Yes" or "No" in the first few lines or first 200 characters
        search_text = full_response[:200].lower()
        # Check for explicit Yes/No
        if 'judgment:' in search_text or 'judgement:' in search_text:
            # Extract after "Judgment:" or "Judgement:"
            match = re.search(r'judg(?:e)?ment:\s*(yes|no)', search_text, re.IGNORECASE)
            if match:
                judgment = match.group(1).capitalize()
        elif 'yes' in search_text and 'no' not in search_text[:search_text.find('yes')+10]:
            # "Yes" appears early and "No" doesn't appear before it
            judgment = "Yes"
        elif 'no' in search_text and 'yes' not in search_text[:search_text.find('no')+10]:
            # "No" appears early and "Yes" doesn't appear before it
            judgment = "No"
        else:
            # Fallback: look for phrases that indicate Yes/No
            if any(phrase in full_response.lower() for phrase in [
                'meets the minimum quality standards',
                'meets the basic requirements',
                'satisfies the criteria',
                'is of good quality'
            ]):
                judgment = "Yes"
            elif any(phrase in full_response.lower() for phrase in [
                'does not meet',
                'fails to meet',
                'violates the core criteria',
                'has significant flaws'
            ]):
                judgment = "No"
    
    # If still no judgment found, default to Unknown
    if judgment is None:
        judgment = "Unknown"
    
    result = {
        'judgment': judgment,
        'explanation': explanation
    }
    
    if return_full:
        result['full_response'] = full_response
    
    return result, evaluation_prompt


def evaluate_summaries_batch(conversations, model="gpt-4o", temperature=1.0, max_workers=5, 
                              return_full=False, client_instance=None, show_progress=True):
    """
    Evaluate multiple summaries in parallel using batch processing.
    
    Args:
        conversations: List of conversation message lists (each is a list of message dicts)
        model: OpenAI model to use (default: "gpt-4o")
        temperature: Temperature for the API call (default: 1.0)
        max_workers: Maximum number of parallel workers (default: 5)
        return_full: If True, return full response dict; if False, return just judgment and explanation
        client_instance: Optional OpenAI client instance (uses module-level client if not provided)
        show_progress: If True, print progress updates (default: True)
    
    Returns:
        List of tuples: (index, result_dict, evaluation_prompt) for each conversation
        Results are returned in the order they complete (may not match input order)
        If you need results in input order, sort by index after receiving results.
    """
    if client_instance is None:
        client_instance = client
    
    results = []
    errors = []
    
    def evaluate_single(idx, conv):
        """Evaluate a single conversation and return index with result."""
        try:
            result, prompt = evaluate_summary(
                conv, 
                model=model, 
                temperature=temperature, 
                return_full=return_full,
                client_instance=client_instance
            )
            return (idx, result, prompt, None)
        except Exception as e:
            return (idx, None, None, str(e))
    
    # Process in parallel
    if show_progress:
        print(f"Evaluating {len(conversations)} conversations with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(evaluate_single, idx, conv): idx 
            for idx, conv in enumerate(conversations)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx, result, prompt, error = future.result()
            if error:
                errors.append((idx, error))
                if show_progress:
                    print(f"  Error on conversation {idx}: {error}")
            else:
                results.append((idx, result, prompt))
                completed += 1
                if show_progress:
                    print(f"  Completed {completed}/{len(conversations)}")
    
    if show_progress:
        print(f"\n✓ Completed: {len(results)} successful, {len(errors)} errors")
    
    # Sort results by index to maintain input order
    results.sort(key=lambda x: x[0])
    
    return results, errors


def generate_abstract(messages, model="gpt-4o", temperature=0.7, client_instance=None):
    """
    Generate an abstract for a paper using OpenAI API.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                 Expected format: [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
                 The system message should contain instructions, user message should contain paper content.
        model: OpenAI model to use (default: "gpt-4o", can use "gpt-4o", "o1-preview", etc.)
        temperature: Temperature for the API call (default: 0.7)
        client_instance: Optional OpenAI client instance (uses module-level client if not provided)
    
    Returns:
        dict with 'abstract' (str) and 'full_response' (OpenAI response object)
    """
    if client_instance is None:
        client_instance = client
    
    # Call OpenAI to generate abstract
    response = client_instance.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    full_response = response
    abstract = response.choices[0].message.content
    
    return {
        'abstract': abstract,
        'full_response': full_response
    }


def generate_abstracts_batch(conversations, model="gpt-4o", temperature=0.7, max_workers=5,
                            client_instance=None, show_progress=True):
    """
    Generate abstracts for multiple papers in parallel using batch processing.
    
    Args:
        conversations: List of conversation message lists (each is a list of message dicts)
                      Each conversation should have 'system' and 'user' messages (no 'assistant' message)
        model: OpenAI model to use (default: "gpt-4o")
        temperature: Temperature for the API call (default: 0.7)
        max_workers: Maximum number of parallel workers (default: 5)
        client_instance: Optional OpenAI client instance (uses module-level client if not provided)
        show_progress: If True, print progress updates (default: True)
    
    Returns:
        List of tuples: (index, result_dict, error) for each conversation
        Results are returned in the order they complete (may not match input order)
        If you need results in input order, sort by index after receiving results.
    """
    if client_instance is None:
        client_instance = client
    
    results = []
    errors = []
    
    def generate_single(idx, conv):
        """Generate abstract for a single conversation and return index with result."""
        try:
            result = generate_abstract(
                conv,
                model=model,
                temperature=temperature,
                client_instance=client_instance
            )
            return (idx, result, None)
        except Exception as e:
            return (idx, None, str(e))
    
    # Process in parallel
    if show_progress:
        print(f"Generating abstracts for {len(conversations)} papers with {max_workers} workers...")
        print(f"Using model: {model}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(generate_single, idx, conv): idx 
            for idx, conv in enumerate(conversations)
        }
        
        # Collect results as they complete with progress bar
        completed = 0
        pbar = None
        if show_progress:
            pbar = tqdm(total=len(conversations), desc="Generating abstracts", unit="abstract")
        
        try:
            for future in as_completed(future_to_idx):
                idx, result, error = future.result()
                if error:
                    errors.append((idx, error))
                    if show_progress and pbar:
                        pbar.write(f"  Error on conversation {idx}: {error}")
                else:
                    results.append((idx, result))
                    completed += 1
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': len(results),
                        'errors': len(errors)
                    })
        finally:
            if pbar:
                pbar.close()
    
    if show_progress:
        print(f"\n✓ Completed: {len(results)} successful, {len(errors)} errors")
    
    # Sort results by index to maintain input order
    results.sort(key=lambda x: x[0])
    
    return results, errors


def display_text(text, role=None, max_width=80, max_chars=None, show_stats=True):
    """
    Display large text in a readable format.
    
    Args:
        text: The text to display
        role: Optional role label (e.g., 'assistant', 'user')
        max_width: Maximum width for text wrapping (default: 80)
        max_chars: Maximum characters to display (None for full text)
        show_stats: Whether to show text statistics (default: True)
    """
    # Truncate if needed
    original_length = len(text)
    display_text_content = text
    if max_chars and len(text) > max_chars:
        display_text_content = text[:max_chars] + f"\n\n... [truncated, showing {max_chars}/{original_length} characters]"
    
    # Show statistics
    if show_stats:
        word_count = len(text.split())
        char_count = len(text)
        line_count = text.count('\n') + 1
        print(f"{'='*80}")
        if role:
            print(f"Role: {role.upper()}")
        print(f"Characters: {char_count:,} | Words: {word_count:,} | Lines: {line_count:,}")
        print(f"{'='*80}\n")
    
    # Wrap and display
    wrapped_lines = []
    for line in display_text_content.split('\n'):
        if len(line) <= max_width:
            wrapped_lines.append(line)
        else:
            # Wrap long lines
            wrapped = textwrap.wrap(line, width=max_width)
            wrapped_lines.extend(wrapped)
    
    # Display with HTML for better formatting
    html_content = '<pre style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; background-color: #f5f5f5; color: #000000; padding: 10px; border-radius: 5px; line-height: 1.5;">'
    html_content += '\n'.join(wrapped_lines)
    html_content += '</pre>'
    display(HTML(html_content))


def display_message(messages, role='assistant', **kwargs):
    """
    Display content from a message with a specific role.
    
    Args:
        messages: List of message dicts
        role: Role to extract ('user' or 'assistant')
        **kwargs: Additional arguments passed to display_text
    """
    for msg in messages:
        if msg['role'] == role:
            display_text(msg['content'], role=role, **kwargs)
            return
    print(f"No message found with role '{role}'")


def load_conversations(input_path):
    """
    Load conversations from a JSONL file.
    
    Args:
        input_path: Path to the JSONL file containing conversations
        
    Returns:
        List of message lists, where each message list contains dicts with 'role' and 'content' keys
    """
    conversations = []
    with open(input_path, 'r', encoding='utf-8') as infile:     
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the JSONL format
                data = json.loads(line)
                if 'messages' not in data:
                    print(f"Warning: Line {line_num} missing 'messages' field, skipping", file=sys.stderr)
                    continue
                
                messages = data['messages']
                conversations.append(messages) 
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Error: Line {line_num} processing error: {e}", file=sys.stderr)
                continue 
    return conversations

