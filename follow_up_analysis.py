import argparse
import json
import logging
import random
import re
import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from scipy.special import binom
from collections import defaultdict

# Import provider-specific libraries
from openai import AsyncOpenAI, RateLimitError
import replicate
import google.generativeai as genai

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Helper Functions ---

def load_adjectives_from_results(data: dict) -> list[str]:
    """Loads adjectives in the correct order from a results file's mapping."""
    mapping = data['adjective_mapping']
    adjectives = sorted(mapping.keys(), key=lambda k: mapping[k])
    logging.info(f"Loaded {len(adjectives)} adjectives from results file.")
    return adjectives

def get_original_questions(data: dict) -> list[dict]:
    """Reconstructs the list of originally sampled MMLU questions."""
    if 'per_question_results' not in data:
        raise ValueError("The input JSON must contain 'per_question_results' to identify which questions to test.")
        
    per_question_results = data['per_question_results']
    # Sort keys by the original index to ensure order is maintained
    question_keys = sorted(per_question_results.keys(), key=lambda k: int(k.split('_')[1]))
    
    all_questions = []
    # Group keys by subject to minimize dataset loading
    subject_to_keys = defaultdict(list)
    for key in question_keys:
        parts = key.split('_')
        subject = '_'.join(parts[2:])
        subject_to_keys[subject].append(key)

    seed = data['parameters']['seed']
    num_per_category = data['parameters']['num_per_category']

    for subject, keys in tqdm(subject_to_keys.items(), desc="Reloading original MMLU questions"):
        dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        
        # Re-generate the original random sample indices for this subject
        random.seed(seed)
        original_indices = random.sample(range(len(dataset)), num_per_category)
        
        # Create a map from original index to question data
        original_q_map = {idx: dataset[idx] for idx in original_indices}
        
        # Re-create the question list in the correct order
        for i in range(num_per_category):
            sample = original_q_map[original_indices[i]]
            sample['subject'] = subject
            all_questions.append(sample)
            
    logging.info(f"Successfully reloaded {len(all_questions)} original questions in the correct order.")
    return all_questions


def format_prompt_template(question_data: dict, adjectives: list[str], template_id: str) -> str:
    """Formats the prompt using different templates."""
    question = question_data['question']
    choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(question_data['choices'])])
    adj_string = ", ".join(adjectives) if adjectives else "clear and direct"

    if template_id == 'original':
        return f"The following is a multiple-choice question. Your answer should be {adj_string}. Provide only the letter corresponding to the correct option.\n\nQuestion: {question}\n\nChoices:\n{choices_str}"
    elif template_id == 'suffix':
        return f"Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer the question. Provide only the letter corresponding to the correct option. Your answer should be {adj_string}."
    elif template_id == 'persona':
        return f"Adopt the persona of an extremely {adj_string} expert to answer the following question. Provide only the letter corresponding to the correct option.\n\nQuestion: {question}\n\nChoices:\n{choices_str}"
    else:
        raise ValueError(f"Unknown template ID: {template_id}")

# --- API Prediction Functions (Complete Set) ---
def extract_answer_letter(content: str) -> str:
    """
    Extracts the final answer letter (A-D) from a model's response,
    handling various formats like reasoning blocks and LaTeX.
    """
    # 1. Strip away DeepSeek's <think> blocks to ignore the reasoning part.
    # The re.DOTALL flag allows '.' to match newlines.
    cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    # 2. Prioritize a LaTeX \boxed{} answer, as it's a very strong signal.
    match = re.search(r'\\boxed{\s*([A-D])\s*}', cleaned_content, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. Prioritize finding a letter that is on a line by itself,
    # checking from the end of the response backwards.
    lines = cleaned_content.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # Check if the entire line is just a single letter from A-D.
        if re.fullmatch(r'[A-D]', line, re.IGNORECASE):
            return line.upper()

    # 4. As a fallback, find the *last* occurrence of a letter A-D in the
    # cleaned content. This is more robust than finding the first one.
    all_matches = re.findall(r'[A-D]', cleaned_content.upper())
    if all_matches:
        return all_matches[-1]

    # 5. If no valid answer is found after all checks, return a failure code.
    return "Z"

async def get_openai_prediction(prompt_content, client, model_name, semaphore):
    """Gets a prediction from OpenAI and extracts the answer letter."""
    async with semaphore:
        try:
            messages = [{"role": "user", "content": prompt_content}]
            response = await client.chat.completions.create(model=model_name, messages=messages)
            content = response.choices[0].message.content
            return extract_answer_letter(content)
        except RateLimitError:
            logging.warning("OpenAI rate limit hit. Sleeping for 20s.")
            await asyncio.sleep(20)
            return await get_openai_prediction(prompt_content, client, model_name, semaphore) # Retry
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}. Retrying after 10s.")
            await asyncio.sleep(10)
            return "Z"

async def get_replicate_prediction(prompt_content, client, model_name, semaphore):
    """Gets a prediction from Replicate, inferring the correct prompt template."""
    async with semaphore:
        template = "{prompt}"
        if 'llama-3' in model_name:
            template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif 'phi-3' in model_name:
            template = "<|user|>\n{prompt}<|end|>\n<|assistant|>"
        elif 'deepseek-r1' in model_name:
            template = "User: {prompt}\n\nAssistant:"
        else:
            template = "{prompt}"
            logging.warning(f"Could not infer prompt template for model '{model_name}'. Using a generic template.")
        try:
            api_input = {"prompt": prompt_content, "prompt_template": template, "temperature": 0.1,"max_tokens": 2048}
            output = await replicate.async_run(model_name, input=api_input)
            content = ""
            # FIX: Handle both streaming and non-streaming responses
            if hasattr(output, '__aiter__'):
                # Handle streaming response
                async for item in output:
                    content += item
            elif isinstance(output, list):
                # Handle non-streaming response (list of strings)
                content = "".join(output)
            return extract_answer_letter(content)
        except replicate.exceptions.ReplicateError as e:
            logging.warning(f"Replicate API error: {e}. Sleeping for 20s.")
            await asyncio.sleep(20)
            return "Z"
        except Exception as e:
            logging.error(f"Replicate API call failed: {e}. Retrying after 10s.")
            await asyncio.sleep(10)
            return "Z"

async def get_gemini_prediction(prompt_content, model, model_name, semaphore):
    """Gets a prediction from Gemini and extracts the answer letter."""
    async with semaphore:
        try:
            await asyncio.sleep(1) # Gemini has a strict RPM limit
            response = await model.generate_content_async(prompt_content)
            content = response.text
            return extract_answer_letter(content)
        except Exception as e:
            logging.error(f"Gemini API call failed: {e}. Retrying after 10s.")
            await asyncio.sleep(10)
            return "Z"

# --- Core SHAP Logic ---

async def calculate_shap_for_question_subset(
    question_data: dict, adjective_subset: list[str], client, model_name: str, 
    num_samples: int, semaphore: asyncio.Semaphore, prediction_function: callable,
    template_id: str = 'original', fixed_adjectives: list = None
) -> np.ndarray:
    """Calculates Shapley values for a SUBSET of adjectives, with optional fixed context."""
    M = len(adjective_subset)
    fixed_adjectives = fixed_adjectives or []
    coalitions = np.zeros((num_samples, M), dtype=int)
    for i in range(num_samples):
        num_features = random.randint(1, M)
        features_on = random.sample(range(M), num_features)
        coalitions[i, features_on] = 1
    coalitions[0, :] = 0
    coalitions[1, :] = 1
    
    correct_answer_letter = chr(65 + question_data['answer'])
    tasks = []
    for coalition_vec in coalitions:
        active_adjectives = [adj for adj, active in zip(adjective_subset, coalition_vec) if active]
        prompt_adjectives = sorted(list(set(active_adjectives + fixed_adjectives)))
        prompt_content = format_prompt_template(question_data, prompt_adjectives, template_id)
        tasks.append(prediction_function(prompt_content, client, model_name, semaphore))

    predictions = await asyncio.gather(*tasks)
    scores = np.array([1 if pred == correct_answer_letter else 0 for pred in predictions])
    
    weights = np.zeros(num_samples)
    for i, coalition in enumerate(coalitions):
        z = np.sum(coalition)
        if z == 0 or z == M:
            weights[i] = 1e9
        else:
            weights[i] = (M - 1) / (binom(M, z) * z * (M - z) + 1e-9)
            
    model_lr = LinearRegression()
    model_lr.fit(X=coalitions, y=scores, sample_weight=weights)
    return model_lr.coef_

# --- Main Execution Phases ---

async def run_template_study(args, base_data, top_adjectives, questions, client, pred_func):
    logging.info("\n--- PHASE 1: TEMPLATE VARIANCE STUDY ---")
    templates = ['original', 'suffix', 'persona']
    template_results = {adj: {} for adj in top_adjectives}
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    for template_id in templates:
        logging.info(f"Running analysis for template: '{template_id}'")
        all_shap_values = []
        
        tasks = [
            calculate_shap_for_question_subset(
                q, top_adjectives, client, base_data['parameters']['model'],
                args.num_samples, semaphore, pred_func, template_id=template_id
            ) for q in questions
        ]
        
        for f in async_tqdm.as_completed(tasks, desc=f"Template '{template_id}'"):
            all_shap_values.append(await f)
        
        mean_shaps = np.mean(all_shap_values, axis=0)
        for i, adj in enumerate(top_adjectives):
            template_results[adj][template_id] = mean_shaps[i]

    print("\n--- Template Variance Results (Mean Shapley) ---")
    df_template = pd.DataFrame(template_results).T
    print(df_template.to_string())
    return df_template.to_dict('index')

async def run_interaction_study(args, base_data, top_adjectives, questions, client, pred_func):
    logging.info("\n--- PHASE 2: SECOND-ORDER INTERACTION STUDY ---")
    interaction_matrix = pd.DataFrame(index=top_adjectives, columns=top_adjectives, dtype=float)
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    for fixed_adj in tqdm(top_adjectives, desc="Interaction Study (Outer Loop)"):
        varied_adjectives = [adj for adj in top_adjectives if adj != fixed_adj]
        all_conditional_shaps = []
        
        tasks = [
            calculate_shap_for_question_subset(
                q, varied_adjectives, client, base_data['parameters']['model'],
                args.num_samples, semaphore, pred_func, fixed_adjectives=[fixed_adj]
            ) for q in questions
        ]

        for f in async_tqdm.as_completed(tasks, desc=f"  Context: '{fixed_adj}'", leave=False):
            all_conditional_shaps.append(await f)
            
        mean_shaps = np.mean(all_conditional_shaps, axis=0)
        for i, varied_adj in enumerate(varied_adjectives):
            interaction_matrix.loc[varied_adj, fixed_adj] = mean_shaps[i]
            
    print("\n--- Interaction Matrix (Conditional Mean Shapley Values) ---")
    print("Row=Adjective A, Col=Adjective B. Value = E[SHAP(A) | B is present]")
    print(interaction_matrix.to_string(float_format="%.4f"))
    return interaction_matrix.to_dict('index')

def get_api_client_and_function(params):
    """Helper to initialize API clients based on saved parameters."""
    provider = params.get('api_provider', 'openai')
    model_name = params['model']
    client, pred_func = None, None

    if provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY environment variable not set.")
        client = AsyncOpenAI(api_key=api_key)
        pred_func = get_openai_prediction
    elif provider == 'replicate':
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token: raise ValueError("REPLICATE_API_TOKEN environment variable not set.")
        os.environ['REPLICATE_API_TOKEN'] = api_token
        pred_func = get_replicate_prediction
    elif provider == 'gemini':
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model_name)
        pred_func = get_gemini_prediction
        
    if client is None and provider not in ['replicate']:
        raise ValueError(f"Could not initialize client for provider: {provider}")
        
    return client, pred_func

async def main(args):
    """Main execution function for follow-up studies."""
    if not args.input_results.exists():
        logging.error(f"Input file not found: {args.input_results}")
        return

    with open(args.input_results, 'r') as f:
        base_data = json.load(f)

    # --- Identify Adjectives for Study ---
    agg_df = pd.DataFrame(base_data['aggregated_results']).T.sort_values(
        'mean_abs_shapley', ascending=False
    )
    top_k_adjectives = agg_df.head(args.top_k).index.tolist()
    logging.info(f"Selected Top-{args.top_k} adjectives for study: {top_k_adjectives}")

    # --- Reload Original Data & Initialize API ---
    original_questions = get_original_questions(base_data)
    client, pred_func = get_api_client_and_function(base_data['parameters'])

    # --- Run Studies ---
    template_results = await run_template_study(args, base_data, top_k_adjectives, original_questions, client, pred_func)
    interaction_results = await run_interaction_study(args, base_data, top_k_adjectives, original_questions, client, pred_func)

    # --- Save Combined Results ---
    final_output = {
        'base_parameters': base_data['parameters'],
        'followup_parameters': {k:v for k,v in vars(args).items() if k != 'input_results'},
        'top_k_adjectives_studied': top_k_adjectives,
        'template_variance_results': template_results,
        'interaction_matrix_results': interaction_results
    }
    
    output_filename = Path(args.input_results.stem.replace('_summary', '') + "_followup_results.json")
    with open(output_filename, 'w') as f:
        json.dump(final_output, f, indent=2)
    logging.info(f"Follow-up analysis complete. Results saved to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run follow-up template and interaction analyses on SHAP results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--input_results", type=Path, required=True, help="Path to the JSON results file from the initial sampling script (e.g., gpt4o_summary.json).")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top adjectives (by absolute impact) to include in the studies.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of coalitions for SHAP. Can be lower for smaller feature sets.")
    parser.add_argument("--max_concurrent_requests", type=int, default=10, help="Maximum number of parallel API requests.")
    
    parser.epilog = (
        "Note: This script requires API keys to be set as environment variables:\n"
        "  - OPENAI_API_KEY\n"
        "  - REPLICATE_API_TOKEN\n"
        "  - GOOGLE_API_KEY\n"
        "The script will use the provider and model specified in the input results file."
    )

    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except (KeyboardInterrupt, ValueError) as e:
        logging.error(f"\nScript stopped. Reason: {e}")