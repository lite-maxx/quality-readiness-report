import pandas as pd
import numpy as np
from anthropic import AnthropicVertex
import random
import json
import re
import time
import streamlit as st
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import List, Dict

# Initialize AnthropicVertex
anthropic = AnthropicVertex(
    project_id="bloomlocalpim",
    region="us-east5"
)

# Global variables for rate limiting
RATE_LIMIT = 1  # Default rate limit (calls per second)
API_CALLS = []

def rate_limited_api_call(prompt):
    global API_CALLS
    current_time = time.time()
    API_CALLS = [call for call in API_CALLS if current_time - call < 1]
    if len(API_CALLS) >= RATE_LIMIT:
        time.sleep(1 - (current_time - API_CALLS[0]))
    API_CALLS.append(time.time())
    
    return make_api_call(prompt)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def make_api_call(prompt):
    try:
        response = anthropic.messages.create(
            model="claude-3-5-sonnet@20240620",
            messages=prompt,
            max_tokens=1024,
            temperature=0
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        raise

def extract_json(text):
    try:
        # Try parsing directly first
        return json.loads(text)
    except json.JSONDecodeError:
        # Remove leading/trailing whitespace
        text = text.strip()

        # Fix cases where quotation marks appear inside the values
        # This will ensure that quotes are properly escaped or removed when necessary
        def fix_quotes(text):
            # Match patterns like: "value " with or similar incomplete quote issues
            text = re.sub(r'"\s*([^"]*?)\s*"\s+with', r'"\1 with', text)
            return text

        # Apply the fix to the text
        cleaned_text = fix_quotes(text)

        # Try parsing the cleaned text
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        # If direct parsing fails, look for JSON-like structure in the text
        match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if match:
            json_candidate = match.group(0)

            # Apply quote fix again on the extracted candidate
            json_candidate = fix_quotes(json_candidate)

            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                st.warning(f"Failed to parse JSON: {json_candidate}")
                return None
        else:
            st.warning(f"No JSON found in the response: {text}")
            return None
        
        
def analyze_quality(text, is_original=True):
    if pd.isna(text):
        return {"overall_quality_score": 0, "spelling_mistakes": 0, "formatting_errors": 0, "language_unformity_score":0, "human_perception_score": 0}

    prompt = [
        {
            "role": "user",
            "content": f"""Assume the role of a Data Quality Analyst.

            You are tasked with analysing a dataset of product data catalog.

            You have to assign a score of overall_quality_score between 0-100 based on following parameters-
                1. spelling_mistakes - How many spelling mistakes are there?
                2. formatting_errors - How many formatting errors are there? A formatting error can be like an incorrect date format, incorrect case (for an item name, it should be title case)
                3. language_uniformity_score - Does the text has content in non-EN language/script?
                4. human_perception_score - How well-structured and comprehensible is the content for humans to perceive efficiently?
            
            Analyze the quality of the following text:
            {text}

            This is from the {"original" if is_original else "enriched"} dataset.
            {"Be more strict and penalizing in your scoring for the original dataset." if is_original else ""}

            Return a JSON object with the following structure:
            {{
                "overall_quality_score": <0-100>,
                "spelling_mistakes": <count>,
                "formatting_errors": <count>,
                "language_uniformity_score": <0-100>,
                "human_perception_score": <0-100>,
                "explanation": "<brief explanation of the overall quality>"
            }}"""
        }
    ]
    
    response = rate_limited_api_call(prompt)
    result = extract_json(response)
    return result if result else {"overall_quality_score": 0, "spelling_mistakes": 0, "formatting_errors": 0,"language_uniformity_score":0, "human_perception_score": 0, "explanation": "Failed to analyze"}

def analyze_relevance(item_name, field_value, field_name, is_original=True):
    if pd.isna(item_name) or pd.isna(field_value):
        return {"relevance_score": 0, "explanation": "Missing data"}

    prompt = [
        {
            "role": "user",
            "content": f""" Assume the role of a Data Quality Analyst.

            You are tasked with analysing a dataset of product data catalog.

            You are provided with the name of an item, and you have to analyse the relevance of the content present for the given field name like description, brand name, category, keywords, etc.

            You have to assign a relevance score of 0-100 based on how well the content would be able to convey the context to a retail online shopping customer for the given item.
            
            Analyze the relevance of the following {field_name} to the item name:
            Item Name: {item_name}
            {field_name}: {field_value}

                        This is from the {"original" if is_original else "enriched"} dataset.
            {"Be more strict and penalizing in your scoring for the original dataset." if is_original else ""}

            Return a JSON object with the following structure:
            {{
                "relevance_score": <0-100>,
                "explanation": "<brief explanation of the relevance>"
            }}"""
        }
    ]
    
    response = rate_limited_api_call(prompt)
    result = extract_json(response)
    return result if result else {"relevance_score": 0, "explanation": "Failed to analyze"}

def analyze_row(row, fields, is_original=True):
    results = {}
    for field in fields:
        if field in row:
            quality_result = analyze_quality(row[field], is_original)
            relevance_result = analyze_relevance(row['properties_name'], row[field], field, is_original)
            results[field] = {
                'quality': quality_result,
                'relevance': relevance_result
            }
    return results

def process_batch(batch: pd.DataFrame, fields: List[str], is_original: bool) -> List[Dict]:
    results = []
    for _, row in batch.iterrows():
        results.append(analyze_row(row, fields, is_original))
    return results

def compare_datasets(df_og, df_enr, max_items=None, batch_size=None):
    all_fields = ['properties_name', 'properties_description', 'brand_name', 'properties_category', 'properties_c__keywords']
    results = []

    df_og_limited = df_og.head(max_items) if max_items else df_og
    df_enr_limited = df_enr.head(max_items) if max_items else df_enr

    if batch_size:
        og_results = []
        enr_results = []
        total_batches = (len(df_og_limited) + batch_size - 1) // batch_size
        for i in range(0, len(df_og_limited), batch_size):
            og_batch = df_og_limited.iloc[i:i+batch_size]
            enr_batch = df_enr_limited.iloc[i:i+batch_size]
            og_results.extend(process_batch(og_batch, all_fields, True))
            enr_results.extend(process_batch(enr_batch, all_fields, False))
            progress = (i + batch_size) / len(df_og_limited)
            st.progress(progress)
            st.write(f"Processed batch {(i // batch_size) + 1} of {total_batches}")
    else:
        og_results = process_batch(df_og_limited, all_fields, True)
        enr_results = process_batch(df_enr_limited, all_fields, False)

    for field in all_fields:
        og_field_results = [result[field] for result in og_results if field in result]
        enr_field_results = [result[field] for result in enr_results if field in result]

        og_completeness = sum(1 for result in og_field_results if result)
        enr_completeness = sum(1 for result in enr_field_results if result)

        og_quality_scores = [result['quality']['overall_quality_score'] for result in og_field_results if result]
        enr_quality_scores = [result['quality']['overall_quality_score'] for result in enr_field_results if result]

        og_relevance_scores = [result['relevance']['relevance_score'] for result in og_field_results if result]
        enr_relevance_scores = [result['relevance']['relevance_score'] for result in enr_field_results if result]

        results.append({
            'Field': field,
            'Original Completeness': og_completeness,
            'Enriched Completeness': enr_completeness,
            'Original Quality Score': round(np.mean(og_quality_scores), 1) if og_quality_scores else 0,
            'Enriched Quality Score': round(np.mean(enr_quality_scores), 1) if enr_quality_scores else 0,
            'Original Relevance Score': round(np.mean(og_relevance_scores), 1) if og_relevance_scores else 0,
            'Enriched Relevance Score': round(np.mean(enr_relevance_scores), 1) if enr_relevance_scores else 0,
        })

    return pd.DataFrame(results)

def summarize_explanations(explanations):
    prompt = [
        {
            "role": "user",
            "content": f"""Summarize the following explanations into a concise gist:
            {explanations}
            
            Provide a brief summary that captures the main points across all explanations."""
        }
    ]
    
    response = rate_limited_api_call(prompt)
    return response.strip()

def create_snapshots(df_og, df_enr, sample_size=5):
    all_fields = ['properties_name', 'properties_description', 'brand_name', 'properties_category', 'properties_c__keywords']
    snapshots = []
    
    common_ids = set(df_og['id']) & set(df_enr['id'])
    sample_ids = random.sample(list(common_ids), min(sample_size, len(common_ids)))
    
    for id in sample_ids:
        og_row = df_og[df_og['id'] == id].iloc[0]
        enr_row = df_enr[df_enr['id'] == id].iloc[0]
        
        snapshot = {
            'Item ID': id,
            'Field': [],
            'Original Value': [],
            'Enriched Value': [],
            'Original Quality': [],
            'Enriched Quality': [],
            'Original Relevance': [],
            'Enriched Relevance': []
        }
        
        for field in all_fields:
            snapshot['Field'].append(field)
            og_value = og_row.get(field, 'N/A')
            enr_value = enr_row.get(field, 'N/A')
            snapshot['Original Value'].append(og_value)
            snapshot['Enriched Value'].append(enr_value)
            
            og_quality = analyze_quality(og_value, is_original=True)
            enr_quality = analyze_quality(enr_value, is_original=False)
            snapshot['Original Quality'].append(og_quality)
            snapshot['Enriched Quality'].append(enr_quality)
            
            og_relevance = analyze_relevance(og_row['properties_name'], og_value, field, is_original=True)
            enr_relevance = analyze_relevance(enr_row['properties_name'], enr_value, field, is_original=False)
            snapshot['Original Relevance'].append(og_relevance)
            snapshot['Enriched Relevance'].append(enr_relevance)
        
        snapshots.append(snapshot)
    
    return snapshots

def save_results(comparison_results, snapshots):
    output_dir = os.path.join(os.getcwd(), "arabie_analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comparison_file = os.path.join(output_dir, f"comparison_results_{timestamp}.csv")
    comparison_results.to_csv(comparison_file, index=False)

    snapshots_file = os.path.join(output_dir, f"snapshots_{timestamp}.json")
    with open(snapshots_file, 'w') as f:
        json.dump(snapshots, f, indent=2)

    return comparison_file, snapshots_file

def main():
    global RATE_LIMIT
    st.set_page_config(page_title="Arabie Data Analysis", layout="wide")
    st.title("Arabie Data Analysis")

    st.header("Upload Data Files")
    col1, col2 = st.columns(2)
    with col1:
        og_file = st.file_uploader("Upload original dataset", type=["csv", "xlsx"])
    with col2:
        enr_file = st.file_uploader("Upload enriched dataset", type=["csv", "xlsx"])

    if og_file and enr_file:
        try:
            df_og = pd.read_csv(og_file) if og_file.name.endswith('.csv') else pd.read_excel(og_file)
            df_enr = pd.read_csv(enr_file) if enr_file.name.endswith('.csv') else pd.read_excel(enr_file)

            df_og = df_og.drop_duplicates()
            df_enr = df_enr.drop_duplicates()
            st.write(f"Loaded {len(df_og)} rows from original dataset and {len(df_enr)} rows from enriched dataset")

            st.write("Original dataset columns:", df_og.columns.tolist())
            st.write("Enriched dataset columns:", df_enr.columns.tolist())

            st.subheader("Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                use_batch_processing = st.radio("Use batch processing?", ('No', 'Yes'), index=0)
                if use_batch_processing == 'Yes':
                    num_batches = st.number_input("Number of batches", min_value=1, value=1)
                else:
                    num_batches = 1
                
                RATE_LIMIT = st.number_input("LLM API rate limit (calls per second)", min_value=1, value=1)
            
            with col2:
                max_items = st.number_input("Number of items/rows to process (0 for all)", min_value=0, value=0)
                max_items = max_items if max_items > 0 else None

            if st.button("Compare Datasets"):
                with st.spinner("Comparing datasets... This may take a while."):
                    batch_size = len(df_og) // num_batches if use_batch_processing == 'Yes' else None
                    comparison_results = compare_datasets(df_og, df_enr, max_items=max_items, batch_size=batch_size)
                    snapshots = create_snapshots(df_og, df_enr)

                st.header("Data Quality Analysis Report")
                st.subheader("Comparison Metrics")
                st.dataframe(comparison_results.style.set_properties(**{'text-align': 'left', 'white-space': 'pre-wrap'}))

                st.subheader("Explanations")
                all_fields = ['properties_name', 'properties_description', 'brand_name', 'properties_category', 'properties_c__keywords']
                for field in all_fields:
                    with st.expander(f"Explanations for {field}"):
                        og_quality_explanations = [snapshot['Original Quality'][i]['explanation'] for snapshot in snapshots for i, f in enumerate(snapshot['Field']) if f == field]
                        enr_quality_explanations = [snapshot['Enriched Quality'][i]['explanation'] for snapshot in snapshots for i, f in enumerate(snapshot['Field']) if f == field]
                        og_relevance_explanations = [snapshot['Original Relevance'][i]['explanation'] for snapshot in snapshots for i, f in enumerate(snapshot['Field']) if f == field]
                        enr_relevance_explanations = [snapshot['Enriched Relevance'][i]['explanation'] for snapshot in snapshots for i, f in enumerate(snapshot['Field']) if f == field]

                        st.write("Original Quality Summary:", summarize_explanations(og_quality_explanations))
                        st.write("Enriched Quality Summary:", summarize_explanations(enr_quality_explanations))
                        st.write("Original Relevance Summary:", summarize_explanations(og_relevance_explanations))
                        st.write("Enriched Relevance Summary:", summarize_explanations(enr_relevance_explanations))

                st.subheader("Sample Snapshots")
                for i, snapshot in enumerate(snapshots):
                    with st.expander(f"Snapshot {i+1} - Item ID: {snapshot['Item ID']}"):
                        for j, field in enumerate(snapshot['Field']):
                            st.write(f"Field: {field}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Original:")
                                st.write(f"Value: {snapshot['Original Value'][j]}")
                                st.write(f"Quality Score: {snapshot['Original Quality'][j]['overall_quality_score']}")
                                st.write(f"Relevance Score: {snapshot['Original Relevance'][j]['relevance_score']}")
                            with col2:
                                st.write("Enriched:")
                                st.write(f"Value: {snapshot['Enriched Value'][j]}")
                                st.write(f"Quality Score: {snapshot['Enriched Quality'][j]['overall_quality_score']}")
                                st.write(f"Relevance Score: {snapshot['Enriched Relevance'][j]['relevance_score']}")

                comparison_file, snapshots_file = save_results(comparison_results, snapshots)
                st.success(f"Results saved to {comparison_file} and {snapshots_file}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()