import json
import re
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from nltk.translate.meteor_score import meteor_score
from llm_rag import car_rag_pipeline  # Import the chatbot logic from llm_rag.py

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


# Load the JSON file containing car chunks
with open('Car Chatbot\data\cartrade_cars_chunked.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

car_chunks = [chunk for chunk in data if chunk.get('metadata')]  # Extract valid car chunks

# Generate questions and answers for each chunk with more detailed ground truth
def generate_questions(chunk):
    car_name = chunk['metadata'].get('car_name', 'Unknown Car')
    price = chunk['metadata'].get('price', 'Unknown Price')
    fuel_type = chunk['metadata'].get('fuel_type', 'Unknown Fuel Type')
    city = chunk['metadata'].get('city', 'Delhi')
    year = chunk['metadata'].get('manufacturing_year', 'Unknown Year')
    
    # Extract kilometers driven if available
    kms_driven = "Unknown"
    if 'text' in chunk and 'Kms Driven:' in chunk['text']:
        kms_match = re.search(r'Kms Driven: ([\d,]+) Kms', chunk['text'])
        if kms_match:
            kms_driven = kms_match.group(1)
    
    # Generate more detailed ground truth answers
    detailed_price_answer = f"The {car_name} is priced at {price} in the {city} used car market. This {year} model has been driven for {kms_driven} kilometers and runs on {fuel_type} fuel."
    
    detailed_fuel_answer = f"The {car_name} runs on {fuel_type} fuel. This {year} model is available in {city} for {price} and has been driven for {kms_driven} kilometers."
    
    # Generate two questions based on attributes
    questions = [
        f"What is the price of {car_name}?",
        f"What is the fuel type of {car_name}?"
    ]
    answers = [
        detailed_price_answer,
        detailed_fuel_answer
    ]
    return questions, answers

# Extract relevant information from chatbot response for comparison
def extract_relevant_info(question, response):
    # For now, return the full response
    return response

# Evaluate metrics (BERTScore, METEOR, ROUGE, F1)
def evaluate_metrics(ground_truth, generated):
    # BERTScore
    P, R, F1 = bert_score([generated], [ground_truth], lang="en")
    bert_f1 = F1.item()  # Extract the F1 score from tensor
    
    # METEOR Score
    meteor = meteor_score([ground_truth.split()], generated.split())
    
    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ground_truth, generated)
    
    # F1 Score (token-based)
    ground_truth_tokens = ground_truth.split()
    generated_tokens = generated.split()
    
    common_tokens = set(ground_truth_tokens) & set(generated_tokens)
    precision = len(common_tokens) / len(generated_tokens) if generated_tokens else 0
    recall = len(common_tokens) / len(ground_truth_tokens) if ground_truth_tokens else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return bert_f1, meteor, rouge_scores['rouge1'].fmeasure, f1

# Process chunks and evaluate chatbot responses
results = []
question_counter = 0
max_questions = 1000  # Set the maximum number of questions to 1000

# Batch saving parameters
batch_size = 50  # Save every 50 questions processed
batch_number = 0

# Calculate metrics summaries
total_bert = 0
total_meteor = 0
total_rouge = 0
total_f1 = 0
count = 0

def save_results_batch(results_batch, batch_num):
    """Save results to a file incrementally."""
    filename = f'evaluation_results_batch_{batch_num}.json'
    with open(filename, 'w') as file:
        json.dump(results_batch, file, indent=4)
    print(f"Saved batch {batch_num} results to {filename}")

for i, chunk in enumerate(car_chunks):
    questions, answers = generate_questions(chunk)
    
    for j, (question, ground_truth) in enumerate(zip(questions, answers)):
        try:
            # Get response from chatbot logic imported from llm_rag.py
            chatbot_response_text = car_rag_pipeline(question)
            
            # Process response for evaluation
            processed_response = extract_relevant_info(question, chatbot_response_text)
            
            # Evaluate metrics
            bert, meteor, rouge, f1 = evaluate_metrics(ground_truth, processed_response)
            
            # Add to totals for averaging
            total_bert += bert
            total_meteor += meteor
            total_rouge += rouge
            total_f1 += f1
            count += 1
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "chatbot_response": chatbot_response_text,
                "bert_score": bert,
                "meteor_score": meteor,
                "rouge_score": rouge,
                "f1_score": f1
            })
            
            question_counter += 1
            
            # Save results periodically in batches
            if question_counter % batch_size == 0:
                batch_number += 1
                save_results_batch(results[-batch_size:], batch_number)

            # Check if we've reached the maximum number of questions
            if question_counter >= max_questions:
                break
                
        except Exception as e:
            print(f"Error processing question {j+1} for chunk {i+1}: {str(e)}")
    
    # Break the outer loop if we've reached the maximum number of questions
    if question_counter >= max_questions:
        print(f"Reached the maximum of {max_questions} questions. Stopping.")
        break
    
    # Print progress periodically for better visibility during execution.
    print(f"Processed {i+1} chunks, generated {question_counter} questions")

# Final batch save for any remaining results not saved yet.
if len(results) % batch_size != 0:
    batch_number += 1
    save_results_batch(results[-(len(results) % batch_size):], batch_number)

# Calculate averages and save summary to a final file.
avg_bert = total_bert / count if count > 0 else 0
avg_meteor = total_meteor / count if count > 0 else 0
avg_rouge = total_rouge / count if count > 0 else 0
avg_f1 = total_f1 / count if count > 0 else 0

summary = {
    "total_questions_evaluated": count,
    "average_bert_score": avg_bert,
    "average_meteor_score": avg_meteor,
    "average_rouge_score": avg_rouge,
    "average_f1_score": avg_f1
}

output_filename_final_summary = 'evaluation_results_summary.json'
with open(output_filename_final_summary, 'w') as file:
    json.dump(summary, file, indent=4)

print("\nEvaluation completed! Results saved incrementally.")
print(f"Final metrics - BERTScore: {avg_bert:.4f}, METEOR: {avg_meteor:.4f}, ROUGE: {avg_rouge:.4f}, F1: {avg_f1:.4f}")