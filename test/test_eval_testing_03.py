import unittest
from unittest.mock import MagicMock, patch, mock_open
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from eval_testing_03 import generate_questions, extract_relevant_info, evaluate_metrics, save_results_batch

class TestEvalTesting03(unittest.TestCase):

    def test_generate_questions(self):
        chunk = {
            "metadata": {
                "car_name": "Test Car",
                "price": "₹10 Lakh",
                "fuel_type": "Petrol",
                "city": "Delhi",
                "manufacturing_year": "2022"
            },
            "text": "Kms Driven: 50,000 Kms"
        }
        questions, answers = generate_questions(chunk)
        self.assertEqual(len(questions), 2)
        self.assertEqual(len(answers), 2)
        self.assertIn("Test Car", answers[0])
        self.assertIn("Petrol", answers[1])

    def test_extract_relevant_info(self):
        question = "Test question"
        response = "Test response"
        extracted = extract_relevant_info(question, response)
        self.assertEqual(extracted, response)

    @patch('eval_testing_03.bert_score')
    @patch('eval_testing_03.meteor_score')
    @patch('eval_testing_03.rouge_scorer.RougeScorer')
    def test_evaluate_metrics(self, mock_rouge_scorer, mock_meteor_score, mock_bert_score):
        mock_bert_score.return_value = (None, None, [0.8])
        mock_meteor_score.return_value = 0.7
        mock_rouge_scorer.return_value.score.return_value = {'rouge1': MagicMock(fmeasure=0.6)}

        bert, meteor, rouge, f1 = evaluate_metrics("ground truth", "generated text")
        self.assertEqual(bert, 0.8)
        self.assertEqual(meteor, 0.7)
        self.assertEqual(rouge, 0.6)
        self.assertIsInstance(f1, float)

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_results_batch(self, mock_json_dump, mock_file):
        results_batch = [{"test": "data"}]
        save_results_batch(results_batch, 1)
        mock_file.assert_called_once_with('Car Chatbot\data\evaluation_results_data_in_batches\evaluation_results_batch_1.json', 'w')
        mock_json_dump.assert_called_once()

if __name__ == '__main__':
    unittest.main()
