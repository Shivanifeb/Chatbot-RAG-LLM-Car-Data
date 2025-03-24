import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import unittest
from unittest.mock import patch, MagicMock
from llm_rag import initialize_clients, clean_price, retrieve_context, format_context_for_llm, generate_answer_with_llm, parse_query_for_filters, car_rag_pipeline

class TestLLMRAG(unittest.TestCase):

    @patch('llm_rag.chromadb.PersistentClient')
    @patch('llm_rag.genai.configure')
    def test_initialize_clients(self, mock_genai_configure, mock_persistent_client):
        collection, genai = initialize_clients()
        self.assertIsNotNone(collection)
        self.assertIsNotNone(genai)

    def test_clean_price(self):
        self.assertEqual(clean_price("₹ 32.8 Lakh"), "₹32.8 Lakh (₹3,280,000.00)")
        self.assertEqual(clean_price("₹ 1.5 Crore"), "₹1.5 Crore (₹15,000,000.00)")

    @patch('llm_rag.chromadb.PersistentClient')
    def test_retrieve_context(self, mock_persistent_client):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        contexts = retrieve_context(mock_collection, "test query")
        self.assertEqual(contexts, [])

    def test_format_context_for_llm(self):
        contexts = [
            {
                "car_name": "Test Car",
                "price": "₹ 10 Lakh",
                "city": "Delhi",
                "fuel_type": "Petrol",
                "manufacturing_year": "2022",
                "url": "https://example.com",
                "content": "Test content"
            }
        ]
        formatted = format_context_for_llm(contexts)
        self.assertIn("Test Car", formatted)
        self.assertIn("₹10 Lakh (₹1,000,000.00)", formatted)

    @patch('llm_rag.genai.GenerativeModel')
    def test_generate_answer_with_llm(self, mock_generative_model):
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "Test answer"
        mock_generative_model.return_value = mock_model

        client = MagicMock()
        answer = generate_answer_with_llm(client, "test query", "test context")
        self.assertEqual(answer, "Test answer")

    def test_parse_query_for_filters(self):
        query = "Honda Civic 2022 in Delhi"
        filters = parse_query_for_filters(query)
        self.assertIn({"car_name": {"$contains": "Honda"}}, filters["$and"])
        self.assertIn({"city": "Delhi"}, filters["$and"])
        self.assertIn({"manufacturing_year": "2022"}, filters["$and"])

    @patch('llm_rag.initialize_clients')
    @patch('llm_rag.retrieve_context')
    @patch('llm_rag.generate_answer_with_llm')
    def test_car_rag_pipeline(self, mock_generate_answer, mock_retrieve_context, mock_initialize_clients):
        mock_initialize_clients.return_value = (MagicMock(), MagicMock())
        mock_retrieve_context.return_value = [{"content": "Test content"}]
        mock_generate_answer.return_value = "Test answer"

        answer = car_rag_pipeline("test query")
        self.assertEqual(answer, "Test answer")

if __name__ == '__main__':
    unittest.main()
