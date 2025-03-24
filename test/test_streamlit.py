import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import unittest
from unittest.mock import patch
import streamlit as st
from streamlit import car_rag_pipeline

class TestStreamlit(unittest.TestCase):

    @patch('streamlit.text_input')
    @patch('streamlit.button')
    @patch('streamlit.write')
    @patch('llm_rag.car_rag_pipeline')
    def test_main(self, mock_car_rag_pipeline, mock_write, mock_button, mock_text_input):
        mock_text_input.return_value = "Test query"
        mock_button.return_value = True
        mock_car_rag_pipeline.return_value = "Test answer"

        import streamlit as st
        st.main()

        mock_text_input.assert_called_once_with("Enter your car-related question:")
        mock_button.assert_called_once_with("Submit")
        mock_car_rag_pipeline.assert_called_once_with("Test query")
        mock_write.assert_called_once_with("Test answer")

if __name__ == '__main__':
    unittest.main()
