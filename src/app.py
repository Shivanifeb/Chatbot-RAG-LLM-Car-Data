import streamlit as st
from llm_rag import car_rag_pipeline

def main():
    st.title("Car RAG System")
    query = st.text_input("Enter your car-related question:")
    
    if st.button("Submit"):
        answer = car_rag_pipeline(query)
        st.write(answer)

if __name__ == "__main__":
    main()