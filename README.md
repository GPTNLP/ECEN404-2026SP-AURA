# GPT-NLP-Lab-Assistant
2025-26 Texas A&amp;M ELEN Senior Design Project

Code for an AI-powered robotic lab assistant that utilizes recent advancements in Retrieval-Augmented Generation to answer questions based on provided lab documents & lectures.

# Features
- **Local LLM Integration**: Utilizes Ollama to run language models locally, removing the need for internet connection.
- **Document Processing**: Supports PDF File type.
- **Vector Database**: Persistent database for efficient retrieval.
- **LightRAG Queries**: Utilizes LightRAG for enhanced retrieval with evidence scoring & transparency.
- **Session Management**: Save & load conversation sessions remotely.

# Architecture
This project combines two seperate approaches for this custom localLLM + LightRAG based project:
- Base implementation inspired by https://github.com/amscotti/local-LLM-with-RAG
- Enhanced retrieval inspired by https://github.com/HKUDS/LightRAG

# Prerequisires
1. Ollama from Meta: https://ollama.ai/
2. Python 3.13+ for all dependencies (found on pythons website)
3. Imports from requirements.txt (Please run 'pip install -r requirements.txt')
