# AURA: GPT-NLP-Lab-Assistant
**2025-26 Texas A&M ELEN Senior Design Project**

Code for an AI-powered robotic lab assistant that utilizes recent advancements in Retrieval-Augmented Generation (RAG) to answer questions based on provided lab documents and lectures. 

Unlike a standard software chatbot, AURA integrates a physical hardware platform (NVIDIA Jetson & ESP32) with a full-stack web dashboard to manage the robot, users, and knowledge base remotely.

## 🌟 Key Features

* **Edge AI Integration**: Utilizes a local NVIDIA Jetson device running Ollama to execute LLMs completely offline, ensuring fast and private inference.
* **Enhanced Knowledge Retrieval (LightRAG)**: Integrates LightRAG for advanced vector-based document queries, providing evidence scoring and transparent answers.
* **Computer Vision & Hardware Integration**: 
    * Uses a camera with YOLOv11 for live face detection.
    * Interfaces with an ESP32 via serial link for physical robot control and battery monitoring.
    * Local Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities for conversational interaction.
* **Comprehensive Web Dashboard**: A React/TypeScript frontend and Python backend to manage:
    * **Role-Based Access**: Student, TA, and Admin portals.
    * **Fleet Management**: Monitor robot status, camera feeds, and battery metrics.
    * **Database Management**: Upload PDFs, build vector databases, and sync them to the physical robot.
    * **Session Management**: Save, view, and load remote conversation histories.

## 🏗️ Architecture Stack

This project is divided into three primary domains:

1.  **Website (Frontend & Cloud Backend)**
    * *Frontend*: React + Vite + TypeScript.
    * *Backend*: Python backend API handling user authentication, device telemetry, and cloud storage syncing.
2.  **JetsonLocal (Edge Agent)**
    * Python-based agent running on an NVIDIA Jetson (Orin/Nano).
    * Manages local RAG vector stores, the Ollama inference engine, camera pipelines, and cloud synchronization.
3.  **Firmware (ESP32)**
    * C++ code managed via PlatformIO for low-level motor control, sensor readings, and serial communication with the Jetson.

*Note: The RAG implementation is inspired by [local-LLM-with-RAG](https://github.com/amscotti/local-LLM-with-RAG) and enhanced retrieval by [LightRAG](https://github.com/HKUDS/LightRAG).*

## 🛠️ Prerequisites

Because this project spans multiple platforms, prerequisites depend on which part of the system you are working on:

### Global / Local LLM
1.  **Ollama**: Install from [https://ollama.ai/](https://ollama.ai/)
2.  **Models**: Pull the required default models (`llama3.2` for text generation, `nomic-embed-text` for embeddings).

### Edge Device (JetsonLocal)
1.  **Python 3.13+**
2.  Install requirements: `pip install -r JetsonLocal/requirements.txt`
3.  *(Optional/Jetson Specific)*: Run `JetsonLocal/setup_orin.sh` for environment setup.

### Web Infrastructure (Website)
1.  **Node.js & npm**: For the React frontend (`cd Website && npm install`).
2.  **Python 3.13+**: For the backend API (`pip install -r Website/backend/requirements.txt`).

### Firmware (ESP32)
1.  **PlatformIO**: Recommended to use as a VS Code extension to build and flash the C++ code found in `JetsonLocal/firmware`.