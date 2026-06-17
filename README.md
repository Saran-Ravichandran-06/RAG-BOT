# RAG BOT - Intelligent Document Assistant

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

A modern, full-stack RAG (Retrieval-Augmented Generation) application that allows you to chat with your documents and web content in real-time.

## ✨ Features

- **Real-Time Streaming**: Live token-by-token text streaming via native Fetch API & Server-Sent Events (SSE).
- **GPU Acceleration**: Built-in diagnostic endpoints verifying Ollama GPU utilization via `nvidia-smi` to ensure high-performance CUDA inference.
- **Document & Web Ingestion**: Upload PDF/Text files or paste a URL to instantly scrape and chat with the content.
- **Interactive Glassmorphism UI**: Sleek, translucent sidebar and chat bubbles with a responsive, animated WebGL shader background.
- **Persistent Chat Sessions**: Full history management with the ability to create, delete, and switch between multiple conversations.
- **Source Transparency**: View the exact context used by the AI to generate answers along with an evaluation badge.

## 🚀 Tech Stack

- **Frontend**: React, Vite, Tailwind CSS, Lucide Icons.
- **Backend**: Python, FastAPI, Uvicorn, Ollama, Sentence-Transformers.
- **Styling**: Modern dark-themed aesthetics with frosted glassmorphism effects.

## 🛠️ Getting Started

### Backend Setup
1. Navigate to the `backend` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server (with hot-reload):
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup
1. Navigate to the `frontend` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the Vite development server:
   ```bash
   npm run dev
   ```

## 📂 Project Structure

- `/frontend`: React + Vite application containing modern UI components and state management.
- `/backend`: FastAPI server handling embeddings, chunking, SSE streaming, and Ollama integration.
- `/backend/data`: Local storage for processed document indices and chat metadata.

## 📄 License

MIT
