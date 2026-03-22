# RAG BOT - Intelligent Document Assistant

A modern, full-stack RAG (Retrieval-Augmented Generation) application that allows you to chat with your documents (PDF, TXT) and web content in real-time.

## ✨ Features

- **Document Ingestion**: Upload PDF and Text files for instant analysis.
- **Web Ingestion**: Paste a URL to scrape and chat with website content.
- **Interactive UI**: Sleek, translucent sidebar with backdrop blur and an animated shader background.
- **Persistent Chat**: Full history management with the ability to create, delete, and switch between sessions.
- **Source Transparency**: View the exact context used by the AI to generate answers.

## 🚀 Tech Stack

- **Frontend**: React, Tailwind CSS, Lucide Icons, Axios.
- **Backend**: Python (FastAPI/Uvicorn), RAG Pipeline (using LLMs for retrieval and reasoning).
- **Styling**: Modern dark-themed aesthetics with glassmorphism effects.

## 🛠️ Getting Started

### Backend Setup
1. Navigate to the `backend` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup
1. Navigate to the `frontend` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

## 📂 Project Structure

- `/frontend`: React application and modern UI components.
- `/backend`: FastAPI server and core RAG logic.
- `/backend/data`: (Excluded from Git) Local storage for processed document indices.

## 📄 License

MIT
