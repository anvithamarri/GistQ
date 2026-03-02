# GistQ - Smart Summarization & Quiz Generator

A powerful AI-powered web application that summarizes text from multiple sources and automatically generates educational quizzes to test comprehension.

## Features

- **Multi-Source Support**: Summarize content from:
  - URLs (web pages)
  - PDF files
  - Text files
  - Direct text input

- **Intelligent Summarization**: 
  - Multiple summarization levels (abstract, summary, article)
  - AI-powered summaries using the BART model
  - Hierarchical processing for long documents
  - Automatic duplicate removal

- **Quiz Generation**:
  - Automatically generates educational quizzes based on summarized content
  - Powered by the Groq API with Llama 3.3 model
  - Customizable number of questions
  - Detailed answer explanations

- **Session Management**:
  - Save and retrieve previous summaries
  - Browse history in the sidebar
  - Delete sessions with one click
  - SQLite database for persistence

- **User-Friendly Interface**:
  - Clean, modern design
  - Responsive layout
  - Real-time progress updates
  - Easy navigation between summaries and quizzes

## Tech Stack

- **Backend**: Flask (Python web framework)
- **AI Models**: 
  - Hugging Face Transformers (BART for summarization)
  - Groq API (Llama 3.3 for quiz generation)
- **Database**: SQLAlchemy with SQLite
- **Content Processing**: BeautifulSoup4, PyPDF2, Requests
- **Frontend**: HTML/CSS/JavaScript

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd auto_summarizer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   Get your Groq API key from [https://console.groq.com](https://console.groq.com)

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

## Usage

### Summarizing Content

1. **Choose your input source**:
   - Enter a URL
   - Upload a PDF file
   - Upload a text file
   - Paste text directly

2. **Select summarization level**:
   - **Abstract**: ~60 words (ultra-concise)
   - **Summary**: ~130 words (standard)
   - **Article**: ~250 words (detailed)

3. **Generate summary** - The app will:
   - Extract text from your source
   - Process it with the BART model
   - Generate an optimized summary
   - Save it to your history

### Generating Quizzes

1. After summarizing content, click **"Generate Quiz"**
2. Specify the number of questions you want
3. Answer the questions (in quiz.html interface)
4. Review detailed explanations for each answer

### Viewing History

- All summaries are saved in the left sidebar
- Click any item to view its summary again
- Delete items with the delete button
- Sorted by creation date (newest first)

## Project Structure

```
auto_summarizer/
├── app.py                 # Main Flask application
├── summarizer.py          # Summarization logic (BART model)
├── models.py             # Database models
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── LICENSE              # License information
├── templates/
│   ├── index.html       # Main interface
│   └── quiz.html        # Quiz interface
└── instance/            # Instance data directory
```

## Performance Considerations

- Maximum file upload size: 10MB
- Automatically truncates very long texts for efficiency
- Uses hierarchical summarization for documents over 1024 tokens
- Caches tokenizer and model to reduce memory usage

## License

See LICENSE file for details.

## Future Enhancements

- Multi-language support
- Custom summarization parameters
- Export summaries to PDF/Word
- Collaborative features
- Integration with more AI models
- Batch processing

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
