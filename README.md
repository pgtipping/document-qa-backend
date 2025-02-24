# Document Q&A Backend

FastAPI service handling document processing and LLM integration for Document Q&A system.

## Features

- Multiple LLM provider support
- Document processing pipeline
- Caching system
- Performance metrics
- Error handling
- Health monitoring

## Quick Start

### Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the server
python run.py
```

### Environment Variables

```env
# LLM API Keys
GROQ_API_KEY=your_groq_api_key
TOGETHER_API_KEY=your_together_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000
```

## API Documentation

See [API Reference](../docs/api-reference.md)

## Deployment

### Heroku Deployment

1. Create Heroku app
2. Configure environment variables
3. Deploy using Git

## Development

- Built with FastAPI
- Python 3.11+
- Type hints throughout
- Comprehensive error handling
