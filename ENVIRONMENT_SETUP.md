# Environment Setup

## API Keys Configuration

To use API keys safely in this project:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your API keys:
   ```bash
   OPENROUTER_API_KEY=your_actual_openrouter_api_key_here
   ```

3. The `.env` file is already included in `.gitignore` and will not be committed to the repository.

## Important Security Notice

Never commit actual API keys to the repository. Always use environment variables or other secure secret management practices.

Your API key has been exposed in the chat history. For security reasons, you should:
1. Regenerate your API key immediately from the OpenRouter dashboard
2. Never share API keys in plain text again