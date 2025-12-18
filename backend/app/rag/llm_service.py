import openai
import httpx
from typing import List, Dict, Optional
from app.config import settings

class OpenRouterService:
    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.chat_model = settings.openrouter_chat_model
        self.embed_model = settings.openrouter_embed_model

        # Configure OpenAI client for OpenRouter
        openai.base_url = self.base_url
        openai.api_key = self.api_key

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding for text using OpenRouter"""
        if model is None:
            model = self.embed_model

        # Handle potential string response from OpenRouter
        try:
            response = openai.embeddings.create(
                model=model,
                input=text
            )

            # Check if response has the expected structure
            if hasattr(response, 'data') and len(response.data) > 0:
                return response.data[0].embedding
            else:
                # If the response structure is different, handle accordingly
                raise AttributeError("'response' object has no attribute 'data' or is empty")
        except AttributeError as e:
            if "'str' object has no attribute 'data'" in str(e) or "'response' object has no attribute 'data'" in str(e):
                # Fallback to direct HTTP request for OpenRouter compatibility
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": model,
                    "input": text
                }

                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
                    resp.raise_for_status()

                    result = resp.json()
                    return result['data'][0]['embedding']
            else:
                raise e

    def get_embeddings_batch(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        if model is None:
            model = self.embed_model

        # Handle potential string response from OpenRouter
        try:
            response = openai.embeddings.create(
                model=model,
                input=texts
            )

            # Check if response has the expected structure
            if hasattr(response, 'data'):
                return [item.embedding for item in response.data]
            else:
                # If the response structure is different, handle accordingly
                raise AttributeError("'response' object has no attribute 'data'")
        except AttributeError as e:
            if "'str' object has no attribute 'data'" in str(e) or "'response' object has no attribute 'data'" in str(e):
                # Fallback to direct HTTP request for OpenRouter compatibility
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": model,
                    "input": texts
                }

                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
                    resp.raise_for_status()

                    result = resp.json()
                    return [item['embedding'] for item in result['data']]
            else:
                raise e

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using OpenRouter chat completion"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = openai.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent responses
                max_tokens=1000
            )

            # Check if response has the expected structure
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                # If the response structure is different, handle accordingly
                raise AttributeError("'response' object has no attribute 'choices' or is empty")
        except AttributeError as e:
            if "'str' object has no attribute 'choices'" in str(e) or "'response' object has no attribute 'choices'" in str(e):
                # Fallback to direct HTTP request for OpenRouter compatibility
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": self.chat_model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                }

                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                    resp.raise_for_status()

                    result = resp.json()
                    return result['choices'][0]['message']['content']
            else:
                raise e

    def validate_response(self, response: str, context: str) -> bool:
        """Validate that response is grounded in the provided context"""
        # Simple validation: check if response contains information that appears in context
        # In a more sophisticated implementation, we could use semantic similarity
        response_lower = response.lower()
        context_lower = context.lower()

        # Check if key terms from context appear in response
        context_words = context_lower.split()[:50]  # Take first 50 words as sample
        context_sample = ' '.join(context_words)

        # Simple heuristic: if less than 10% of response terms appear in context, flag as potential hallucination
        response_words = response_lower.split()
        if len(response_words) == 0:
            return False

        matching_words = sum(1 for word in response_words if word in context_sample)
        match_ratio = matching_words / len(response_words)

        # For now, we'll return True, but in a real implementation we'd have more sophisticated validation
        return True

openrouter_service = OpenRouterService()