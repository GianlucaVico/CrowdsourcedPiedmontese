from openai import OpenAI
from google import genai
import os
import tiktoken
import dotenv
dotenv.load_dotenv()
import ipdb

CLOSED_MODELS = {
    "google/gemini-2.5-flash-preview-09-2025",
    "openai/gpt-4o-mini",
}

class ClosedModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
    def __call__(self, msgs, do_sample=False, max_new_tokens=10):
        max_new_tokens = max(max_new_tokens, 16) # GPT-5 minimum
        for msg in msgs:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_completion_tokens=max_new_tokens,
                temperature=0.0,
                n=1,
                messages=msg,
                # extra_body={"reasoning": {
                #     "effort": "low"
                # }}
            )
            answer = completion.choices[0].message.content
            print(answer)
            yield [{
                "generated_text": [{"content": answer}]
            }]

class GeminiTokenCounter:
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-09-2025"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  
    def __call__(self, texts: list[str], *args, **kwargs):
        counts = []
        for text in texts:
            total_tokens = self.client.models.count_tokens(
                model=self.model_name, contents=text
            )
            counts.append(total_tokens.total_tokens)
        return {"length": counts}
    
class GPTTokenCounter:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
    def __call__(self, texts: list[str], *args, **kwargs):
        counts = [len(self.tokenizer.encode(text)) for text in texts]
        return {"length": counts}