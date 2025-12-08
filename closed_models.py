from openai import OpenAI
import os
import dotenv
dotenv.load_dotenv()

CLOSED_MODELS = {
    "google/gemini-2.5-flash-preview-09-2025",
    "openai/gpt-5.1",
}

class ClosedModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
    def __call__(self, msgs, do_sample=False, max_new_tokens=10):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_completion_tokens=max_new_tokens,
            temperature=0.0,
            n=1,
            messages=msgs,
            # extra_body={"reasoning": {"enabled": False}}
        )
        answer = completion.choices[0].message.content
        return [{
            "generated_text": [{"content": answer}]
        }]



# from google import genai

# client = genai.Client()
# prompt = "The quick brown fox jumps over the lazy dog."

# # Count tokens using the new client method.
# total_tokens = client.models.count_tokens(
#     model="gemini-2.0-flash", contents=prompt
# )
# print("total_tokens: ", total_tokens)
# # ( e.g., total_tokens: 10 )

# response = client.models.generate_content(
#     model="gemini-2.0-flash", contents=prompt
# )

# # The usage_metadata provides detailed token counts.
# print(response.usage_metadata)
# # ( e.g., prompt_token_count: 11, candidates_token_count: 73, total_token_count: 84 )