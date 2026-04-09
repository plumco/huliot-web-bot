import google.generativeai as genai

genai.configure(api_key="AIzaSyAWfBkPc1n2VeJfGrmkskLMRg6YxhHj4QQ")

print("YOUR AVAILABLE MODELS:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)