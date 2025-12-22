# MCP (Model Context Protocol) and Free AI API Integration

## 📖 Table of Contents
1. [What is MCP?](#what-is-mcp)
2. [Setting Up MCP](#setting-up-mcp)
3. [Free AI APIs Available](#free-ai-apis-available)
4. [Connecting to AI APIs](#connecting-to-ai-apis)
5. [Customization and Fine-tuning](#customization-and-fine-tuning)
6. [Testing AI Models](#testing-ai-models)
7. [Complete Examples](#complete-examples)

---

## What is MCP?

### Model Context Protocol (MCP)
- **Protocol**: Standard way to interact with AI models
- **Context Management**: Handles conversation context
- **API Integration**: Connects to various AI services
- **Customization**: Allows model customization

### Key Features
- **Unified Interface**: Same interface for different AI providers
- **Context Preservation**: Maintains conversation history
- **Streaming Support**: Real-time responses
- **Error Handling**: Robust error management

---

## Setting Up MCP

### Installation

```bash
# Install MCP SDK
pip install mcp-sdk

# Or install from source
git clone https://github.com/modelcontextprotocol/mcp-sdk
cd mcp-sdk
pip install -e .
```

### Basic Setup

```python
from mcp import MCPClient, MCPConfig

# Configure MCP
config = MCPConfig(
    api_key="your-api-key",
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000
)

# Create client
client = MCPClient(config)
```

---

## Free AI APIs Available

### 1. OpenAI (Free Tier Available)

```python
# OpenAI API (has free tier with credits)
import openai

openai.api_key = "your-api-key"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### 2. Hugging Face Inference API (Free)

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "The future of AI is",
    "parameters": {
        "max_length": 100,
        "temperature": 0.7
    }
})

print(output)
```

### 3. Cohere API (Free Tier)

```python
import cohere

co = cohere.Client("your-api-key")

response = co.generate(
    model='command',
    prompt='Write a story about AI',
    max_tokens=200,
    temperature=0.7
)

print(response.generations[0].text)
```

### 4. Anthropic Claude (Free Tier)

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content[0].text)
```

### 5. Google Gemini (Free)

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("Explain machine learning")
print(response.text)
```

### 6. Local Models (Completely Free)

```python
# Using Ollama (runs models locally)
import requests

def query_ollama(prompt, model="llama2"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Usage
response = query_ollama("What is AI?")
print(response)
```

---

## Connecting to AI APIs

### Unified API Wrapper

```python
class AIAPIClient:
    def __init__(self, provider="openai", api_key=None):
        self.provider = provider
        self.api_key = api_key
        self._setup_client()
    
    def _setup_client(self):
        if self.provider == "openai":
            import openai
            openai.api_key = self.api_key
            self.client = openai
        elif self.provider == "huggingface":
            import requests
            self.client = requests
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
        elif self.provider == "cohere":
            import cohere
            self.client = cohere.Client(self.api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel('gemini-pro')
    
    def generate(self, prompt, max_tokens=100, temperature=0.7):
        if self.provider == "openai":
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        
        elif self.provider == "huggingface":
            response = self.client.post(
                "https://api-inference.huggingface.co/models/gpt2",
                headers=self.headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_length": max_tokens,
                        "temperature": temperature
                    }
                }
            )
            return response.json()[0]["generated_text"]
        
        elif self.provider == "cohere":
            response = self.client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.generations[0].text
        
        elif self.provider == "gemini":
            response = self.client.generate_content(prompt)
            return response.text

# Usage
client = AIAPIClient(provider="openai", api_key="your-key")
response = client.generate("Explain AI", max_tokens=200)
print(response)
```

### Streaming Responses

```python
def stream_response(provider, prompt, api_key):
    if provider == "openai":
        import openai
        openai.api_key = api_key
        
        stream = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    elif provider == "ollama":
        import requests
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt, "stream": True},
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]

# Usage
for chunk in stream_response("openai", "Tell me a story", "your-key"):
    print(chunk, end="", flush=True)
```

---

## Customization and Fine-tuning

### 1. Prompt Engineering

```python
class PromptEngineer:
    def __init__(self):
        self.system_prompts = {
            "assistant": "You are a helpful AI assistant.",
            "coder": "You are an expert Python programmer.",
            "writer": "You are a creative writer.",
            "analyst": "You are a data analyst."
        }
    
    def create_prompt(self, role, user_input, context=None):
        system = self.system_prompts.get(role, self.system_prompts["assistant"])
        
        prompt = f"{system}\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += f"User: {user_input}\n\nAssistant:"
        
        return prompt

# Usage
engineer = PromptEngineer()
prompt = engineer.create_prompt("coder", "Write a function to sort a list")
response = client.generate(prompt)
```

### 2. Fine-tuning with Few-Shot Learning

```python
def create_few_shot_prompt(examples, query):
    prompt = "Examples:\n\n"
    
    for example in examples:
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"
    
    prompt += f"Input: {query}\nOutput:"
    
    return prompt

# Examples
examples = [
    {"input": "positive", "output": "This is a positive sentiment."},
    {"input": "negative", "output": "This is a negative sentiment."},
    {"input": "neutral", "output": "This is a neutral sentiment."}
]

query = "happy"
prompt = create_few_shot_prompt(examples, query)
response = client.generate(prompt)
```

### 3. Custom Model Wrapper

```python
class CustomAIModel:
    def __init__(self, provider, api_key, custom_config=None):
        self.provider = provider
        self.api_key = api_key
        self.config = custom_config or {}
        self.conversation_history = []
    
    def chat(self, message, role="user"):
        # Add to history
        self.conversation_history.append({"role": role, "content": message})
        
        # Generate response
        response = self._generate_with_context()
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _generate_with_context(self):
        # Use conversation history for context
        if self.provider == "openai":
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=self.config.get("model", "gpt-3.5-turbo"),
                messages=self.conversation_history,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1000)
            )
            return response.choices[0].message.content
    
    def reset_conversation(self):
        self.conversation_history = []
    
    def set_system_message(self, message):
        self.conversation_history = [{"role": "system", "content": message}]

# Usage
model = CustomAIModel("openai", "your-key", {"temperature": 0.9})
model.set_system_message("You are a helpful coding assistant.")
response = model.chat("How do I sort a list in Python?")
print(response)
```

---

## Testing AI Models

### 1. Response Quality Testing

```python
def test_ai_response(client, test_cases):
    results = []
    
    for test_case in test_cases:
        prompt = test_case["prompt"]
        expected_keywords = test_case.get("expected_keywords", [])
        
        response = client.generate(prompt)
        
        # Check if response contains expected keywords
        contains_keywords = all(
            keyword.lower() in response.lower() 
            for keyword in expected_keywords
        )
        
        results.append({
            "prompt": prompt,
            "response": response,
            "passed": contains_keywords,
            "expected_keywords": expected_keywords
        })
    
    return results

# Test cases
test_cases = [
    {
        "prompt": "What is machine learning?",
        "expected_keywords": ["algorithm", "data", "learn"]
    },
    {
        "prompt": "Explain neural networks",
        "expected_keywords": ["neurons", "layers", "weights"]
    }
]

results = test_ai_response(client, test_cases)
for result in results:
    print(f"Test: {result['prompt']}")
    print(f"Passed: {result['passed']}")
    print(f"Response: {result['response'][:100]}...")
    print()
```

### 2. Performance Testing

```python
import time
import statistics

def benchmark_ai_api(client, prompts, iterations=10):
    latencies = []
    responses = []
    
    for prompt in prompts:
        iteration_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            response = client.generate(prompt)
            end_time = time.time()
            
            latency = end_time - start_time
            iteration_times.append(latency)
            responses.append(response)
        
        latencies.append({
            "prompt": prompt,
            "mean_latency": statistics.mean(iteration_times),
            "std_latency": statistics.stdev(iteration_times),
            "min_latency": min(iteration_times),
            "max_latency": max(iteration_times)
        })
    
    return latencies, responses

# Benchmark
prompts = [
    "What is AI?",
    "Explain machine learning",
    "Describe neural networks"
]

latencies, responses = benchmark_ai_api(client, prompts, iterations=5)

for latency in latencies:
    print(f"Prompt: {latency['prompt']}")
    print(f"Mean latency: {latency['mean_latency']:.2f}s")
    print(f"Std dev: {latency['std_latency']:.2f}s")
    print()
```

### 3. A/B Testing Different Models

```python
def compare_models(prompts, models_config):
    results = {}
    
    for model_name, config in models_config.items():
        client = AIAPIClient(provider=config["provider"], api_key=config["api_key"])
        model_results = []
        
        for prompt in prompts:
            start_time = time.time()
            response = client.generate(prompt, **config.get("params", {}))
            latency = time.time() - start_time
            
            model_results.append({
                "prompt": prompt,
                "response": response,
                "latency": latency,
                "response_length": len(response)
            })
        
        results[model_name] = model_results
    
    return results

# Compare models
models_config = {
    "gpt-3.5": {
        "provider": "openai",
        "api_key": "key1",
        "params": {"max_tokens": 100}
    },
    "gemini": {
        "provider": "gemini",
        "api_key": "key2",
        "params": {}
    }
}

prompts = ["Explain AI", "What is ML?"]
comparison = compare_models(prompts, models_config)

for model_name, results in comparison.items():
    print(f"\n{model_name}:")
    avg_latency = sum(r["latency"] for r in results) / len(results)
    print(f"Average latency: {avg_latency:.2f}s")
```

---

## Complete Examples

### Example 1: Complete AI Chat Application

```python
class AIChatBot:
    def __init__(self, provider, api_key):
        self.client = AIAPIClient(provider, api_key)
        self.conversation_history = []
    
    def chat(self, user_message):
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Generate response
        response = self.client.generate(
            self._format_conversation(),
            max_tokens=500
        )
        
        # Add assistant response
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _format_conversation(self):
        # Format conversation history as prompt
        formatted = ""
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted
    
    def reset(self):
        self.conversation_history = []

# Usage
bot = AIChatBot("openai", "your-key")
response = bot.chat("Hello!")
print(response)
```

### Example 2: AI-Powered Code Generator

```python
class CodeGenerator:
    def __init__(self, api_key):
        self.client = AIAPIClient("openai", api_key)
        self.system_prompt = "You are an expert Python programmer. Generate clean, efficient code."
    
    def generate_function(self, description, language="python"):
        prompt = f"{self.system_prompt}\n\nGenerate a {language} function that: {description}"
        
        response = self.client.generate(prompt, max_tokens=500)
        
        # Extract code from response
        code = self._extract_code(response)
        
        return code
    
    def _extract_code(self, response):
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        return response

# Usage
generator = CodeGenerator("your-key")
code = generator.generate_function("sorts a list of numbers")
print(code)
```

---

## Key Takeaways

1. **MCP**: Standard protocol for AI model interaction
2. **Free APIs**: Many providers offer free tiers
3. **Unified Interface**: Create wrapper for multiple providers
4. **Customization**: Use prompt engineering and few-shot learning
5. **Testing**: Benchmark and compare different models
6. **Local Models**: Use Ollama for completely free local models

---

## Next Steps

- **[18_Complete_AI_QA_Guide.md](18_Complete_AI_QA_Guide.md)** - All AI-related Q&A
- **[16_Library_Comparison_Benchmarks.md](16_Library_Comparison_Benchmarks.md)** - Library comparisons

---

**Master AI API integration to build powerful AI applications!**

