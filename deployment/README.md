# FastAPI Deployment Backend



## Directory Structure

```text
deployment/
├── .env                        # Environment variables (API keys, LLM models, DB URLs)
├── main.py                     # FastAPI application entry point and initialization
├── rag_pipeline.py             # This is Essawy's RAG pipeline implementation
├── README.md                   # Documentation for the deployment directory (this file)
├── api/
│   └── routes.py               # API endpoints definitions (e.g., /chat routes)
├── core/
│   └── config.py               # Configuration management and loading from .env
├── schemas/
│   ├── chat.py                 # Pydantic models for data validation (Requests & Responses)
│   └── prompts.py              # Contains prompts as function for dynamic generation
└── services/
    ├── emotion_detection.py    # Service logic for analyzing user emotions
    ├── intent_classifier.py    # Service logic for classifying user intents
    ├── language_detection.py   # Service for identifying the language of the prompt
    └── rag_service.py          # Service acting as a bridge between the API and Qdrant DB/LLM

```
---

### .env file

This file contains all sensitive information such as API keys and database URL, and some configuration variables like the model names, and the number of results to retrieve from the database.

```
GROQ_API_KEY=your_key_here
QDRANT_API_KEY=your_key_here
QDRANT_URL=https://xxxx-xxxx-xxxx.region-0.aws.cloud.qdrant.io


# The following variables specify the models to be used for different tasks. You can change these to use different models as needed.

# The names should be formatted this way "<provider_name>/<model_name_within_provider>" to be compatible with the LiteLLM.

INTENT_LLM_MODEL="groq/openai/gpt-oss-120b"
GENERATION_LLM_MODEL="groq/openai/gpt-oss-120b"

    
RETRIEVAL_TOP_K=3      # Number of top results to retrieve from the Qdrant 
EMBED_MODEL="all-MiniLM-L6-v2" # The embedding model name
```

### main.py

This is simple fastapi application entry point, it loads the models and connect the router endpoints to the app.

### rag_pipeline.py

This file contains the RAG pipeline implmentation, which we should integrate into our backend. (Essawy's implmenetation).

### api directory

This directory contains the API endpoints definitions, it has four endpoints for now:

- POST "/detect-language": Detects the language of the input prompt.
- POST "/detect-emotion": Analyzes the emotion of the input prompt.
- POST "/classify-intent": Classifies the intent of the input prompt.
- POST "/chat": The main endpoint that takes the user prompt, detects language, emotion, and intent, retrieves relevant information from the Qdrant database, and generates a response using the LLM.

### core directory

This directory contains the configuration management logic, it loads the environment variables from the system environment variables.

Its purpose is to ensure the keys that should be present in the runtime.

```python
class Settings(BaseSettings):
INTENT_LLM_MODEL: str = "groq/llama-3.1-8b-instant"
GENERATION_LLM_MODEL: str = "groq/llama-3.3-70b-versatile"

QDRANT_URL: str = "http://localhost:6333"
QDRANT_API_KEY: str

RETRIEVAL_TOP_K: int = 3
EMBED_MODEL: str = "all-MiniLM-L6-v2"

class Config:
    extra = 'ignore'

settings = Settings()
```

It just contains some default values for the configuration variables, and it loads the values from the environment variables if they are present, otherwise it uses the default values.


### schemas directory

This directory contains the Pydantic models for data validation, it has two files:
- chat.py: Models for chat-related data structures (e.g., message formats, response schemas)
- prompts.py: Models for handling and generating prompts, it contains functions that generate dynamic prompts based on the input data.

The `chat.py` file defines the formats of the incoming requests and outgoing responses for the chat endpoint, ensuring that the data is structured correctly and validated before processing.


### services directory

This directory contains the service logic for different functionalities of the application, it has four files:
- emotion_detection.py: Contains the logic for analyzing the emotion of the user prompt.
- intent_classifier.py: Contains the logic for classifying the intent of the user prompt.
- language_detection.py: Contains the logic for identifying the language of the user prompt.
- rag_service.py: Contains the logic for the RAG pipeline, it acts as a bridge between the API endpoints and the Qdrant database/LLM. It handles the retrieval of relevant information from the database and the generation of responses using the LLM based on the retrieved information and the user prompt.

    > Keep in mind that we need to integrate Essawy's RAG pipeline implementation into the `rag_service.py` file, and we can modify it as needed to fit our specific use case and requirements.

---

### Notes:

- I used LiteLLM as unified LLM interface, we just change the model by simply modifying the model name n the .env file.

- We need to add the translation service as emotion classifier only works with English text.

- We need to maintain the coversation history in the future to provide better responses.

- We need to add caching mechanism to store the most frequent queries and their responses to improve the response time.

