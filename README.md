# Mental Health RAG Chatbot.

A conversational chatbot for mental health support built with RAG 
(Retrieval-Augmented Generation) and NLP techniques.

The system understands the language you write in, detects how you are 
feeling, understands what you are asking, and answers using a knowledge 
base of real counseling conversations.

---

## How It Works

Every message the user sends passes through four modules in order:

1. Language Detection - identifies what language the message is written in
2. Emotion Classifier - detects the emotional state of the user
3. Intent Classifier - decides what the user wants
4. RAG Pipeline - retrieves relevant counseling knowledge and generates a response

If the user is just saying hello or goodbye, the system replies directly 
without going through the knowledge base. If the user is asking a mental 
health question, the full pipeline runs.

---

## Module 1 - Language Detection

Classifies the language of the user's message using a scikit-learn pipeline
(TF-IDF character/word features + LinearSVC), trained on the
[papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)
dataset.

Results (held-out test set, 10k samples):

- Test accuracy: **99.56%**

Supported labels: `ar`, `bg`, `de`, `el`, `en`, `es`, `fr`, `hi`, `it`, `ja`,
`nl`, `pl`, `pt`, `ru`, `sw`, `th`, `tr`, `ur`, `vi`, `zh`

How to use:

When you first import or use the language detection module, the model downloads automatically 
from Google Drive (~135MB). This takes a few minutes. You only need to download it once - 
it gets saved to `models/language_detector.joblib` and reuses that copy on future runs.

```python
from deployment.language_detection import detect_language

# First call downloads the model, then detects language
language = detect_language("Hello, how are you?")
print(language)  # Output: en
```

Or run the inference script directly:

```bash
python deployment/language_detection.py
```

Artifacts:

- Notebook: `notebooks/module1_language_detection.ipynb` (training code)
- Inference code: `deployment/language_detection.py`
- Outputs: `outputs/module1/` (confusion matrix, metrics)

---
## Module 2 — Emotion Classifier

Classifies the emotional state of the user's message using a fine-tuned DistilBERT model
trained on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

Supported labels: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`

Preprocessing:

Before training, the dataset was cleaned:
- Removed duplicates and empty texts
- Stripped HTML artifacts and URLs
- Normalized whitespace
- Filtered extreme-length outliers (< 3 or > 300 words)

| Split      | Rows after cleaning |
|------------|-------------------|
| Train      | 15,991            |
| Validation | 1,999             |
| Test       | 2,000             |

Training

A TF-IDF + Logistic Regression baseline was built first as a reference point, then DistilBERT
was fine-tuned with class-weighted loss to handle label imbalance
(joy: 5,359 examples vs. surprise: only 572).


Results (validation set, 2,000 samples):

| Model                  | Accuracy | Macro F1 |
|------------------------|----------|----------|
| TF-IDF + LogReg        | 88%      | 0.84     |
| DistilBERT (epoch 4)   | 94%      | 0.92     |

How to use:

When you first import or use the emotion detection module, it downloads the model files 
automatically from Google Drive. This takes a few minutes on first use. The model files 
get saved to `models/distilbert/` and reuse that copy on future runs.

```python
from deployment.emotion_detection import predict

# First call downloads the model, then predicts emotion
emotion = predict("I'm feeling really sad today")
print(emotion)  # Output: sadness
```

For batch predictions on multiple texts:

```python
from deployment.emotion_detection import predict_batch

emotions = predict_batch(["I'm happy", "I'm angry", "I'm scared"])
print(emotions)  # Output: ['joy', 'anger', 'fear']
```

Or run the inference script directly:

```bash
python deployment/emotion_detection.py
```

Artifacts:

- Notebook: `notebooks/module2_emotion_classifier.ipynb` (training code)
- Inference script: `deployment/emotion_detection.py`
- Results: `outputs/module2/` (classification reports, visualizations)
---
## Module 2 — Emotion Classifier

Classifies the emotional state of the user's message using a fine-tuned DistilBERT model
trained on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

Supported labels: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`

Preprocessing:

Before training, the dataset was cleaned:
- Removed duplicates and empty texts
- Stripped HTML artifacts and URLs
- Normalized whitespace
- Filtered extreme-length outliers (< 3 or > 300 words)

| Split      | Rows after cleaning |
|------------|-------------------|
| Train      | 15,991            |
| Validation | 1,999             |
| Test       | 2,000             |

Training

A TF-IDF + Logistic Regression baseline was built first as a reference point, then DistilBERT
was fine-tuned with class-weighted loss to handle label imbalance
(joy: 5,359 examples vs. surprise: only 572).


Results (validation set, 2,000 samples):

| Model                  | Accuracy | Macro F1 |
|------------------------|----------|----------|
| TF-IDF + LogReg        | 88%      | 0.84     |
| DistilBERT (epoch 4)   | 94%      | 0.92     |


Model Weights:

The fine-tuned model is too large to store in the repo. You have two options:

1. **Train from scratch** — run the training cell in the notebook (~2 hours on CPU)
2. **Download pre-trained weights** — run the download cell to pull the model from Google Drive
into `models/distilbert/`, then skip straight to the evaluation cell


Artifacts

- Notebook: `notebooks/module2_emotion_classifier.ipynb`
- Inference script: `deployment/emotion_detection.py`
- Metrics & classification reports: `outputs/module2/`

Quick check after downloading or training:

```bash
python deployment/emotion_detection.py
```
---

## Module 3 - Intent Classifier

Classifies what the user wants using LLM prompting via the Groq API. 
No training data or model weights required.

The module was evaluated using two approaches: zero-shot prompting 
(no examples given to the model) and few-shot prompting (examples 
given for each intent). Both were tested on a basic and hard set of 
cases covering mixed intent, edge cases, 
and adversarial inputs.

Results:

Basic test set:
- Zero-shot accuracy: 100%
- Few-shot accuracy: 100%

Hard test set:
- Zero-shot accuracy: 100%
- Few-shot accuracy: 82.35%

Zero-shot was selected as the final approach. On ambiguous inputs, 
few-shot examples caused the model to pattern-match to the nearest 
example rather than reason about the full message. Zero-shot had no 
such anchoring and handled all edge and adversarial cases correctly.

Model used: llama-3.3-70b-versatile via Groq API

Possible intents: `greeting`, `goodbye`, `gratitude`, `asking_mental_health_question`, `out_of_scope`

How to use:

The intent classifier uses the Groq API, so no model download is needed. You just need to add 
your Groq API key to the `.env` file.

```python
from deployment.intent_classifier import classify_intent

# Needs GROQ_API_KEY in .env
intent = classify_intent("I'm feeling depressed")
print(intent)  # Output: asking_mental_health_question
```

Artifacts:

- Notebook: `notebooks/module3_intent_classifier.ipynb` (train + evaluate)
- Inference script: `deployment/intent_classifier.py`

---

## Setup

1. Clone the repo and create a virtual environment

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
# Then edit .env and add: GROQ_API_KEY=your_key_here
```

4. The language and emotion models download automatically on first use - just import the modules 
and they will download in the background.

## Module 4 - RAG Pipeline

Answers mental health questions by retrieving relevant counseling conversations
from a vector database and generating a grounded, empathetic response using an LLM.

Dataset: [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)

---

### Dataset Overview

3,512 rows of real counseling conversations, each containing a user question
and a counselor response. The same question can appear multiple times with
different counselor responses, giving multiple perspectives on the same problem.

After cleaning:
- Removed 234 exact duplicate rows (same question and same response)
- Removed 36 empty or near-empty responses (spam, links, single words)
- Final: 3,476 rows, 991 unique questions

![Responses per Question](outputs/module4/responses_per_question.png)

The dataset covers family and relationship problems heavily. Topics like career
stress or eating disorders are barely represented, so the system will perform
better on the dominant topics.

![Topic Distribution](outputs/module4/topic_distribution.png)

Most questions have 1-4 counselor responses. Ten questions have 20+ responses
each (max 47). These will be handled with MMR filtering during indexing to avoid
storing near-identical perspectives.

### Pipeline

991 unique questions were embedded using sentence-transformers (all-MiniLM-L6-v2)
and indexed in Qdrant Cloud. Each document stores the question vector, consolidated
responses, and detected topics as metadata.

Before indexing, responses were filtered and consolidated:
- MMR filtering at 0.75 similarity removed redundant responses, bringing the average
  from 3.53 down to 2.31 responses per question
- For questions with 10+ responses, similar response groups were consolidated using
  an LLM, bringing the final average to 2.10 responses per question with a max of 12

### Retrieval and Generation

Every user message goes through query rewriting before hitting Qdrant. The rewriter
strips irrelevant personal details, expands vague emotional terms into descriptive
language, and extracts personal context like gender and relationship status. This
improves retrieval quality significantly on noisy or vague inputs.

The top 3 most relevant questions are retrieved from Qdrant. All their responses
are passed to the LLM along with the detected emotion and personal context to
generate a personalized, empathetic response.

Models used:
- Embeddings: all-MiniLM-L6-v2 via sentence-transformers
- Query rewriting: llama-3.3-70b-versatile via Groq
- Generation: openai/gpt-oss-120b via Groq

How to use:

Needs `GROQ_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` in `.env`.

```python
from deployment.rag_pipeline import rag_answer

result = rag_answer(
    user_message="I have been feeling very anxious and cannot sleep",
    emotion="fear"
)
print(result["answer"])
```

Artifacts:

- Notebook: `notebooks/module4_rag_pipeline.ipynb`
- Inference script: `deployment/rag_pipeline.py`
- Outputs: `outputs/module4/`
