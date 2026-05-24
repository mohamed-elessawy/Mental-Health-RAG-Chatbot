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
(TF-IDF character/word features + `LinearSVC`), trained on the
[papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)
dataset (70k train / 10k test, 20 languages).

### Approach

The notebook trains **four model variants** and picks the best by validation accuracy:

| Builder | Features | Notes |
|---------|----------|--------|
| `build_model_1` → `model_1_full` | char + char_wb + word n-grams | Highest capacity |
| `build_model_2` → `model_2_char_only` | char n-grams only | **Best on validation** — faster training |
| `build_model_3` → `model_3_word_charwb` | char_wb + word n-grams | Balanced |
| `build_model_4` → `model_4_compact` | compact char n-grams | Smallest / fastest |

Flow: stratified 90/10 train/validation split → EDA → compare all four on train/val/test →
learning curve for the winner → retrain on train+val → evaluate on test → save model.

### Results

Held-out test set (10k samples) — see `outputs/module1/test_metrics.csv`:

- **Best on validation:** `model_2_char_only` (selected before final retrain on train+val)
- Test accuracy: **99.56%** (after retraining the validation winner on train+val)

Supported labels: `ar`, `bg`, `de`, `el`, `en`, `es`, `fr`, `hi`, `it`, `ja`,
`nl`, `pl`, `pt`, `ru`, `sw`, `th`, `tr`, `ur`, `vi`, `zh`

### Notebook

`notebooks/module1_language_detection.ipynb` — run top to bottom.

| Section | What it does |
|---------|----------------|
| 1. Setup | Imports and paths (`outputs/module1/`, `models/`) |
| 2. Model definitions | `build_model_1` … `build_model_4` |
| 3. Training helpers | `fit_and_score`, learning curve, confusion matrix plot |
| 4. Load dataset | Hugging Face load + stratified validation split |
| 5. EDA | Language balance tables and distribution plots |
| 6. Compare models | Train/val/test accuracy table (saves `model_comparison.csv` when run) |
| 7. Learning curve | Test accuracy vs training size → `learning_curve.png` |
| 8. Final model | Test metrics, confusion matrix plot, `language_detector.joblib` |
| 9. Demo | Sample predictions on multilingual text |

### Outputs (`outputs/module1/`)

All artifacts use a **flat folder** (no nested `metrics/` or `figures/`).

**Committed in this branch**

| File | Description |
|------|-------------|
| `test_metrics.csv` | Best model (`model_2_char_only`), test accuracy, sample counts |
| `eda_overview.png` | EDA — bar chart (samples/lang), pie chart, length histogram, length boxplot |
| `learning_curve.png` | Train vs test accuracy vs training size (`model_2_char_only`) |

**Also written by the notebook** (re-run sections 5–8 to regenerate if needed):

| File | Section |
|------|---------|
| `test_metrics.json` | 8 — same summary as the CSV |
| `model_comparison.csv` | 6 — train / val / test accuracy per variant |
| `learning_curve.png` | Train vs test accuracy vs training size |
| `eda_overview.png` | Bar chart (samples/lang), pie chart, length histogram, length boxplot |
### Inference

`deployment/language_detection.py` — `load()` then `detect_language(text)`.

**Model weights:** Run the notebook to produce `models/language_detector.joblib`
locally (~135MB; too large for standard GitHub without Git LFS). Or download the
[pre-trained model](https://drive.google.com/drive/u/0/folders/1HwFTJHIGh-YJUSwL2ehmQyKZ9espjhtu)
and place it at `models/language_detector.joblib`.

```python
from deployment.language_detection import load, detect_language

load()
detect_language("How are you feeling today?")  # -> "en"
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

Possible intents: greeting, goodbye, gratitude, 
asking_mental_health_question, out_of_scope

Model used: llama-3.3-70b-versatile via Groq API

Artifacts:

- Notebook: `notebooks/module3_intent_classifier.ipynb` (train + evaluate)
- Inference script: `deployment/intent_classifier.py`

## Setup

1. Clone the repo and create a virtual environment

2. Install dependencies
pip install -r requirements.txt

3. Copy .env.example to .env and fill in your API keys