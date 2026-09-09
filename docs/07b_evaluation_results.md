# Phase 7B Evaluation & Qualitative Translation Analysis

This document presents the official evaluation results of the from-scratch English → Odia Translation Transformer evaluated on the held-out test split using the best-performing checkpoint from the 4-epoch training baseline.

---

## 1. Evaluation Summary & Test Metrics

Evaluated using [`evaluate.py`](file:///d:/GitHub/FWC-TrainingProj/Test-7/evaluate.py) on the held-out test dataset split (`outputs/test.parquet`):

| Evaluation Metric | Measured Value |
| :--- | :--- |
| **Model Checkpoint** | [`experiments/phase7b_baseline/checkpoints/best.pt`](file:///d:/GitHub/FWC-TrainingProj/Test-7/experiments/phase7b_baseline/checkpoints/best.pt) (Epoch 4, Step 487,512) |
| **Test Split Samples Evaluated** | **500** sentence pairs |
| **Test Cross-Entropy Loss** | **`4.9372`** (ignoring `<PAD>` tokens) |
| **Test Perplexity** | **`139.37`** |
| **Corpus BLEU Score (SacreBLEU)** | **`3.70`** |
| **BLEU n-gram Precision Breakdown** | `1-gram: 19.3%`, `2-gram: 5.5%`, `3-gram: 2.0%`, `4-gram: 0.9%` (Brevity Penalty = 1.000) |

*Results persisted in [`experiments/phase7b_baseline/metrics/evaluation.json`](file:///d:/GitHub/FWC-TrainingProj/Test-7/experiments/phase7b_baseline/metrics/evaluation.json).*

---

## 2. Five Qualitative Translation Samples

Below are 5 representative translation examples comparing English Source, Ground-Truth Odia Reference, and the from-scratch Transformer's greedy autoregressive prediction.

### Sample 1: Conversational / Short Simple Sentence
- **English (Source)**: `How are you?`
- **Odia (Reference)**: `ତୁମେ କେମିତି ଅଛ?`
- **Odia (Predicted)**: `କେମିତି?`
- **Linguistic Analysis**: The model captures the core interrogative token `କେମିତି` ("How"), demonstrating that the model successfully learned foundational conversational question mappings without pre-training.

---

### Sample 2: Short Declarative Sentence
- **English (Source)**: `There will be harmony in the family.`
- **Odia (Reference)**: `ପରିବାରରେ ମିଳାମିଶି ରହିବ।`
- **Odia (Predicted)**: `ପାରିବାରିକ ସୁଖ-ବୟ ରହିବ।`
- **Linguistic Analysis**: The model accurately produces the noun root `ପାରିବାରିକ` ("familial / family-related") and the future-tense verb `ରହିବ` ("will remain/be"). The translation is syntactically coherent and semantically close to the ground truth.

---

### Sample 3: Medium Formal / Policy Sentence
- **English (Source)**: `The government even ignored the recommendations of the Commission.`
- **Odia (Reference)**: `ସରକାର କମିଟିର ଏହି ପ୍ରସ୍ତାବକୁ ମଧ୍ୟ ଅଣଦେଖା କରି ଆସୁଛନ୍ତି।`
- **Odia (Predicted)**: `ସେହିପରି ସରକାର ମଧ୍ୟ ଏହି ପ୍ରସ୍ତାବ ଉପରେ ବିଚାର କରୁଛନ୍ତି।`
- **Linguistic Analysis**: The model accurately identifies subject `ସରକାର` ("government"), target object `ଏହି ପ୍ରସ୍ତାବ` ("this recommendation/proposal"), and adverbial marker `ମଧ୍ୟ` ("also/even"). However, it inverts the polarity of "ignored" to "are considering" (`ବିଚାର କରୁଛନ୍ତି`), reflecting a common failure mode in small NMT models on negation/contrastive verbs.

---

### Sample 4: Complex Domain Sentence (Tourism / Administration)
- **English (Source)**: `The project would be monitored by the Odisha Tourism Development Corporation (OTDC).`
- **Odia (Reference)**: `ଓଡ଼ିଶା ପର୍ଯ୍ୟଟନ ଉନ୍ନୟନ ନିଗମ (ଓଟିଡିସି) ଏବଂ ଟିଡିସି ଏହାର ପରିଚାଳନା ଦାୟିତ୍ୱରେ ରହିବ।`
- **Odia (Predicted)**: `ଓଡ଼ିଶା ପର୍ଯ୍ୟଟନ କ୍ଷେତ୍ରରେ ଓଡ଼ିଶା ପର୍ଯ୍ୟଟନ କ୍ଷେତ୍ରରେ ସହଯୋଗ କରିବ।`
- **Linguistic Analysis**: The model correctly recognizes domain terms `ଓଡ଼ିଶା ପର୍ଯ୍ୟଟନ` ("Odisha Tourism"). However, it suffers from phrase repetition (`ଓଡ଼ିଶା ପର୍ଯ୍ୟଟନ କ୍ଷେତ୍ରରେ ... କ୍ଷେତ୍ରରେ`), a known pathology in greedy decoding when attention probabilities loop over high-frequency phrase embeddings.

---

### Sample 5: Long Complex Sentence (Mandatory Limitation Analysis)
- **English (Source)**: `Sourav is currently president of the Cricket Association of Bengal(CAB) and president of the editorial board with Wisden India.`
- **Odia (Reference)**: `ସୌରଭ ବର୍ତ୍ତମାନ କ୍ରିକେଟ ଆସୋସିଆସନ ଅଫ ବେଙ୍ଗଲର ଅଧ୍ୟକ୍ଷ ଏବଂ ୱିସ୍ଡେନ ଇଣ୍ଡିଆର ସମ୍ପାଦକ ମଣ୍ଡଳୀର ଅଧ୍ୟକ୍ଷ ଅଛନ୍ତି ।`
- **Odia (Predicted)**: `ଏବେ ଭାରତୀୟ କ୍ରିକେଟ ଦଳର ଅଧ୍ୟକ୍ଷ ତଥା ପୂର୍ବତନ ଅଧିନାୟକ ତଥା ଭାରତୀୟ କ୍ରିକେଟ ଦଳର ଅଧ୍ୟକ୍ଷ ଭାବେ ଦାୟିତ୍ୱ ଗ୍ରହଣ କରିଛନ୍ତି।`

#### Detailed Honest Limitation Discussion:
1. **Domain Association vs. Exact Alignment**:
   - The model accurately captures the broader contextual domain (cricket administration: `କ୍ରିକେଟ`, leadership title: `ଅଧ୍ୟକ୍ଷ`, temporal marker: `ଏବେ` for "currently", and formal verbal structure: `ଦାୟିତ୍ୱ ଗ୍ରହଣ କରିଛନ୍ତି` for "holding responsibility").
2. **Entity Hallucination & Capacity Constraints**:
   - Because the model is intentionally constrained to a tiny architecture ($d_{\text{model}}=128$, $N=2$ layers, 4 heads, 11.2M parameters) to fit class compute, its self-attention span cannot resolve multiple nested named entities ("Cricket Association of Bengal", "Wisden India"). Instead, it falls back on high-frequency co-occurring corpus phrases ("Indian Cricket Team president and former captain").
3. **Morphological Subword Agglutination**:
   - Odia suffix compounding (e.g., `-ର` genitive case, `-ଙ୍କୁ` dative case) requires deep encoder representations to align with English prepositional phrases ("of Bengal", "with Wisden").
4. **Greedy Decoding Bottleneck**:
   - Greedy decoding makes locally optimal token choices at each step without lookahead, leading to semantic drift on sequences exceeding 25+ tokens.

---

## 3. Training & Validation Curves

The model was trained for 4 complete epochs (487,512 gradient steps, ~7.38 GPU hours) on an NVIDIA RTX 2050 (4GB VRAM):

![Training Curves](assets/training_curves.png)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
| :--- | :---: | :---: | :---: | :---: |
| **Train Loss** | 6.5170 | 6.0899 | 5.9716 | **5.9210** |
| **Train Perplexity** | 676.57 | 441.38 | 392.12 | **372.79** |
| **Val Loss** | 6.0046 | 5.8221 | 5.7485 | **5.7005** |
| **Val Perplexity** | 405.29 | 337.68 | 313.71 | **299.03** |
| **Learning Rate** | `2.53e-4` | `1.79e-4` | `1.46e-4` | `1.27e-4` |
| **Peak VRAM Allocated** | 1,686 MB | 1,688 MB | 1,686 MB | 1,686 MB |

