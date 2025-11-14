# TAYNEX: Bilingual Translation and Cultural Knowledge Retrieval System for Tay Language Preservation

## Executive Summary

This research presents **TAYNEX**, an integrated bilingual translation and cultural knowledge retrieval architecture for Tay↔Vietnamese translation. The system combines Neural Machine Translation (NMT) with Retrieval-Augmented Generation (RAG) to preserve and promote the cultural heritage and linguistic diversity of Vietnam's minority Tay ethnic group while ensuring content authenticity through verified knowledge sources.

---

## I. Project Motivation & Significance

### The Problem

Tay language—spoken by Vietnam's second-largest ethnic minority—is rapidly declining due to:
- **Limited digital resources**: fragmented, inconsistent data without standardization
- **Absence of reliable tools**: no bidirectional translation or knowledge query systems
- **Accessibility barriers**: cultural knowledge remains isolated from younger generations and researchers
- **Data scarcity**: no public, sentence-aligned bilingual corpus for modern NMT applications

### The Solution

TAYNEX addresses these challenges through:
1. A curated, normalized Tay–Vietnamese bilingual corpus (23,202 sentence pairs)
2. Symmetric translation models (Tay→Vietnamese and Vietnamese→Tay)
3. A knowledge retrieval layer with source attribution
4. An open-source web platform enabling community participation

---

## II. System Architecture

### 2.1 Overall Pipeline (Five-Stage Processing)

```
Input Text
    ↓
[Stage 1] Loanword/Entity Detection & Named Entity Recognition
    ↓
[Stage 2] Word Segmentation (PMI-based for Tay)
    ↓
[Stage 3] Lexical Mapping (Solr-indexed bilingual dictionary lookup)
    ↓
[Stage 4] Neural Machine Translation (tayvnam / viettay models)
    ↓
[Stage 5] Post-processing & Context-Aware Disambiguation
    ↓
Output Translation + Knowledge Attribution
```

### 2.2 Key Components

| Component | Function | Technology |
|-----------|----------|------------|
| **Loanword Detection** | Preserve stable borrowed terms and proper nouns | NER + rule-based filtering |
| **Word Segmentation** | Extract meaningful units from unsegmented Tay text | Pointwise Mutual Information (PMI) |
| **Lexical Mapping** | Direct dictionary lookup before NMT | Apache Solr (indexed dictionary) |
| **NMT Core** | Translate unmapped segments | mBART-50, BARTPho, NLLB-200 |
| **Post-processing** | Normalize output, resolve ambiguity, add citations | Context-based rules + RAG |
| **Knowledge Retrieval (RAG)** | Augment translations with verified cultural context | Vector embeddings + Gemini API |

---

## III. Dataset & Methodology

### 3.1 Corpus Construction

**Source Origins:**
- Academic textbooks and materials
- News archives and publications
- Tay–Vietnamese dictionaries (university-curated)
- Licensed publications

**Statistics:**
- **Initial records**: 23,340
- **After cleaning**: 23,202 (138 approximate duplicates removed)
- **No empty/exact duplicates**: Data integrity confirmed
- **Unicode distribution**: ~23,207 Latin characters; ~5,581 special characters

**Train/Validation/Test Split:**
- **Training**: 85% (19,722 pairs)
- **Validation**: 10% (2,320 pairs)
- **Test**: 5% (1,160 pairs)
- **No data leakage** across splits (verified via fuzzy-match)

### 3.2 Data Normalization Process

| Step | Operation | Preservation Rules |
|------|-----------|-------------------|
| **Unicode** | NFC normalization; remove hidden control characters | Maintain diacritics for tonal markers |
| **Punctuation** | Merge ellipsis; normalize brackets/quotes | Keep original spacing |
| **Named Entities** | No forced Vietnamization; fix obvious typos only | Preserve cultural/geographic names |
| **Deduplication** | Three-layer: empty, exact hash, fuzzy (similarity threshold) | Retain semantic variants |

### 3.3 Controlled Data Augmentation

Applied only to training data; validation/test remain clean:

| Augmentation | Strategy | Frequency |
|--------------|----------|-----------|
| **Delete** | Remove 1 token (max) from sentences ≥6 tokens | Prob ≤ 0.15 |
| **Swap** | Adjacent token/phrase reordering | Non-boundary, ≥8 tokens |
| **Noise** | Light punctuation/whitespace perturbation | Prob ≤ 0.15 |
| **Protection** | Dictionary of terms, locations, units excluded | All augmentations respect this list |

---

## IV. Model Architecture & Baselines

### 4.1 Symmetric Models

**tayvnam (Tay→Vietnamese):**
- Two-stage training: (i) Tay self-encoding (optional light denoising); (ii) Tay–Vietnamese fine-tuning
- Primary variant: BARTPho-syllable
- Alternative: mBART-50

**viettay (Vietnamese→Tay):**
- Direct fine-tuning on mBART-50 or BARTPho
- Output constraints: syllable/orthography preservation
- Proper noun protection list

### 4.2 Baseline Model Families

| Model Family | Representatives | Rationale |
|--------------|-----------------|-----------|
| **Multilingual** | mBART-50, mT5-base | Generalization across low-resource languages |
| **Vietnamese-Specialist** | BARTPho (word/syllable) | Pre-trained on 20GB Vietnamese; leverages linguistic similarity |
| **Low-Resource Optimized** | NLLB-200 distilled (600M) | Designed for minority languages |
| **Legacy/Reference** | MarianMT, M2M-100 | Comparison baselines |

### 4.3 Decoding Strategies

| Strategy | Characteristics | Use Case |
|----------|-----------------|----------|
| **Greedy** | Single best token per step | Baseline; fast but prone to suboptimality |
| **Beam Search** | Maintains k parallel hypotheses | Production default; best BLEU/METEOR scores |
| **Top-k Sampling** | Sample from k most likely tokens | Diverse outputs; requires temperature tuning |
| **Top-p (Nucleus)** | Sample from cumulative probability mass | Controlled diversity; risks repetition |

---

## V. Experimental Results

### 5.1 Vietnamese→Tay Translation

| Model | Augmentation | BLEU×100 | METEOR×10 | Decoding | Notes |
|-------|--------------|----------|-----------|----------|-------|
| **mBART-50** | ✓ | **43.90** | **3.22** | Beam | **BEST** |
| mBART-50 | ✓ | 41.80 | 3.14 | Greedy | |
| mBART-50 | ✓ | 42.70 | 3.20 | Top-k | |
| mBART-50 | ✓ | 42.40 | 3.18 | Top-p | |
| BARTPho-syllable | ✓ | 37.66 | 2.897 | Beam | |
| M2M-100 | ✓ | 39.60 | 2.850 | Beam | |
| mT5 | ✓ | 25.50 | 2.330 | Beam | |
| MarianMT | ✓ | 17.87 | 2.666 | Beam | |

**Key Finding:** mBART-50 + Beam Search optimal for Vi→Tay; +23.03 BLEU improvement via augmentation.

### 5.2 Tay→Vietnamese Translation

| Model | Augmentation | BLEU×100 | METEOR×10 | Decoding | Notes |
|-------|--------------|----------|-----------|----------|-------|
| **BARTPho-syllable** | ✓ | **38.90** | **2.769** | Beam | **BEST** |
| BARTPho-syllable | ✓ | 37.55 | 2.646 | Greedy | |
| BARTPho-syllable | ✓ | 33.21 | 2.344 | Top-k | |
| BARTPho-syllable | ✓ | 35.71 | 2.506 | Top-p | |
| mBART-50 | ✓ | 36.40 | 2.621 | Beam | |
| NLLB-200 600M | ✓ | 32.52 | 2.402 | Beam | |
| M2M-100 | ✓ | 33.90 | 2.388 | Beam | |
| mT5 | ✓ | 31.60 | 2.176 | Beam | |
| MarianMT | ✓ | 30.80 | 2.301 | Beam | |

**Key Finding:** BARTPho-syllable + Beam Search optimal for Tay→Vi; Vietnamese pre-training advantages evident (+8.78 BLEU via augmentation).

### 5.3 Impact of Data Augmentation

All models showed consistent improvement:
- **Average BLEU gain**: +4 to +8 points
- **Trend**: Augmentation prevents overfitting on small corpus (23K pairs)
- **Strategy effectiveness**: Noise + Delete + Swap > any single technique

### 5.4 Decoding Strategy Analysis

- **Beam Search** (k=5): Highest and most stable scores; recommended for production
- **Greedy**: ~1–2 BLEU points lower; fast baseline
- **Top-k/Top-p**: +5–10% diversity; trade-off with precision; suitable for interactive/exploratory use

---

## VI. LinguaViet Web Platform

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Client)                       │
│          HTML5/CSS3/JS – Responsive (Grid/Flexbox)         │
│        Fetch API calls → Backend (low bandwidth)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────┐      ┌──────────▼────────────┐
│ Translation API  │      │   Orchestration &     │
│   (Flask +       │      │   Chatbot API         │
│   PyTorch)       │      │   (Node.js/FastAPI)   │
│                  │      │                       │
│ • tayvnam model  │      │ • Request routing     │
│ • viettay model  │      │ • RAG pipeline        │
│ • Inference      │      │ • Gemini integration  │
│   (GPU Tesla T4) │      └───────────────────────┘
└──────────────────┘
```

### 6.2 Core Features

#### A. Bidirectional Translation

```
User Input (Tay or Vietnamese)
    ↓
POST /translate {text, direction, strategy}
    ↓
[Translation Pipeline] (pre-processing → NMT → post-processing)
    ↓
JSON Response {translation, confidence, source_entity_mapping}
    ↓
Display with optional source attribution
```

#### B. Cultural Knowledge Retrieval (RAG)

```
User Query: "What does 'hát Sluông' mean?"
    ↓
[Vector Embedding] (encode query)
    ↓
[Retrieval] (k-NN search in knowledge base index)
    ↓
[Augmentation] (context + query → prompt template)
    ↓
[Generation] (Gemini API with constraint: cite sources)
    ↓
Answer + Source Citations + Confidence Score
```

**Knowledge Base Contents:**
- Cultural practices & folklore
- Historical context
- Traditional terms & crafts
- Regional variants
- Ceremonial language

#### C. Interactive Features

- **Real-time translation** with multiple decoding strategies
- **Confidence scoring** via model uncertainty
- **Source attribution** for all cultural queries
- **Structured export** (JSON/CSV) for downstream editing
- **Community feedback loop** for continuous refinement

---

## VII. Key Contributions

### 7.1 Scientific Artifacts

1. **Tay–Vietnamese Bilingual Corpus**
   - First standardized, sentence-aligned, publicly-oriented dataset
   - 23,202 normalized pairs
   - Open methodology for future expansion
   - Enables reproducible NMT research

2. **TAYNEX Architecture**
   - Unified pipeline for low-resource bilingual translation
   - Integration of dictionary lookup + NMT + RAG
   - Symmetric design supporting both translation directions
   - Modular, extensible to other language pairs

3. **Pre-trained Models & Checkpoints**
   - Fine-tuned tayvnam (Tay→Vietnamese) baseline
   - Fine-tuned viettay (Vietnamese→Tay) baseline
   - Training scripts for reproducibility
   - Configuration files (hyperparameters, thresholds)

### 7.2 Practical Applications

| Domain | Use Case | Impact |
|--------|----------|--------|
| **Bilingual Education** | Curriculum aids, terminology sheets, reading comprehension | Reduces translation burden on educators |
| **Cultural Heritage Digitization** | Folk songs, oral histories, ceremonies, traditional texts | Archival preservation with source traceability |
| **Public Services** | Health advisories, agricultural guidance, administrative forms | Bridges language barrier in service delivery |
| **Community Participation** | Crowdsourced feedback loop via web platform | Continuous quality improvement |

### 7.3 Methodological Innovations

- **Controlled augmentation** without over-sampling or semantic corruption
- **PMI-based segmentation** for unsegmented minority languages
- **Multi-strategy evaluation** (Beam/Greedy/Top-k/Top-p) for realistic deployment scenarios
- **RAG integration** to mitigate hallucination in cultural content
- **Automated data quality pipeline** with fuzzy-match deduplication

---

## VIII. Limitations & Future Work

### 8.1 Current Limitations

| Limitation | Impact | Mitigation Strategy |
|------------|--------|-------------------|
| Corpus size (23K pairs) | May miss dialect variants | Expand via crowdsourcing + collaboration with cultural centers |
| Orthography inconsistency | Some historical texts use variants | Develop normalization rules; create variant lexicons |
| Automatic metrics (BLEU/METEOR) | Don't capture semantic naturalness fully | Conduct human evaluation rounds with native speakers |
| Computational constraints (Tesla T4) | Cannot train largest models (NLLB-3.3B, M2M-12B) | Explore distillation; apply quantization techniques |

### 8.2 Future Directions

1. **Speech-to-Text Integration**
   - Develop ASR for Tay dialect variants
   - Enable spoken ↔ written bidirectional translation
   - Accessibility for elderly community members

2. **Cross-Lingual Transfer Learning**
   - Leverage related minority languages (Nung, Thai, Zhuang)
   - Zero-shot translation to closely-related languages
   - Shared linguistic resources

3. **Human Evaluation Framework**
   - Structured assessments with native speakers
   - Adequacy, fluency, and terminology accuracy ratings
   - Regular evaluation campaigns for quality assurance

4. **Dataset Expansion**
   - Fieldwork collection from Tay communities
   - Ethnolinguistic documentation partnerships
   - Crawling and cleaning historical archives

5. **Specialized Sub-domains**
   - Domain adaptation for agriculture, medicine, law
   - In-domain terminology dictionaries
   - Genre-specific model variants

---

## IX. Conclusion

TAYNEX represents a comprehensive, practically-grounded approach to minority language preservation through technology. By combining:

- A rigorously curated bilingual corpus
- Symmetric, optimized NMT models (tayvnam / viettay)
- Knowledge retrieval with source attribution
- An open, participatory web platform

...the system addresses both **linguistic sustainability** and **content authenticity**—critical for minority languages facing rapid decline. The research demonstrates that transfer learning from Vietnamese-specialized models (BARTPho) combined with controlled data augmentation and strategic decoding can achieve meaningful quality (BLEU ~38–44) even with limited training data.

More broadly, TAYNEX serves as a **reproducible framework** for other Vietnamese minority languages (Nung, Bahnar, K'ho, etc.), with open-source artifacts enabling community-driven, long-term language preservation efforts. The integration of RAG explicitly addresses the "hallucination" problem in generative systems, making it suitable for sensitive cultural heritage applications.

---

## X. References

### Foundational Models & NMT

- Vaswani et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Bahdanau et al. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*.
- Lewis et al. (2020). BART: Denoising sequence-to-sequence pre-training. *ACL 2020*.
- Liu et al. (2020). Multilingual denoising pre-training for neural machine translation (mBART). *ACL 2020*.
- Tang et al. (2021). Multilingual translation from denoising pre-training. *ACL 2021*.
- Fan et al. (2021). Beyond English-Centric multilingual machine translation (M2M-100). *JMLR 2021*.
- Costa-jussà et al. (2022). No Language Left Behind: Scaling human-centered machine translation (NLLB-200). *Science* 377(6610).

### Vietnamese Language Models

- Nguyen et al. (2021). BARTPho: Pre-trained sequence-to-sequence models for Vietnamese.
- Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers (BERT). *NAACL 2019*.

### Evaluation & Low-Resource NMT

- Papineni et al. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL 2002*.
- Banerjee & Lavie (2005). METEOR: An automatic metric for MT evaluation. *ACL Workshop 2005*.
- Koehn & Knowles (2017). Six challenges for neural machine translation. *NMT Workshop 2017*.

### Knowledge Retrieval & Mitigation

- Lewis et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP. *NeurIPS 2020*.
- Guerreiro et al. (2023). Hallucinations in neural machine translation: Sources, taxonomy, and prevention. *TACL 2023*.

### Infrastructure

- Potter & Grainger (2014). Solr in Action. Manning Publications.
- Smiley & Pugh (2021). Apache Solr Reference Guide (8.x).
- Vu et al. (2018). VnCoreNLP: A toolkit for Vietnamese NLP.

### Statistical Methods

- Church & Hanks (1990). Word association norms, mutual information, and lexicography. *Computational Linguistics* 16(1).
- Jurafsky & Martin (2023, draft 3rd ed.). Speech and Language Processing. Chapter on collocations/PMI.

---

## Appendix: Quick Start Guide

### Installation & Setup

```bash
# Clone repository
git clone https://github.com/flappychill/TayViet_AI_Translate_Research_Project.git
cd tay-viet-nmt-suite

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
# (Instructions in experiments/README.md)
```

### Running Experiments

```bash
# Train Tay→Vietnamese (BARTPho baseline)
python experiments/nmt/train_bartpho.py

# Train Vietnamese→Tay (mBART-50)
python experiments/nmt/train_mbart50.py

# Custom model configuration
python experiments/nmt/train_generic.py \
  --csv data/bitext/tay_vi.csv \
  --out runs/custom \
  --model facebook/mbart-large-50-many-to-many-mmt \
  --src_code en_XX --tgt_code vi_VN
```

### Web Platform

```bash
# Start translation API (GPU)
python api/app.py

# Start web server (CPU)
gunicorn -w 2 -b 0.0.0.0:8000 web.app:app

# Open browser
# http://localhost:8000
```

---

**Project Repository:** https://github.com/flappychill/TayViet_AI_Translate_Research_Project  
**Lead Researchers:** Lê Trường Minh Đăng
**Team member**: Thiều Quang Tâm
**Advisor:** Lab manager. Nguyễn Khánh Lợi and other mentors (HCMUT EE Machine Learning & IoT Lab - University of Technology, Vietnam National University - Ho Chi Minh)
