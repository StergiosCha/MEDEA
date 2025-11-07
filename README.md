# 🔮 MEDEA-NEUMOUSA: Neuro-Symbolic AI for Classical Studies

> *"ἡ Μήδεια ἀνίστησι τὰς φωνὰς τῶν νεκρῶν"*  
> *"Medea resurrects the voices of the dead"*

**Where ancient voices rise and logic meets learning**

## 🏛️ The Mythology

**MEDEA-NEUMOUSA** embodies two mythological forces:

- **Medea** (Μήδεια) - The sorceress who destroyed Talos, the bronze automaton, with her intelligence and magic
- **Neumousa** (Νεύμουσα) - "Of the Nine (NEU) Muses" - Goddesses of neural (NEU) knowledge and the arts

Together, they symbolize the triumph of intelligent reasoning over mindless computation—combining **neural** (LLM) and **symbolic** (logic) AI to resurrect ancient languages and unlock deep textual understanding.

## ⚡ Six Powers of MEDEA-NEUMOUSA

### 1. 🔮 Necromancer: Cross-Lingual Resurrection
Translate between 18 ancient and dead languages using multi-provider LLM technology:

**Two Modes:**
- **Simple Mode** (text >200 chars): Fast direct translation with timeout protection
- **Enhanced Mode** (text ≤200 chars): Genre detection, LLM-based morphological interpretation, semantic analysis, philological notes

**Quality Control:**
- English contamination detection (flags if >30% modern English)
- Anachronism detection (computer, internet, etc.)
- Language-specific validation (Greek diacritics, Norse runes, Sanskrit Devanagari)
- Confidence scoring by tier (0.95 for Ancient Greek/Latin/Sanskrit → 0.6 for Sumerian/Hittite)

**Supported Languages:** Ancient Greek, Latin, Sanskrit, Old English, Old Norse, Gothic, Old Church Slavonic, Classical Syriac, Biblical Hebrew, Biblical Aramaic, Middle High German, Avestan, Old Persian, Coptic, Classical Arabic, Sumerian, Akkadian, Hittite

### 2. 🕸️ Knowledge Graph Oracle
Extract structured knowledge from unstructured classical texts:

**Three Extraction Modes:**
- **BASIC**: Only explicitly stated entities and relations
- **RAG_ENHANCED**: + Implied relations and background knowledge
- **EXTERNAL_ENRICHED**: + Wikipedia enrichment with auto-transliteration

**Features:**
- Domain detection (historical, literary, scientific, political, general)
- Dynamic prompting based on domain
- Multi-strategy extraction (3 attempts with varied temperature to avoid LLM recitation)
- Text chunking for large documents (>15,000 chars) with sentence coherence
- RDF/Turtle output for semantic web compatibility
- Interactive network visualization with clustering

**Output:** Entities, relations, RDF graph, semantic network data, confidence metrics

### 3. 😢 Emotion Knowledge Graph
Extract emotional landscapes from classical texts:

**Supported Emotions:** anger, joy, sadness, fear, disgust, hatred, love, pride, shame, hope, despair, surprise, anxiety, grief, jealousy, gratitude, and more

**Dual Metrics:**
- **Intensity (0.0-1.0)**: How textually dominant the emotion is (determines node size)
- **Sentiment Score (-1.0 to +1.0)**: Positive/negative polarity (determines node color)

**Process:**
- LLM-based structured JSON extraction
- Pattern-based fallback detection
- Emotion-entity-relation triple extraction
- RDF/Turtle generation
- Visual emotion graph (nodes sized by intensity, colored by sentiment)

### 4. 🤖 Zeugma: Neuro-Symbolic Reasoning
The crown jewel: Combine neural extraction with symbolic Prolog inference for explainable, logical analysis.

**Why Neuro-Symbolic?**
- **Neural (LLM)**: Extracts facts from natural language, handles ambiguity
- **Symbolic (Prolog)**: Performs logical inference, transitive reasoning, guarantees consistency

**Workflow:**
1. **Extract** (neural): LLM extracts entities and relations → generates Prolog knowledge base
2. **Query** (symbolic): Run first-order logic queries with variables, quantifiers, negation

**Prolog Capabilities:**
- **Existential queries**: `invaded(X, attica)` → "Who invaded Attica?"
- **Universal quantification**: `forall(entity(X, _, person), commander_of(X, _))` → "Are all persons commanders?"
- **Negation**: `entity(X, _, person), \+ commander_of(X, _)` → "Persons who are NOT commanders"
- **Conjunction**: `invaded(X, Y), leader_of(Z, X)` → "Who invaded AND who led them?"
- **Disjunction**: `(son_of(X, hellen) ; originated_from(X, hellen))` → "Sons OR things that originated from Hellen"
- **Transitive chains**: `transitive_relation(has_authority_over, X, Y)`
- **Predicate variables**: `connected_by(achilles, mutilene, Pred)` → "HOW is Achilles connected to Mutilene?"
- **Aggregation**: `findall(X, invaded(_, X), Places)`, `count_type(person, N)`

**Inference Rules:**
- Logical: implication, transitivity, contrapositive, symmetry, reflexivity
- Domain-specific: command chains, conflict resolution, geographic reasoning, temporal logic
- Higher-order: `any_relation/3`, `connected_by/3`, `relation/3` (predicates as variables!)

### 5. 📊 Semantic Oracle
Deep semantic analysis powered by LLMs:

**Three Capabilities:**
- **Similarity Analysis**: Compare texts semantically with confidence scoring and detailed explanation
- **Clustering**: Group texts by semantic themes with automatic cluster naming
- **Textual Echoes**: Find intertextual parallels and allusions across corpora

**Features:**
- Multi-provider LLM support (Gemini, OpenAI, Anthropic)
- Structured JSON output
- Confidence scoring
- Thematic similarity detection

### 6. 📏 Linguistic Distance
Measure diachronic linguistic evolution across five dimensions:

**11 Pre-Configured Historical Pairs:**
- Ancient Greek → Modern Greek (2700 years)
- Latin → Italian/Spanish/French/Romanian (2000 years)
- Old Church Slavonic → Bulgarian/Russian/Serbian/Czech (1100 years)
- Gothic → German (1700 years)
- Sanskrit → Hindi (3500 years)

**Five Dimensions:**
1. **Lexical (Swadesh)**: Normalized Levenshtein distance on 100-word core vocabulary
2. **Phonological (ASJP)**: Edit distance on phonetic transcriptions
3. **Syntactic (Universal Dependencies)**: Jensen-Shannon divergence on POS/dependency distributions
4. **Typological (WALS)**: Feature-based typological distance
5. **URIEL+**: Advanced featural distance matrix

**Interpretation:**
- <0.2: Very closely related
- 0.2-0.4: Moderately related
- 0.4-0.6: Moderately distant
- 0.6-0.8: Quite distant
- \>0.8: Very distant

**Data Sources:** 16 CoNLL-U treebanks, 16 Swadesh lists, 16 ASJP phonetic files, WALS feature database

## 🌍 18 Ancient Languages by Confidence Tier

**Tier 0 (0.95 confidence):**
- Ancient Greek (grc) - Homeric, Attic, Koine
- Latin (lat) - Classical through Renaissance
- Sanskrit (san) - Vedic and Classical

**Tier 1 (0.85 confidence):**
- Old English (ang) - Anglo-Saxon literature
- Old Norse (non) - Eddas and sagas
- Gothic (got) - Wulfila's Bible

**Tier 2 (0.75 confidence):**
- Old Church Slavonic (chu) - Cyrillic ecclesiastical
- Classical Syriac (syc) - Christian literature
- Biblical Hebrew (hbo) - Torah and Tanakh
- Biblical Aramaic (arc) - Daniel and Ezra
- Middle High German (gmh) - Nibelungenlied
- Avestan (ave) - Zoroastrian texts
- Old Persian (peo) - Achaemenid inscriptions

**Tier 3 (0.6 confidence):**
- Coptic (cop) - Christian Egyptian
- Classical Arabic (arb) - Pre-modern Arabic
- Sumerian (sux) - Cuneiform tablets
- Akkadian (akk) - Babylonian/Assyrian
- Hittite (hit) - Anatolian cuneiform

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url> MEDEA-NEUMOUSA
cd MEDEA-NEUMOUSA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set up your LLM API keys (at least one required):

```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys
nano .env
```

**Supported LLM Providers:**
- `MEDEA_GEMINI_API_KEY` - Google Gemini (recommended: gemini-2.0-flash-exp)
- `MEDEA_OPENAI_API_KEY` - OpenAI (GPT-4, GPT-3.5-turbo)
- `MEDEA_ANTHROPIC_API_KEY` - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)

The app will automatically use available providers with fallback support.

### 3. Optional: Install SWI-Prolog (for Zeugma)

**macOS:**
```bash
brew install swi-prolog
```

**Ubuntu/Debian:**
```bash
sudo apt install swi-prolog
```

**Windows:** Download from https://www.swi-prolog.org/download/stable

Without Prolog, all features except Zeugma will work normally.

### 4. Launch

```bash
# Start the application
python run.py
```

Visit `http://localhost:8000` to access the web interface!

**API Documentation:** `http://localhost:8000/docs` (OpenAPI/Swagger)

## 📖 API Usage Examples

### 1. Necromancer Translation

```python
import requests

# Translate Latin to Ancient Greek
response = requests.post("http://localhost:8000/api/v1/linguistic_analysis/translate", json={
    "text": "Arma virumque cano, Troiae qui primus ab oris",
    "source_language": "lat",
    "target_language": "grc",
    "preferred_provider": "gemini"  # Optional: gemini, openai, or anthropic
})

result = response.json()
print(f"Translation: {result['translation']}")
print(f"Confidence: {result['confidence']}")
print(f"Mode: {result['mode']}")  # Simple or Enhanced
```

### 2. Knowledge Graph Extraction

```python
# Extract entities and relations from classical text
response = requests.post("http://localhost:8000/api/v1/kg/extract", json={
    "text": "Themistocles persuaded the Athenians to build walls at Piraeus.",
    "reasoning_mode": "RAG_ENHANCED",  # BASIC, RAG_ENHANCED, or EXTERNAL_ENRICHED
    "preferred_provider": "gemini"
})

kg = response.json()
print(f"Entities: {len(kg['entities'])}")
print(f"Relations: {len(kg['relations'])}")
print(f"RDF: {kg['rdf_representation']}")
```

### 3. Emotion Knowledge Graph

```python
# Extract emotional landscape from text
response = requests.post("http://localhost:8000/api/v1/emotion/extract", json={
    "text": "Achilles raged with fury as grief consumed his heart for Patroclus.",
    "preferred_provider": "anthropic"
})

emotions = response.json()
for emotion in emotions['emotions']:
    print(f"{emotion['emotion']}: intensity={emotion['intensity']}, sentiment={emotion['sentiment']}")
```

### 4. Zeugma Neuro-Symbolic Reasoning

```python
# Step 1: Extract knowledge graph and generate Prolog KB
extract_response = requests.post("http://localhost:8000/api/v1/zeugma/extract", json={
    "text": "Hellen was the son of Deucalion. Hellen gained power in Phthiotis.",
    "reasoning_mode": "RAG_ENHANCED",
    "preferred_provider": "gemini"
})

# Step 2: Query with first-order logic
query_response = requests.post("http://localhost:8000/api/v1/zeugma/query", json={
    "queries": [
        "son_of(X, deucalion)",  # Who is Deucalion's son?
        "son_of(X, Y), gained_power_in(X, Z)",  # Who is someone's son AND gained power somewhere?
        "findall(X, entity(X, _, person), People)"  # List all persons
    ]
})

for result in query_response.json()['results']:
    print(f"Query: {result['query']}")
    print(f"Solutions: {result['solutions']}")
```

### 5. Semantic Analysis

```python
# Find semantic similarity
response = requests.post("http://localhost:8000/api/v1/semantic/similarity", json={
    "text1": "Achilles was a great warrior",
    "text2": "Hector was a brave fighter",
    "preferred_provider": "openai"
})

print(f"Similarity: {response.json()['similarity_score']}")
print(f"Explanation: {response.json()['explanation']}")

# Find textual echoes (intertextuality)
echoes_response = requests.post("http://localhost:8000/api/v1/semantic/echoes", json={
    "source_text": "Sing, O goddess, the anger of Achilles",
    "target_texts": [
        "Tell me, O muse, of that ingenious hero who travelled far",
        "I sing of arms and the man"
    ],
    "preferred_provider": "gemini"
})
```

### 6. Linguistic Distance

```python
# Measure diachronic linguistic evolution
response = requests.post("http://localhost:8000/api/v1/linguistic_analysis/distance", json={
    "ancient_language": "lat",
    "modern_language": "ita",
    "dimensions": ["lexical", "phonological", "syntactic", "typological"]
})

distance = response.json()
print(f"Overall Distance: {distance['overall_distance']}")
print(f"Lexical: {distance['lexical_distance']}")
print(f"Phonological: {distance['phonological_distance']}")
```

## 🛠️ Complete API Reference

### Linguistic Analysis (Necromancer + Distance)
- `POST /api/v1/linguistic_analysis/translate` - Cross-lingual translation
- `POST /api/v1/linguistic_analysis/distance` - Linguistic distance measurement
- `GET /api/v1/linguistic_analysis/language-pairs` - Available historical pairs
- `GET /api/v1/linguistic_analysis/status` - Module status

### Knowledge Graph Oracle
- `POST /api/v1/kg/extract` - Extract entities and relations
- `POST /api/v1/kg/validate` - Validate RDF/Turtle output
- `GET /api/v1/kg/status` - Module status

### Emotion Knowledge Graph
- `POST /api/v1/emotion/extract` - Extract emotion graph
- `POST /api/v1/emotion/analyze` - Analyze emotional intensity
- `GET /api/v1/emotion/status` - Module status

### Zeugma Neuro-Symbolic
- `POST /api/v1/zeugma/extract` - Extract KG and generate Prolog KB
- `POST /api/v1/zeugma/query` - Query Prolog KB with first-order logic
- `POST /api/v1/zeugma/process` - Combined extraction + query
- `GET /api/v1/zeugma/status` - Module status (includes Prolog availability)

### Semantic Oracle
- `POST /api/v1/semantic/similarity` - Pairwise text similarity
- `POST /api/v1/semantic/cluster` - Multi-text clustering
- `POST /api/v1/semantic/echoes` - Find textual echoes/intertextuality
- `GET /api/v1/semantic/status` - Module status

**Full documentation:** `/docs` (Swagger UI) or `/redoc` (ReDoc)

## 🧙‍♀️ Technology Stack

MEDEA-NEUMOUSA combines cutting-edge AI with classical scholarship:

**Backend Framework:**
- **FastAPI** - Modern async Python web framework with automatic OpenAPI docs
- **Pydantic** - Data validation and settings management
- **Uvicorn** - Lightning-fast ASGI server

**LLM Providers (Multi-Provider Architecture):**
- **Google Gemini** - gemini-2.0-flash-exp (primary, recommended)
- **OpenAI** - GPT-4, GPT-3.5-turbo
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus
- Automatic fallback and provider selection

**Symbolic AI:**
- **SWI-Prolog** - First-order logic inference engine for Zeugma
- Custom Prolog KB generation from LLM-extracted facts
- Higher-order predicates for meta-reasoning

**Linguistic Data:**
- **Universal Dependencies** - 16 CoNLL-U treebanks for syntactic analysis
- **Swadesh Lists** - 16 languages for lexical distance
- **ASJP Database** - Phonetic transcriptions for phonological distance
- **WALS** - World Atlas of Language Structures for typological features
- **URIEL+** - Advanced typological feature vectors

**Text Processing:**
- **NetworkX** - Graph analysis for knowledge and semantic networks
- **RDFLib** - RDF/Turtle generation for semantic web
- **Levenshtein** - Edit distance calculations
- **SciPy** - Jensen-Shannon divergence for distributional analysis

**Frontend:**
- **Jinja2** - Server-side templating
- **Vanilla JavaScript** - No heavyweight frameworks
- **D3.js** / **Cytoscape.js** - Interactive graph visualizations

## 🎯 Why MEDEA-NEUMOUSA?

### For Classical Scholars
- **18 Ancient Languages**: Comprehensive coverage from Homer to Hammurabi
- **Confidence Tiers**: Know when to trust translations (0.95 for Greek/Latin → 0.6 for Sumerian)
- **Quality Validation**: Anachronism detection, contamination checks, language-specific rules
- **Emotion Analysis**: Map the emotional landscape of classical texts with intensity and sentiment
- **Diachronic Analysis**: Track linguistic evolution across 5 dimensions over millennia

### For Digital Humanists
- **Knowledge Graphs**: Extract structured entities and relations from unstructured texts
- **Semantic Analysis**: Find textual echoes, cluster by theme, measure similarity
- **Neuro-Symbolic Reasoning**: Combine LLM extraction with Prolog logical inference
- **RDF/Turtle Export**: Semantic web compatibility for linked open data
- **Multi-Modal Extraction**: BASIC, RAG-enhanced, and external-enriched modes

### For Computational Linguists
- **5-Dimensional Distance**: Lexical, phonological, syntactic, typological, URIEL+
- **11 Historical Pairs**: Pre-configured ancient→modern language evolution
- **Real Linguistic Data**: Universal Dependencies, Swadesh, ASJP, WALS
- **Explainable AI**: Prolog inference provides transparent reasoning chains
- **First-Order Logic**: Full quantification, negation, conjunction, disjunction

### For Developers
- **Multi-Provider LLM**: Choose Gemini, OpenAI, or Anthropic with automatic fallback
- **Clean Architecture**: Modular, async, type-hinted Python with Pydantic validation
- **RESTful API**: FastAPI with automatic OpenAPI/Swagger documentation
- **Docker Ready**: Containerized deployment with all dependencies (see `render-deployment` branch)
- **Error Handling**: Graceful degradation and comprehensive logging

## 🤝 Contributing

MEDEA-NEUMOUSA welcomes contributions from classical scholars, digital humanists, and developers:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-improvement`)
3. **Add** your enhancements (new languages, inference rules, linguistic data)
4. **Test** thoroughly (run existing test suite)
5. **Submit** a pull request with clear documentation

**Areas Needing Contributions:**
- Additional ancient languages (Etruscan, Phoenician, Sogdian, etc.)
- More historical language pairs for distance analysis
- Enhanced Prolog inference rules for domain-specific reasoning
- Improved emotion detection patterns
- Performance optimizations for large corpora

## 🚀 Deployment

**Local Development:**
```bash
python run.py
```

**Docker (with SWI-Prolog):**
```bash
docker build -t medea-neumousa .
docker run -p 8000:8000 --env-file .env medea-neumousa
```

**Cloud Deployment:**
See `DEPLOYMENT.md` and `RENDER_QUICKSTART.md` in the `render-deployment` branch for detailed instructions on deploying to Render, Railway, or other cloud platforms.

## 📜 License

MEDEA-NEUMOUSA is released under the **MIT License**. Free to use with attribution.

## 🙏 Acknowledgments

- **Universal Dependencies Project** - Treebanks for syntactic analysis
- **WALS** - World Atlas of Language Structures
- **Swadesh & ASJP** - Lexical and phonetic databases
- **SWI-Prolog Community** - Symbolic reasoning foundation
- **Google, OpenAI, Anthropic** - LLM API access
- **Perseus Digital Library** - Inspiration for digital classics
- **All classical scholars** - Who preserve ancient voices across millennia

---

## 📚 Citation

If you use MEDEA-NEUMOUSA in academic work, please cite:

```bibtex
@software{medea_neumousa_2025,
  author = {Chatzikyriakidis, Stergios},
  title = {MEDEA-NEUMOUSA: Neuro-Symbolic AI for Classical Studies},
  year = {2025},
  url = {https://github.com/yourusername/MEDEA-NEUMOUSA},
  note = {Combining neural language models with symbolic logic for ancient language analysis}
}
```

---

*"As Medea destroyed the bronze automaton Talos with her sorcery, so does MEDEA-NEUMOUSA bring ancient languages back from the dead—combining neural intelligence with symbolic logic to resurrect the voices of antiquity."*

**🔮 Where ancient voices rise and logic meets learning 🔮**

© 2025 Stergios Chatzikyriakidis. Where ancient wisdom meets cutting-edge AI.
