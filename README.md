# 🔮 MEDEA-NEUMOUSA: The Talos Destroyer

> *"ἡ Μήδεια ἀνίστησι τὰς φωνὰς τῶν νεκρῶν"*  
> *"Medea resurrects the voices of the dead"*

**Where ancient voices rise and bronze giants fall**

## 🏛️ The Mythology

**MEDEA-NEUMOUSA** combines two powerful forces from Greek mythology:

- **Medea** (Μήδεια) - The sorceress who destroyed Talos, the bronze automaton guarding Crete
- **Neumousa** (Νεύμουσα) - "Of the Nine Muses" - The goddesses of arts, literature, and knowledge

Together, they represent the triumph of intelligent sorcery over mindless automation, making MEDEA-NEUMOUSA the perfect name for an AI that brings ancient languages back to life.

## ⚡ Powers of MEDEA-NEUMOUSA

### 🔮 Necromancer Module
Advanced translation between ancient and dead languages using cutting-edge LLM technology:
- **20+ Ancient Languages**: Latin, Ancient Greek, Sanskrit, Gothic, Hittite, Hebrew, Aramaic, and more
- **Scholarly Quality**: Morphological analysis, etymological connections, cultural context
- **Confidence Scoring**: Know how reliable each translation is
- **Batch Processing**: Resurrect multiple texts efficiently

### 🏛️ Classical Analysis
Deep linguistic analysis for classical texts:
- Morphological parsing and lemmatization
- Prosodic analysis and meter detection
- Named entity recognition for ancient texts
- Intertextuality mapping between authors

### 🕸️ Semantic Networks
Map relationships between ancient texts and concepts:
- Semantic similarity clustering
- Interactive network visualizations
- Cross-textual reference detection
- Thematic analysis across corpora

### 📜 Manuscript Processing
Digital humanities tools for ancient documents:
- OCR for damaged inscriptions
- Paleographic analysis
- Fragment reconstruction
- TEI-compliant XML generation

## 🌍 Supported Ancient Languages

MEDEA-NEUMOUSA commands the voices of **20+ ancient languages**:

### Classical Foundation
- **Latin** - Classical, Medieval, Renaissance
- **Ancient Greek** - Homeric, Attic, Koine, Byzantine

### Ancient Near Eastern
- **Hebrew** - Biblical and post-biblical
- **Aramaic** - Biblical and Jewish Aramaic
- **Akkadian** - Babylonian and Assyrian
- **Syriac** - Classical Syriac texts

### Ancient European
- **Gothic** - Wulfila's Bible
- **Old English** - Anglo-Saxon literature
- **Old Norse** - Eddas and sagas
- **Middle High German** - Medieval literature

### Indo-European & Anatolian
- **Sanskrit** - Vedic and Classical
- **Hittite** - Cuneiform archives
- **Avestan** - Zoroastrian texts
- **Old Persian** - Achaemenid inscriptions

### Late Antique
- **Coptic** - Christian Egyptian
- **Arabic** - Classical Arabic

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and enter the realm
git clone <repository-url> MEDEA-NEUMOUSA
cd MEDEA-NEUMOUSA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install the powers
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

**Required:** Set your `MEDEA_GEMINI_API_KEY` in the `.env` file.

### 3. Awaken MEDEA

```bash
# Start the sorceress
python run.py
```

Visit `http://localhost:8000` to see MEDEA-NEUMOUSA in action!

## 📖 API Usage

### Necromancer Translation

```python
import httpx

# Resurrect Latin as Ancient Greek
response = httpx.post("http://localhost:8000/api/v1/necromancer/resurrect", json={
    "text": "Arma virumque cano, Troiae qui primus ab oris",
    "source_language": "lat",
    "target_language": "grc"
})

result = response.json()
print(f"Original: {result['source_text']}")
print(f"Resurrected: {result['resurrected_text']}")
print(f"Confidence: {result['confidence']}")
```

### Batch Resurrection

```python
# Resurrect multiple texts at once
response = httpx.post("http://localhost:8000/api/v1/necromancer/mass_resurrect", json={
    "texts": [
        "Cogito ergo sum",
        "Veni, vidi, vici",
        "Carpe diem"
    ],
    "source_language": "lat",
    "target_language": "grc"
})
```

### Linguistic Analysis

```python
# Divine the connection between ancient words
response = httpx.post("http://localhost:8000/api/v1/necromancer/divine", json={
    "word1": "pater",
    "language1": "lat", 
    "word2": "πατήρ",
    "language2": "grc"
})
```

## 🛠️ API Endpoints

### Necromancer Module
- `POST /api/v1/necromancer/resurrect` - Single text translation
- `POST /api/v1/necromancer/mass_resurrect` - Batch translation
- `POST /api/v1/necromancer/divine` - Linguistic relationship analysis
- `GET /api/v1/necromancer/languages` - Supported languages

### Classical Analysis
- `POST /api/v1/classical/analyze` - Morphological analysis
- `POST /api/v1/classical/prosody` - Meter and prosody analysis
- `POST /api/v1/classical/entities` - Named entity recognition

### Semantic Networks
- `POST /api/v1/semantic/similarity` - Text similarity analysis
- `POST /api/v1/semantic/cluster` - Clustering analysis
- `POST /api/v1/networks/build` - Network generation

## 🧙‍♀️ The Technology Behind the Magic

MEDEA-NEUMOUSA combines cutting-edge AI with classical scholarship:

- **FastAPI** - Modern async Python web framework
- **Google Gemini 2.5 Flash** - Advanced language model for translation
- **Classical Language Toolkit (CLTK)** - Specialized NLP for ancient languages
- **Sentence Transformers** - Semantic embeddings
- **NetworkX** - Graph analysis for textual relationships
- **ChromaDB** - Vector database for semantic search

## 🎯 Why MEDEA-NEUMOUSA?

### For Classical Scholars
- **Scholarly Quality**: Real linguistic analysis, not just word substitution
- **Cultural Context**: Understanding of historical and cultural nuances
- **Multiple Approaches**: Lexical, morphological, and contextual analysis
- **Confidence Metrics**: Know when to trust the translation

### For Digital Humanists
- **Batch Processing**: Handle large corpora efficiently
- **API-First**: Integrate with existing digital humanities workflows
- **Semantic Analysis**: Discover hidden connections in ancient texts
- **Export Formats**: TEI, JSON, CSV for further analysis

### For Developers
- **Modern Architecture**: Clean, modular, testable code
- **Async Processing**: Handle multiple requests efficiently  
- **Comprehensive Documentation**: OpenAPI/Swagger docs
- **Docker Ready**: Easy deployment and scaling

## 🤝 Contributing

MEDEA-NEUMOUSA welcomes contributions from classical scholars, digital humanists, and developers:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** your magical improvements
4. **Test** your necromantic abilities
5. **Submit** a pull request

## 📜 License

MEDEA-NEUMOUSA is released under the MIT License. See `LICENSE` for details.

## 🙏 Acknowledgments

- **Classical Language Toolkit (CLTK)** - Foundation for ancient language processing
- **Google Gemini** - The LLM that powers our necromancy
- **Perseus Digital Library** - Inspiration for digital classics
- **All classical scholars** - Who preserve ancient voices for posterity

---

*"As Medea destroyed the bronze automaton Talos with her sorcery, so does MEDEA-NEUMOUSA bring ancient languages back from the dead, making the guardians of old texts obsolete through the power of intelligent magic."*

**🔮 Where ancient wisdom meets cutting-edge AI 🔮**