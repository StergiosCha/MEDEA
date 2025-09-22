from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# URIEL imports with fallback
try:
    from urielplus import urielplus
    URIEL_AVAILABLE = True
    print("✓ URIEL+ (urielplus) available")
except ImportError:
    URIEL_AVAILABLE = False
    print("⚠ Warning: urielplus not available. Install with: pip install urielplus")

# --- REAL LINGUISTIC ANALYSIS IMPLEMENTATION ---

class SwadeshLexicalAnalyzer:
    """Real Swadesh list analysis"""
    
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.swadesh_files = {
            "Ancient Greek": "swadesh_ancient_greek.txt",
            "Modern Greek": "swadesh_modern_greek.txt",
            "Latin": "swadesh_latin.txt",
            "Italian": "swadesh_italian.txt",
            "Spanish": "swadesh_spanish.txt",
            "French": "swadesh_french.txt",
            "Romanian": "swadesh_romanian.txt",
            "Old Church Slavonic": "swadesh_old_church_slavonic.txt",
            "Bulgarian": "swadesh_bulgarian.txt",
            "Russian": "swadesh_russian.txt",
            "Serbian": "swadesh_serbian.txt",
            "Gothic": "swadesh_gothic.txt",
            "German": "swadesh_german.txt",
            "Sanskrit": "swadesh_sanskrit.txt",
            "Czech": "swadesh_czech.txt",
            "Hindi": "swadesh_hindi.txt"
        }

    def load_word_list(self, language: str) -> Optional[List[str]]:
        """Load actual Swadesh word list"""
        if language not in self.swadesh_files:
            return None
            
        filepath = os.path.join(self.base_path, self.swadesh_files[language])
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                words = []
                for line in f:
                    word = line.strip().split()[0] if line.strip() else ""
                    if word and ' ' not in word:  # Only single words
                        words.append(word)
                return words if words else None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def normalized_edit_distance(self, s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein distance"""
        if s1 == s2:
            return 0.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 1.0
        
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[len1][len2] / max(len1, len2)

    def calculate_distance(self, lang1: str, lang2: str) -> Optional[float]:
        """Calculate real lexical distance using Swadesh lists"""
        words1 = self.load_word_list(lang1)
        words2 = self.load_word_list(lang2)
        
        if not words1 or not words2:
            return None
        
        # Compare word-by-word for aligned concepts
        total_concepts = min(len(words1), len(words2))
        if total_concepts < 10:  # Need minimum data
            return None
        
        distances = []
        for i in range(total_concepts):
            if words1[i] and words2[i]:
                distance = self.normalized_edit_distance(words1[i], words2[i])
                distances.append(distance)
        
        return np.mean(distances) if distances else None


class UrielTypologicalAnalyzer:
    """Real URIEL+ typological analysis"""
    
    def __init__(self):
        if not URIEL_AVAILABLE:
            self.uriel = None
            return
            
        self.uriel = urielplus.URIELPlus()
        self.uriel_codes = {
            "Ancient Greek": "grc",
            "Modern Greek": "ell",
            "Latin": "lat",
            "Italian": "ita",
            "Spanish": "spa",
            "French": "fra",
            "Romanian": "ron",
            "Bulgarian": "bul",
            "Russian": "rus",
            "Serbian": "srp",
            "Gothic": "got",
            "German": "deu",
            "Sanskrit": "san",
            "Hindi": "hin",
            "Czech": "ces"
        }

    def calculate_distance(self, lang1: str, lang2: str) -> Optional[float]:
        """Calculate real URIEL+ distance"""
        if not URIEL_AVAILABLE or not self.uriel:
            return None
            
        if lang1 not in self.uriel_codes or lang2 not in self.uriel_codes:
            return None
        
        try:
            lang1_code = self.uriel_codes[lang1]
            lang2_code = self.uriel_codes[lang2]
            distance = self.uriel.new_distance("featural", [lang1_code, lang2_code])
            return distance
        except Exception:
            return None


class WalsTypologicalAnalyzer:
    """Real WALS typological analysis"""
    
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.wals_file = os.path.join(base_path, "sane_wals.xlsx")
        self.wals_mapping = {
            "Ancient Greek": "Classical Greek, κλασική Αττική",
            "Modern Greek": "Greek",
            "Latin": "Latin",
            "Italian": "Italian",
            "Spanish": "Spanish",
            "French": "French",
            "Romanian": "Romanian",
            "Old Church Slavonic": "Old Church Slavonic",
            "Bulgarian": "Bulgarian",
            "Russian": "Russian",
            "Serbian": "Serbo-Croatian",
            "Gothic": "Gothic",
            "German": "German",
            "Sanskrit": "Sanskrit",
            "Czech": "Czech",
            "Hindi": "Hindi"
        }
        self.language_features = self._load_wals_data()

    def _load_wals_data(self) -> Dict:
        """Load WALS data from Excel file"""
        if not os.path.exists(self.wals_file):
            print(f"WALS file not found: {self.wals_file}")
            return {}
            
        try:
            df = pd.read_excel(self.wals_file, sheet_name="WALS_full_table")
            df = df.dropna(subset=["Feature #"])
            language_columns = [col for col in df.columns if col not in ["Feature Name", "Feature #"]]
            
            language_features = {lang: {} for lang in language_columns}
            for _, row in df.iterrows():
                feature_id = row["Feature #"]
                for lang in language_columns:
                    value = row[lang]
                    if not pd.isna(value):
                        language_features[lang][feature_id] = value
            
            print(f"✓ WALS data loaded: {len(language_features)} languages")
            return language_features
        except Exception as e:
            print(f"Error loading WALS data: {e}")
            return {}

    def calculate_distance(self, lang1: str, lang2: str) -> Optional[float]:
        """Calculate real WALS distance"""
        if not self.language_features:
            return None
            
        wals1 = self.wals_mapping.get(lang1)
        wals2 = self.wals_mapping.get(lang2)
        
        if not wals1 or not wals2:
            return None
            
        if wals1 not in self.language_features or wals2 not in self.language_features:
            return None
        
        features1 = self.language_features[wals1]
        features2 = self.language_features[wals2]
        
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return None
        
        differences = sum(1 for f in common_features if features1[f] != features2[f])
        return differences / len(common_features)


class UDSyntacticAnalyzer:
    """Real UD treebank syntactic analysis"""
    
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.treebank_files = {
            "Ancient Greek": "grc_perseus-ud-train.conllu",
            "Modern Greek": "el_gdt-ud-train.conllu",
            "Latin": "la_ittb-ud-train.conllu",
            "Italian": "it_isdt-ud-train.conllu",
            "Spanish": "es_ancora-ud-train.conllu",
            "French": "fr_ftb-ud-train.conllu",
            "Romanian": "ro_rrt-ud-train.conllu",
            "Old Church Slavonic": "cu_proiel-ud-train.conllu",
            "Bulgarian": "bg_btb-ud-train.conllu",
            "Russian": "ru_syntagrus-ud-train.conllu",
            "Serbian": "sr_set-ud-train.conllu",
            "Czech": "cs_cac-ud-train.conllu",
            "Gothic": "got_proiel-ud-train.conllu",
            "German": "de_gsd-ud-train.conllu",
            "Sanskrit": "sa_ufal-ud-train.conllu",
            "Hindi": "hi_hdtb-ud-train.conllu"
        }

    def parse_conllu(self, filepath: str, max_sentences: int = 1000) -> Optional[List]:
        """Parse CoNLL-U file"""
        if not os.path.exists(filepath):
            print(f"Treebank file not found: {filepath}")
            return None
            
        try:
            sentences = []
            current_sentence = []
            sentence_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('#'):
                        continue
                    elif line == '':
                        if current_sentence:
                            sentences.append(current_sentence)
                            current_sentence = []
                            sentence_count += 1
                            if sentence_count >= max_sentences:
                                break
                    else:
                        parts = line.split('\t')
                        if len(parts) >= 8 and not '-' in parts[0] and not '.' in parts[0]:
                            token_data = {
                                'id': parts[0], 'form': parts[1], 'lemma': parts[2],
                                'upos': parts[3], 'xpos': parts[4], 'feats': parts[5],
                                'head': parts[6], 'deprel': parts[7]
                            }
                            current_sentence.append(token_data)
            
            if current_sentence and sentence_count < max_sentences:
                sentences.append(current_sentence)
            
            print(f"✓ Parsed {len(sentences)} sentences from {os.path.basename(filepath)}")
            return sentences if sentences else None
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None

    def extract_syntactic_features(self, sentences: List) -> Dict:
        """Extract syntactic features from UD sentences"""
        if not sentences:
            return {}
        
        pos_counts = Counter()
        deprel_counts = Counter()
        total_tokens = 0
        dependency_distances = []
        
        for sentence in sentences:
            for token in sentence:
                if not token['id'].isdigit():
                    continue
                    
                current_id = int(token['id'])
                head_id = int(token['head']) if token['head'].isdigit() and token['head'] != '0' else None
                
                pos_counts[token['upos']] += 1
                deprel_counts[token['deprel']] += 1
                total_tokens += 1
                
                if head_id:
                    distance = abs(current_id - head_id)
                    dependency_distances.append(distance)
        
        if total_tokens == 0:
            return {}
        
        return {
            'pos_frequencies': {pos: count/total_tokens for pos, count in pos_counts.items()},
            'deprel_frequencies': {rel: count/total_tokens for rel, count in deprel_counts.items()},
            'avg_dependency_distance': np.mean(dependency_distances) if dependency_distances else 0
        }

    def jensen_shannon_distance(self, p: Dict, q: Dict) -> float:
        """Calculate Jensen-Shannon distance between distributions"""
        all_keys = set(p.keys()) | set(q.keys())
        if not all_keys:
            return 1.0
        
        p_arr = np.array([p.get(k, 0) for k in all_keys])
        q_arr = np.array([q.get(k, 0) for k in all_keys])
        
        # Normalize
        p_arr = p_arr / np.sum(p_arr) if np.sum(p_arr) > 0 else p_arr
        q_arr = q_arr / np.sum(q_arr) if np.sum(q_arr) > 0 else q_arr
        
        # Add epsilon to avoid log(0)
        p_arr = np.where(p_arr == 0, 1e-10, p_arr)
        q_arr = np.where(q_arr == 0, 1e-10, q_arr)
        
        # Jensen-Shannon divergence
        m = (p_arr + q_arr) / 2
        js_div = 0.5 * np.sum(p_arr * np.log(p_arr / m)) + 0.5 * np.sum(q_arr * np.log(q_arr / m))
        
        return np.sqrt(js_div)

    def calculate_distance(self, lang1: str, lang2: str) -> Optional[float]:
        """Calculate real syntactic distance"""
        if lang1 not in self.treebank_files or lang2 not in self.treebank_files:
            return None
        
        filepath1 = os.path.join(self.base_path, self.treebank_files[lang1])
        filepath2 = os.path.join(self.base_path, self.treebank_files[lang2])
        
        sentences1 = self.parse_conllu(filepath1)
        sentences2 = self.parse_conllu(filepath2)
        
        if not sentences1 or not sentences2:
            return None
        
        features1 = self.extract_syntactic_features(sentences1)
        features2 = self.extract_syntactic_features(sentences2)
        
        if not features1 or not features2:
            return None
        
        # Combine POS and dependency relation distances
        pos_distance = self.jensen_shannon_distance(
            features1.get('pos_frequencies', {}), 
            features2.get('pos_frequencies', {})
        )
        deprel_distance = self.jensen_shannon_distance(
            features1.get('deprel_frequencies', {}), 
            features2.get('deprel_frequencies', {})
        )
        
        return (pos_distance + deprel_distance) / 2


class ASJPPhonologicalAnalyzer:
    """Real ASJP phonological analysis"""
    
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.asjp_files = {
            "Ancient Greek": "a_greek_asjp.txt",
            "Modern Greek": "m_greek_asjp.txt",
            "Latin": "latin_asjp.txt",
            "Italian": "italian_asjp.txt",
            "Spanish": "spanish_asjp.txt",
            "French": "french_asjp.txt",
            "Romanian": "romanian_asjp.txt",
            "Old Church Slavonic": "old_church_slavonic_asjp.txt",
            "Bulgarian": "bulgarian_asjp.txt",
            "Russian": "russian_asjp.txt",
            "Serbian": "serbian_asjp.txt",
            "Gothic": "gothic_asjp.txt",
            "German": "german_asjp.txt",
            "Sanskrit": "sanskrit_asjp.txt",
            "Czech": "czech_asjp.txt",
            "Hindi": "hindi_asjp.txt"
        }

    def load_asjp_data(self, language: str) -> Optional[Dict]:
        """Load ASJP phonetic data"""
        if language not in self.asjp_files:
            return None
            
        filepath = os.path.join(self.base_path, self.asjp_files[language])
        if not os.path.exists(filepath):
            return None
        
        try:
            concept_words = {}
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle format: "1 I	j3 //"
                    if '//' in line:
                        # Split by // and take everything before it
                        before_comment = line.split('//')[0].strip()
                        parts = before_comment.split()
                        
                        if len(parts) >= 3 and parts[0].isdigit():
                            concept_num = int(parts[0])
                            # The phonetic transcription is the last part before //
                            phonetic = parts[-1].strip()
                            
                            # Handle multiple variants separated by commas
                            if ',' in phonetic:
                                phonetic = phonetic.split(',')[0].strip()
                            
                            if phonetic:
                                concept_words[concept_num] = phonetic
                    else:
                        # Fallback for lines without //
                        parts = line.split()
                        if len(parts) >= 3 and parts[0].isdigit():
                            concept_num = int(parts[0])
                            phonetic = parts[2].strip()  # Third column is phonetic
                            
                            if ',' in phonetic:
                                phonetic = phonetic.split(',')[0].strip()
                            
                            if phonetic:
                                concept_words[concept_num] = phonetic
            
            print(f"✓ ASJP data loaded for {language}: {len(concept_words)} concepts")
            return concept_words if concept_words else None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def normalized_edit_distance(self, s1: str, s2: str) -> float:
        """Calculate normalized edit distance for phonetic strings"""
        if s1 == s2:
            return 0.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 1.0
        
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[len1][len2] / max(len1, len2)

    def calculate_distance(self, lang1: str, lang2: str) -> Optional[float]:
        """Calculate real phonological distance"""
        data1 = self.load_asjp_data(lang1)
        data2 = self.load_asjp_data(lang2)
        
        if not data1 or not data2:
            return None
        
        common_concepts = set(data1.keys()) & set(data2.keys())
        if len(common_concepts) < 10:  # Need minimum data
            return None
        
        distances = []
        for concept_num in common_concepts:
            word1 = data1[concept_num]
            word2 = data2[concept_num]
            if word1 and word2:
                distance = self.normalized_edit_distance(word1, word2)
                distances.append(distance)
        
        return np.mean(distances) if distances else None


class RealLinguisticDistanceAnalyzer:
    """Main analyzer using real linguistic data"""
    
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        print(f"Initializing analyzers with base path: {base_path}")
        
        self.swadesh_analyzer = SwadeshLexicalAnalyzer(base_path)
        self.uriel_analyzer = UrielTypologicalAnalyzer()
        self.wals_analyzer = WalsTypologicalAnalyzer(base_path)
        self.syntactic_analyzer = UDSyntacticAnalyzer(base_path)
        self.phonological_analyzer = ASJPPhonologicalAnalyzer(base_path)
        
        # Historical pairs with metadata
        self.historical_pairs = {
            "Ancient Greek → Modern Greek": ("Ancient Greek", "Modern Greek", 2700, "Hellenic"),
            "Latin → Italian": ("Latin", "Italian", 2000, "Italic"),
            "Latin → Spanish": ("Latin", "Spanish", 2000, "Italic"),
            "Latin → French": ("Latin", "French", 2000, "Italic"),
            "Latin → Romanian": ("Latin", "Romanian", 2000, "Italic"),
            "Old Church Slavonic → Bulgarian": ("Old Church Slavonic", "Bulgarian", 1100, "Slavic"),
            "Old Church Slavonic → Russian": ("Old Church Slavonic", "Russian", 1100, "Slavic"),
            "Old Church Slavonic → Serbian": ("Old Church Slavonic", "Serbian", 1100, "Slavic"),
            "Gothic → German": ("Gothic", "German", 1700, "Germanic"),
            "Old Church Slavonic → Czech": ("Old Church Slavonic", "Czech", 1100, "Slavic"),
            "Sanskrit → Hindi": ("Sanskrit", "Hindi", 3500, "Indo-Aryan")
        }

    def calculate_distances(self, lang1: str, lang2: str, dimensions: List[str]) -> Dict[str, Optional[float]]:
        """Calculate real distances for specified dimensions"""
        results = {}
        print(f"Calculating distances for {lang1} vs {lang2}, dimensions: {dimensions}")
        
        for dimension in dimensions:
            print(f"  Calculating {dimension}...")
            if dimension == "lexical":
                results[dimension] = self.swadesh_analyzer.calculate_distance(lang1, lang2)
            elif dimension == "typological":
                # Use WALS as primary typological source
                results[dimension] = self.wals_analyzer.calculate_distance(lang1, lang2)
            elif dimension == "uriel":
                results[dimension] = self.uriel_analyzer.calculate_distance(lang1, lang2)
            elif dimension == "syntactic":
                results[dimension] = self.syntactic_analyzer.calculate_distance(lang1, lang2)
            elif dimension == "phonological":
                results[dimension] = self.phonological_analyzer.calculate_distance(lang1, lang2)
            else:
                results[dimension] = None
            
            if results[dimension] is not None:
                print(f"    ✓ {dimension}: {results[dimension]:.4f}")
            else:
                print(f"    ✗ {dimension}: No data")
        
        return results


# Initialize the real analyzer
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Try different possible base paths
possible_paths = [
    os.path.join(current_dir, "app"),
    current_dir,
    os.path.join(current_dir, "data"),
    os.path.join(current_dir, "..")
]

base_path = ""
for path in possible_paths:
    test_files = ["sane_wals.xlsx", "swadesh_latin.txt", "grc_perseus-ud-train.conllu"]
    if any(os.path.exists(os.path.join(path, f)) for f in test_files):
        base_path = path
        print(f"Found data files in: {base_path}")
        break

try:
    analyzer = RealLinguisticDistanceAnalyzer(base_path)
    print(f"✓ Real linguistic distance analyzer initialized")
except Exception as e:
    analyzer = None
    print(f"Warning: Could not initialize real linguistic distance analyzer: {e}")

# --- FastAPI Router & Models ---

router = APIRouter()

class LanguageDistanceRequest(BaseModel):
    language1: str
    language2: str
    analysis_dimensions: List[str] = ["lexical", "syntactic", "typological", "phonological", "uriel"]

class LanguageDistanceResponse(BaseModel):
    language1: str
    language2: str
    overall_distance: Optional[float]
    component_distances: Dict[str, Optional[float]]
    confidence_score: float
    analysis_summary: str
    interpretation: str
    available_data: Dict[str, bool]
    family_info: Dict[str, str]

class LanguageInfo(BaseModel):
    name: str
    family: str
    treebank_file: str
    swadesh_file: Optional[str]
    asjp_file: Optional[str]
    available: bool

@router.get("/languages", response_model=List[LanguageInfo])
async def get_supported_languages():
    if not analyzer:
        raise HTTPException(status_code=503, detail="Linguistic analyzer not available")
    
    languages = []
    seen_languages = set()
    
    for pair_name, (ancient, modern, years, family) in analyzer.historical_pairs.items():
        for lang in [ancient, modern]:
            if lang not in seen_languages:
                seen_languages.add(lang)
                
                # Check file availability
                swadesh_available = analyzer.swadesh_analyzer.load_word_list(lang) is not None
                treebank_available = (lang in analyzer.syntactic_analyzer.treebank_files and 
                                    os.path.exists(os.path.join(analyzer.base_path, 
                                                               analyzer.syntactic_analyzer.treebank_files[lang])))
                asjp_available = analyzer.phonological_analyzer.load_asjp_data(lang) is not None
                
                languages.append(LanguageInfo(
                    name=lang,
                    family=family,
                    treebank_file=analyzer.syntactic_analyzer.treebank_files.get(lang, ""),
                    swadesh_file=analyzer.swadesh_analyzer.swadesh_files.get(lang),
                    asjp_file=analyzer.phonological_analyzer.asjp_files.get(lang),
                    available=swadesh_available or treebank_available or asjp_available
                ))
    
    return sorted(languages, key=lambda x: x.name)

@router.post("/calculate", response_model=LanguageDistanceResponse)
async def calculate_linguistic_distance(request: LanguageDistanceRequest):
    if not analyzer:
        raise HTTPException(status_code=503, detail="Linguistic analyzer not available")

    lang1, lang2 = request.language1, request.language2
    
    if lang1 == lang2:
        return LanguageDistanceResponse(
            language1=lang1,
            language2=lang2,
            overall_distance=0.0,
            component_distances={dim: 0.0 for dim in request.analysis_dimensions},
            confidence_score=1.0,
            analysis_summary="Identity comparison",
            interpretation="The languages are identical.",
            available_data={dim: True for dim in request.analysis_dimensions},
            family_info={"language1_family": "Same", "language2_family": "Same"}
        )

    try:
        # Calculate real distances
        distances = analyzer.calculate_distances(lang1, lang2, request.analysis_dimensions)
        
        # Determine data availability
        available_data = {dim: distances[dim] is not None for dim in request.analysis_dimensions}
        
        # Calculate overall distance from available dimensions
        valid_distances = [d for d in distances.values() if d is not None]
        
        if not valid_distances:
            raise HTTPException(
                status_code=404, 
                detail="No distance calculations available - check if data files exist for these languages."
            )

        overall_distance = np.mean(valid_distances)
        confidence_score = len(valid_distances) / len(request.analysis_dimensions)

        # Generate interpretation
        if overall_distance < 0.2:
            interpretation = f"{lang1} and {lang2} are very closely related languages."
        elif overall_distance < 0.4:
            interpretation = f"{lang1} and {lang2} show moderate linguistic distance."
        elif overall_distance < 0.6:
            interpretation = f"{lang1} and {lang2} are moderately distant languages."
        elif overall_distance < 0.8:
            interpretation = f"{lang1} and {lang2} are quite linguistically distant."
        else:
            interpretation = f"{lang1} and {lang2} are very linguistically distant."

        analysis_summary = f"Calculated {len(valid_distances)} out of {len(request.analysis_dimensions)} requested distance metrics using real linguistic data."

        # Get family info
        family1 = family2 = "Unknown"
        for _, (ancient, modern, _, family) in analyzer.historical_pairs.items():
            if lang1 in (ancient, modern):
                family1 = family
            if lang2 in (ancient, modern):
                family2 = family

        return LanguageDistanceResponse(
            language1=lang1,
            language2=lang2,
            overall_distance=overall_distance,
            component_distances=distances,
            confidence_score=round(confidence_score, 3),
            analysis_summary=analysis_summary,
            interpretation=interpretation,
            available_data=available_data,
            family_info={"language1_family": family1, "language2_family": family2}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating distances: {str(e)}")

@router.get("/pairs")
async def get_historical_pairs():
    if not analyzer:
        raise HTTPException(status_code=503, detail="Linguistic analyzer not available")
    
    pairs = []
    for pair_name, (ancient, modern, years, family) in analyzer.historical_pairs.items():
        pairs.append({
            "pair_name": pair_name,
            "ancient_language": ancient,
            "modern_language": modern,
            "years_separation": years,
            "language_family": family
        })
    return pairs

@router.get("/data-availability/{language}")
async def check_data_availability(language: str):
    if not analyzer:
        raise HTTPException(status_code=503, detail="Linguistic analyzer not available")
    
    # Check actual file availability
    swadesh_available = analyzer.swadesh_analyzer.load_word_list(language) is not None
    
    treebank_file = analyzer.syntactic_analyzer.treebank_files.get(language, "")
    treebank_available = treebank_file and os.path.exists(os.path.join(analyzer.base_path, treebank_file))
    
    asjp_available = analyzer.phonological_analyzer.load_asjp_data(language) is not None
    
    # For WALS and URIEL, we need to test with a dummy comparison since they need two languages
    try:
        wals_available = analyzer.wals_analyzer.calculate_distance(language, language) is not None
    except:
        wals_available = False
    
    try:
        uriel_available = analyzer.uriel_analyzer.calculate_distance(language, language) is not None
    except:
        uriel_available = False

    return {
        "language": language,
        "data_sources": {
            "treebank": {"available": treebank_available, "enables": ["syntactic"]},
            "swadesh": {"available": swadesh_available, "enables": ["lexical"]},
            "asjp": {"available": asjp_available, "enables": ["phonological"]},
            "wals": {"available": wals_available, "enables": ["typological"]},
            "uriel": {"available": uriel_available, "enables": ["uriel"]},
        },
    }