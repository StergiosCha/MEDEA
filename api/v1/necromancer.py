from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import google.generativeai as genai
import json
import re
import os
from dotenv import load_dotenv
from functools import lru_cache
import asyncio
from datetime import datetime
import hashlib
from collections import defaultdict

# --- Setup Router ---
router = APIRouter()

# --- Load Environment ---
load_dotenv()
GEMINI_API_KEY = os.getenv("MEDEA_GEMINI_API_KEY")

# --- Configure Gemini with fallback ---
MODEL_NAMES = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
]

models = []
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    for model_name in MODEL_NAMES:
        try:
            m = genai.GenerativeModel(model_name)
            models.append((model_name, m))
        except Exception as e:
            pass
else:
    models = []

# Primary model for backwards compatibility
model = models[0][1] if models else None

# --- Smart Mode Configuration ---
SIMPLE_MODE_THRESHOLD = 200  # Characters
MAX_TEXT_LENGTH = 2000       # Hard limit  
SIMPLE_MODE_TIMEOUT = 15     # Seconds
ENHANCED_MODE_TIMEOUT = 45   # Seconds

# --- Enhanced Language Capabilities ---
LANGUAGE_CAPABILITIES = {
    # Tier 1: High Confidence (0.8+)
    "lat": {
        "name": "Latin", 
        "confidence": 0.95, 
        "script": "Latin", 
        "tier": 1,
        "period": "Classical/Medieval",
        "family": "Indo-European"
    },
    "grc": {
        "name": "Ancient Greek", 
        "confidence": 0.9, 
        "script": "Greek", 
        "tier": 1,
        "period": "Classical",
        "family": "Indo-European"
    },
    "he": {
        "name": "Hebrew", 
        "confidence": 0.85, 
        "script": "Hebrew", 
        "tier": 1,
        "period": "Biblical/Classical",
        "family": "Semitic"
    },
    "enm": {
        "name": "Middle English", 
        "confidence": 0.85, 
        "script": "Latin", 
        "tier": 1,
        "period": "1100-1500",
        "family": "Indo-European"
    },
    
    # Tier 2: Medium Confidence (0.6-0.8)
    "sa": {
        "name": "Sanskrit", 
        "confidence": 0.75, 
        "script": "Devanagari", 
        "tier": 2,
        "period": "Vedic/Classical",
        "family": "Indo-European"
    },
    "ara": {
        "name": "Arabic", 
        "confidence": 0.8, 
        "script": "Arabic", 
        "tier": 2,
        "period": "Classical",
        "family": "Semitic"
    },
    "ang": {
        "name": "Old English", 
        "confidence": 0.7, 
        "script": "Latin", 
        "tier": 2,
        "period": "450-1100",
        "family": "Indo-European"
    },
    "non": {
        "name": "Old Norse", 
        "confidence": 0.65, 
        "script": "Latin", 
        "tier": 2,
        "period": "700-1300",
        "family": "Indo-European"
    },
    "frm": {
        "name": "Middle French", 
        "confidence": 0.7, 
        "script": "Latin", 
        "tier": 2,
        "period": "1300-1600",
        "family": "Indo-European"
    },
    "xno": {
        "name": "Anglo-Norman", 
        "confidence": 0.6, 
        "script": "Latin", 
        "tier": 2,
        "period": "1066-1300",
        "family": "Indo-European"
    },
    
    # Tier 3: Lower Confidence (0.4-0.6)
    "arc": {
        "name": "Aramaic", 
        "confidence": 0.6, 
        "script": "Aramaic", 
        "tier": 3,
        "period": "Classical",
        "family": "Semitic"
    },
    "gmh": {
        "name": "Middle High German", 
        "confidence": 0.6, 
        "script": "Latin", 
        "tier": 3,
        "period": "1050-1350",
        "family": "Indo-European"
    },
    "cop": {
        "name": "Coptic", 
        "confidence": 0.5, 
        "script": "Coptic", 
        "tier": 3,
        "period": "200-1700",
        "family": "Afroasiatic"
    },
    "syr": {
        "name": "Syriac", 
        "confidence": 0.45, 
        "script": "Syriac", 
        "tier": 3,
        "period": "Classical",
        "family": "Semitic"
    },
    "got": {
        "name": "Gothic", 
        "confidence": 0.4, 
        "script": "Gothic", 
        "tier": 3,
        "period": "300-700",
        "family": "Indo-European"
    },
    
    # Tier 4: Experimental (0.2-0.4)
    "peo": {
        "name": "Old Persian", 
        "confidence": 0.4, 
        "script": "Cuneiform", 
        "tier": 4,
        "period": "600-300 BCE",
        "family": "Indo-European"
    },
    "av": {
        "name": "Avestan", 
        "confidence": 0.35, 
        "script": "Avestan", 
        "tier": 4,
        "period": "Zoroastrian",
        "family": "Indo-European"
    },
    "akk": {
        "name": "Akkadian", 
        "confidence": 0.3, 
        "script": "Cuneiform", 
        "tier": 4,
        "period": "2500-100 BCE",
        "family": "Semitic"
    },
    "hit": {
        "name": "Hittite", 
        "confidence": 0.25, 
        "script": "Cuneiform", 
        "tier": 4,
        "period": "1650-1200 BCE",
        "family": "Indo-European"
    },
    "egy": {
        "name": "Middle Egyptian", 
        "confidence": 0.3, 
        "script": "Hieroglyphic", 
        "tier": 4,
        "period": "2055-1650 BCE",
        "family": "Afroasiatic"
    },
    "luv": {
        "name": "Luvian", 
        "confidence": 0.15, 
        "script": "Luvian hieroglyphs", 
        "tier": 4,
        "period": "1400-700 BCE",
        "family": "Indo-European"
    },
    "pal": {
        "name": "Palaic", 
        "confidence": 0.1, 
        "script": "Cuneiform", 
        "tier": 4,
        "period": "1650-1200 BCE",
        "family": "Indo-European"
    }
}

# --- Genre Detection Patterns ---
GENRE_PATTERNS = {
    "epic": {
        "keywords": ["hero", "battle", "glory", "rage", "wrath", "war", "sing", "muse", "achilles", "troy"],
        "patterns": [r"sing.*goddess", r"rage.*son", r"tell.*tale"]
    },
    "religious": {
        "keywords": ["god", "prayer", "sacred", "temple", "divine", "holy", "lord", "blessed", "faith"],
        "patterns": [r"hallowed.*name", r"thy.*kingdom", r"deliver.*evil"]
    },
    "legal": {
        "keywords": ["law", "decree", "judgment", "witness", "contract", "court", "justice", "penalty"],
        "patterns": [r"whereas.*party", r"hereby.*declared", r"witness.*whereof"]
    },
    "philosophical": {
        "keywords": ["truth", "wisdom", "knowledge", "being", "essence", "nature", "virtue", "soul"],
        "patterns": [r"what.*is", r"nature.*of", r"therefore.*conclude"]
    },
    "historical": {
        "keywords": ["year", "king", "battle", "empire", "reign", "conquest", "chronicle", "anno"],
        "patterns": [r"in.*year.*of", r"king.*ruled", r"came.*pass"]
    },
    "poetic": {
        "keywords": ["love", "beauty", "heart", "soul", "sweet", "fair", "verse", "rhyme", "meter"],
        "patterns": [r"o.*beloved", r"shall.*compare", r"when.*shall"]
    }
}

GENRE_INSTRUCTIONS = {
    "epic": "Preserve heroic diction, formulaic expressions, and epic conventions. Maintain elevated register.",
    "religious": "Preserve sacred terminology, ritual language, and reverent tone. Handle divine names carefully.",
    "legal": "Maintain precise legal terminology, formal structure, and ceremonial language.",
    "philosophical": "Capture abstract concepts, logical structure, and technical terminology accurately.",
    "historical": "Preserve chronological accuracy, proper names, and factual precision.",
    "poetic": "Consider meter, rhythm, literary devices, and aesthetic qualities.",
    "general": "Maintain appropriate register and cultural context."
}

# --- Analytics Storage ---
translation_stats = {
    "total_translations": 0,
    "language_pairs": defaultdict(int),
    "success_rates": defaultdict(list),
    "confidence_scores": defaultdict(list),
    "genre_distribution": defaultdict(int),
    "error_patterns": defaultdict(int),
    "mode_usage": defaultdict(int)
}

# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    genre_hint: Optional[str] = Field(None, description="Optional genre hint")
    include_analysis: bool = Field(True, description="Include philological analysis")
    include_alternatives: bool = Field(True, description="Include alternative translations")
    force_mode: Optional[str] = Field(None, description="Force 'simple' or 'enhanced' mode")

class BatchTranslationRequest(BaseModel):
    requests: List[TranslationRequest] = Field(..., max_items=10)

class PhilologicalAnalysisRequest(BaseModel):
    text: str
    language: str
    detailed: bool = Field(True, description="Include detailed morphological analysis")

# --- Utility Functions ---
def extract_json(text: str) -> str:
    """Extract JSON object from LLM output safely."""
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        return array_match.group(0)
    
    return text

def is_english_response(translation: str, target_lang: str) -> bool:
    """Enhanced detection of English in non-English translations"""
    if not translation or len(translation.strip()) < 3:
        return False
    
    translation = translation.lower().strip()
    
    common_english_words = {
        "the", "and", "of", "to", "a", "in", "that", "have", "i", "it", "for", 
        "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
        "by", "from", "they", "she", "or", "an", "will", "my", "one", "all", "would",
        "there", "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no", "just",
        "him", "know", "take", "people", "into", "year", "your", "good", "some",
        "could", "them", "see", "other", "than", "then", "now", "look", "only",
        "come", "its", "over", "think", "also", "back", "after", "use", "two",
        "how", "our", "work", "first", "well", "way", "even", "new", "want",
        "because", "any", "these", "give", "day", "most", "us", "wrath", "sing",
        "goddess", "son", "destructive", "rage", "declare", "sprung", "peleus",
        "translation", "means", "would", "be", "this", "is", "achilles"
    }
    
    words = re.findall(r'\b\w+\b', translation)
    if len(words) == 0:
        return False
    
    english_word_count = sum(1 for word in words if word.lower() in common_english_words)
    english_ratio = english_word_count / len(words)
    
    threshold = 0.4 if len(words) < 5 else 0.3
    
    if english_ratio > threshold:
        return True
    
    english_phrases = [
        "wrath sing", "sing goddess", "son of", "rage of", "declare o goddess",
        "this is", "the translation", "would be", "means", "in english",
        "peleus' son", "achilleus", "achilles", "translation error",
        "cannot translate", "unable to", "sorry i cannot"
    ]
    
    return any(phrase in translation for phrase in english_phrases)

def detect_anachronisms(translation: str) -> List[str]:
    """Detect anachronistic terms in translation"""
    modern_terms = [
        "computer", "internet", "smartphone", "television", "radio", "car", "plane",
        "democracy", "capitalism", "socialism", "nationalism", "globalization",
        "technology", "digital", "electronic", "nuclear", "atomic", "plastic"
    ]
    
    found_anachronisms = []
    translation_lower = translation.lower()
    
    for term in modern_terms:
        if term in translation_lower:
            found_anachronisms.append(term)
    
    return found_anachronisms

def detect_genre(text: str, source_lang: str = None) -> str:
    """Detect text genre based on keywords and patterns"""
    text_lower = text.lower()
    
    genre_scores = defaultdict(float)
    
    for genre, patterns in GENRE_PATTERNS.items():
        keyword_count = sum(1 for keyword in patterns["keywords"] if keyword in text_lower)
        genre_scores[genre] += keyword_count * 0.5
        
        pattern_count = sum(1 for pattern in patterns["patterns"] if re.search(pattern, text_lower))
        genre_scores[genre] += pattern_count * 1.0
    
    if not genre_scores:
        return "general"
    
    return max(genre_scores.items(), key=lambda x: x[1])[0]

@lru_cache(maxsize=100)
def get_language_info(lang_code: str) -> Dict[str, Any]:
    """Cached language information retrieval"""
    return LANGUAGE_CAPABILITIES.get(lang_code, {})

def validate_languages(source_lang: str, target_lang: str):
    """Validate language codes and compatibility"""
    if source_lang not in LANGUAGE_CAPABILITIES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported source language: {source_lang}. Supported: {list(LANGUAGE_CAPABILITIES.keys())}"
        )
    
    if target_lang not in LANGUAGE_CAPABILITIES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported target language: {target_lang}. Supported: {list(LANGUAGE_CAPABILITIES.keys())}"
        )
    
    if source_lang == target_lang:
        raise HTTPException(
            status_code=400, 
            detail="Source and target languages must be different"
        )

def should_use_simple_mode(text: str, force_mode: str = None) -> bool:
    """Determine whether to use simple or enhanced mode"""
    if force_mode == "simple":
        return True
    if force_mode == "enhanced":
        return False
    return len(text) > SIMPLE_MODE_THRESHOLD

# --- Simple Translation Mode ---
def craft_simple_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Create simple, fast translation prompt"""
    source_info = get_language_info(source_lang)
    target_info = get_language_info(target_lang)
    
    return f"""Translate this {source_info['name']} text to {target_info['name']}.

TEXT: "{text}"

Instructions:
- Provide accurate, natural translation
- Preserve meaning and style  
- Be concise but complete

Output ONLY valid JSON:
{{
    "translation": "your translation in {target_info['name']}",
    "confidence": 0.8,
    "notes": "brief notes if needed"
}}"""

async def simple_translation_mode(request: TranslationRequest) -> dict:
    """Fast, simple translation for longer texts"""
    if not model:
        raise HTTPException(status_code=500, detail="Translation service unavailable")
    
    try:
        prompt = craft_simple_prompt(request.text, request.source_language, request.target_language)
        
        response = await asyncio.wait_for(
            model.generate_content_async(
                [prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1024
                )
            ),
            timeout=SIMPLE_MODE_TIMEOUT
        )
        
        result_json = extract_json(response.text.strip())
        result = json.loads(result_json)
        
        return {
            "source_text": request.text,
            "resurrected_text": result.get("translation", ""),
            "source_language": {
                "code": request.source_language,
                "name": LANGUAGE_CAPABILITIES[request.source_language]["name"],
                "confidence_tier": LANGUAGE_CAPABILITIES[request.source_language]["tier"]
            },
            "target_language": {
                "code": request.target_language, 
                "name": LANGUAGE_CAPABILITIES[request.target_language]["name"],
                "confidence_tier": LANGUAGE_CAPABILITIES[request.target_language]["tier"]
            },
            "confidence_metrics": {
                "final_confidence": result.get("confidence", 0.7),
                "mode_confidence": 0.8
            },
            "translation_metadata": {
                "mode": "simple",
                "reasoning": "Used simple mode for fast, reliable translation",
                "notes": result.get("notes", ""),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, 
            detail="Simple translation timeout - the spirits are overwhelmed"
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Simple translation parsing failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Simple translation failed: {str(e)}"
        )

# --- Enhanced Translation Mode (Your existing functions) ---
def craft_enhanced_philological_prompt(text: str, source_lang: str, detailed: bool = True) -> str:
    """Create comprehensive philological analysis prompt"""
    lang_info = get_language_info(source_lang)
    source_name = lang_info.get("name", source_lang)
    period = lang_info.get("period", "Unknown")
    script = lang_info.get("script", "Unknown")
    family = lang_info.get("family", "Unknown")
    
    detail_level = "comprehensive" if detailed else "basic"
    
    return f"""You are a world-class philologist specializing in {source_name} from the {period} period.

ANALYSIS TARGET:
- Text: "{text}"
- Language: {source_name}
- Script: {script}
- Family: {family}
- Analysis Level: {detail_level}

OUTPUT MUST BE VALID JSON:
{{
    "textual_assessment": {{
        "text_quality": "assessment of text integrity",
        "potential_issues": ["list of textual problems if any"],
        "manuscript_notes": "relevant manuscript tradition information"
    }},
    "morphological_analysis": [
        {{
            "word": "original_word_form",
            "position": 1,
            "lemma": "dictionary_headword",
            "part_of_speech": "noun/verb/adjective/particle/etc",
            "literal_meaning": "direct_dictionary_meaning",
            "contextual_meaning": "meaning_in_this_context",
            "cultural_significance": "cultural_or_religious_importance"
        }}
    ],
    "semantic_interpretation": {{
        "literal_meaning": "word_for_word_interpretation",
        "contextual_meaning": "meaning_considering_context",
        "cultural_layers": "cultural_and_historical_significance"
    }},
    "philological_assessment": {{
        "confidence_level": 0.0,
        "challenging_elements": ["list of difficult aspects"],
        "scholarly_notes": "additional academic observations"
    }}
}}"""

def craft_contextual_translation_prompt(
    text: str, 
    analysis: dict, 
    target_lang: str, 
    genre: str = "general",
    source_lang: str = None
) -> str:
    """Create context-aware translation prompt"""
    
    target_info = get_language_info(target_lang)
    source_info = get_language_info(source_lang) if source_lang else {}
    
    target_name = target_info.get("name", target_lang)
    target_period = target_info.get("period", "Classical")
    
    genre_instruction = GENRE_INSTRUCTIONS.get(genre, GENRE_INSTRUCTIONS["general"])
    
    analysis_summary = json.dumps(analysis, indent=2)
    
    return f"""EXPERT ANCIENT LANGUAGE TRANSLATOR

TRANSLATION CONTEXT:
- Source Text: "{text}"
- Source Language: {source_info.get('name', 'Unknown')}
- Target Language: {target_name} ({target_period})
- Genre: {genre}
- Genre Requirements: {genre_instruction}

CRITICAL TRANSLATION PRINCIPLES:
1. **Language Authenticity**: Output MUST be in {target_name}, never English
2. **Historical Accuracy**: Use vocabulary appropriate to the {target_period} period
3. **Cultural Sensitivity**: Preserve cultural concepts and religious terminology

PHILOLOGICAL ANALYSIS PROVIDED:
{analysis_summary}

REQUIRED OUTPUT FORMAT (VALID JSON ONLY):
{{
    "primary_translation": "Main translation in {target_name} - NEVER use English here",
    "alternative_translations": [
        {{
            "translation": "Alternative rendering in {target_name}",
            "rationale": "Scholarly justification for this variant",
            "confidence": 0.0
        }}
    ],
    "translation_methodology": {{
        "approach": "translation strategy employed",
        "key_decisions": ["major translation choices made"],
        "cultural_adaptations": "how cultural elements were handled"
    }},
    "quality_assessment": {{
        "lexical_confidence": 0.0,
        "grammatical_confidence": 0.0, 
        "cultural_confidence": 0.0,
        "overall_confidence": 0.0
    }},
    "scholarly_notes": {{
        "untranslatable_elements": "concepts that resist direct translation",
        "philological_commentary": "academic observations on translation process"
    }}
}}

REMEMBER: The primary_translation field must contain ONLY {target_name}, never English."""

async def perform_philological_analysis(text: str, source_lang: str, detailed: bool = True) -> dict:
    """Perform comprehensive philological analysis"""
    if not model:
        raise HTTPException(status_code=500, detail="Analysis service unavailable")
    
    try:
        prompt = craft_enhanced_philological_prompt(text, source_lang, detailed)
        
        response = await asyncio.wait_for(
            model.generate_content_async(
                [prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096
                )
            ),
            timeout=ENHANCED_MODE_TIMEOUT / 3
        )
        
        analysis_json = extract_json(response.text.strip())
        return json.loads(analysis_json)
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Analysis timeout")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Analysis parsing failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def perform_contextual_translation(
    text: str,
    analysis: dict,
    target_lang: str,
    genre: str = "general",
    source_lang: str = None
) -> dict:
    """Perform context-aware translation"""
    if not model:
        raise HTTPException(status_code=500, detail="Translation service unavailable")
    
    try:
        prompt = craft_contextual_translation_prompt(text, analysis, target_lang, genre, source_lang)
        
        response = await asyncio.wait_for(
            model.generate_content_async(
                [prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4096
                )
            ),
            timeout=ENHANCED_MODE_TIMEOUT / 2
        )
        
        translation_json = extract_json(response.text.strip())
        result = json.loads(translation_json)
        
        # Validate translation quality
        primary_translation = result.get("primary_translation", "")
        
        # Check for English contamination
        if is_english_response(primary_translation, target_lang):
            result["primary_translation"] = f"[ENGLISH_DETECTED_IN_{target_lang.upper()}_OUTPUT]"
            if "quality_assessment" not in result:
                result["quality_assessment"] = {}
            result["quality_assessment"]["overall_confidence"] = 0.1
            if "scholarly_notes" not in result:
                result["scholarly_notes"] = {}
            result["scholarly_notes"]["validation_error"] = "AI model produced English instead of target language"
        
        # Check for anachronisms
        anachronisms = detect_anachronisms(primary_translation)
        if anachronisms:
            if "scholarly_notes" not in result:
                result["scholarly_notes"] = {}
            result["scholarly_notes"]["anachronisms_detected"] = anachronisms
            if "quality_assessment" in result and "cultural_confidence" in result["quality_assessment"]:
                result["quality_assessment"]["cultural_confidence"] *= 0.7
        
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Enhanced translation timeout")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Translation parsing failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def update_analytics(
    source_lang: str, 
    target_lang: str, 
    genre: str, 
    success: bool, 
    mode: str,
    confidence_scores: dict = None
):
    """Update translation analytics"""
    translation_stats["total_translations"] += 1
    translation_stats["language_pairs"][f"{source_lang}-{target_lang}"] += 1
    translation_stats["genre_distribution"][genre] += 1
    translation_stats["success_rates"][f"{source_lang}-{target_lang}"].append(success)
    translation_stats["mode_usage"][mode] += 1
    
    if confidence_scores:
        for metric, score in confidence_scores.items():
            translation_stats["confidence_scores"][metric].append(score)

# --- Main Smart Resurrection Endpoint ---
@router.post("/resurrect")
async def resurrect_language(request: TranslationRequest, background_tasks: BackgroundTasks):
    """Smart translation with automatic mode selection"""
    if not model:
        raise HTTPException(status_code=500, detail="Translation service unavailable")
    
    # Text length validation
    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long! Maximum {MAX_TEXT_LENGTH} characters. The spirits cannot handle such lengthy incantations."
        )
    
    validate_languages(request.source_language, request.target_language)
    
    # Determine mode
    use_simple = should_use_simple_mode(request.text, request.force_mode)
    
    try:
        if use_simple:
            # Use simple mode for long texts
            result = await simple_translation_mode(request)
            result["auto_mode_selection"] = {
                "selected_mode": "simple",
                "reason": f"Text length ({len(request.text)} chars) > {SIMPLE_MODE_THRESHOLD} chars" if not request.force_mode else "User forced simple mode",
                "text_length": len(request.text)
            }
            mode = "simple"
        else:
            # Use enhanced mode for short texts
            try:
                genre = request.genre_hint or detect_genre(request.text, request.source_language)
                
                # Philological Analysis
                analysis = None
                if request.include_analysis:
                    analysis = await perform_philological_analysis(request.text, request.source_language)
                else:
                    analysis = {"morphological_analysis": [], "semantic_interpretation": {"literal_meaning": request.text}}
                
                # Enhanced Translation
                translation_result = await perform_contextual_translation(
                    request.text, 
                    analysis, 
                    request.target_language, 
                    genre,
                    request.source_language
                )
                
                primary_translation = translation_result.get("primary_translation", "")
                confidence_scores = translation_result.get("quality_assessment", {})
                
                overall_confidence = confidence_scores.get("overall_confidence", 0.0)
                
                # Prepare enhanced response
                result = {
                    "source_text": request.text,
                    "source_language": {
                        "code": request.source_language,
                        "name": LANGUAGE_CAPABILITIES[request.source_language]["name"],
                        "confidence_tier": LANGUAGE_CAPABILITIES[request.source_language]["tier"]
                    },
                    "target_language": {
                        "code": request.target_language, 
                        "name": LANGUAGE_CAPABILITIES[request.target_language]["name"],
                        "confidence_tier": LANGUAGE_CAPABILITIES[request.target_language]["tier"]
                    },
                    "detected_genre": genre,
                    "resurrected_text": primary_translation,
                    "confidence_metrics": {
                        **confidence_scores,
                        "final_confidence": overall_confidence
                    },
                    "translation_metadata": {
                        "methodology": translation_result.get("translation_methodology", {}),
                        "scholarly_notes": translation_result.get("scholarly_notes", {}),
                        "timestamp": datetime.now().isoformat(),
                        "mode": "enhanced"
                    }
                }
                
                # Add optional components
                if request.include_analysis:
                    result["philological_analysis"] = analysis
                
                if request.include_alternatives:
                    result["alternative_translations"] = translation_result.get("alternative_translations", [])
                
                result["auto_mode_selection"] = {
                    "selected_mode": "enhanced",
                    "reason": f"Text length ({len(request.text)} chars) ≤ {SIMPLE_MODE_THRESHOLD} chars" if not request.force_mode else "User forced enhanced mode",
                    "text_length": len(request.text)
                }
                mode = "enhanced"
                
            except HTTPException as e:
                if e.status_code == 408:  # Timeout - fallback to simple mode
                    result = await simple_translation_mode(request)
                    result["auto_mode_selection"] = {
                        "selected_mode": "simple",
                        "reason": "Enhanced mode timeout - automatically fell back to simple mode",
                        "original_attempt": "enhanced",
                        "text_length": len(request.text)
                    }
                    mode = "simple_fallback"
                else:
                    raise e
        
        # Update analytics in background
        background_tasks.add_task(
            update_analytics,
            request.source_language,
            request.target_language, 
            result.get("detected_genre", "general"),
            result.get("confidence_metrics", {}).get("final_confidence", 0.0) > 0.5,
            mode,
            result.get("confidence_metrics", {})
        )
        
        return result
        
    except Exception as e:
        # Log error pattern
        translation_stats["error_patterns"][type(e).__name__] += 1
        
        raise HTTPException(
            status_code=500,
            detail=f"Smart necromancy failed: {type(e).__name__} - {str(e)}"
        )

# --- Additional Endpoints ---
@router.post("/resurrect-simple")
async def resurrect_simple_forced(request: TranslationRequest, background_tasks: BackgroundTasks):
    """Force simple translation mode"""
    request.force_mode = "simple"
    return await resurrect_language(request, background_tasks)

@router.post("/resurrect-enhanced")
async def resurrect_enhanced_forced(request: TranslationRequest, background_tasks: BackgroundTasks):
    """Force enhanced translation mode (may timeout on long texts)"""
    request.force_mode = "enhanced"
    return await resurrect_language(request, background_tasks)

@router.get("/")
async def necromancer_status():
    """Get smart necromancer status and capabilities"""
    return {
        "status": "Smart Necromancer ready to resurrect ancient voices",
        "version": "3.0.0-smart-mode",
        "api_configured": GEMINI_API_KEY is not None,
        "message": "MEDEA-NEUMOUSA Smart Necromancer Module",
        "features": {
            "auto_mode_selection": True,
            "simple_mode_threshold": f"{SIMPLE_MODE_THRESHOLD} characters",
            "max_text_length": f"{MAX_TEXT_LENGTH} characters",
            "timeout_protection": True,
            "fallback_system": True
        },
        "capabilities": {
            "supported_languages": len(LANGUAGE_CAPABILITIES),
            "language_tiers": {
                "tier_1": len([k for k, v in LANGUAGE_CAPABILITIES.items() if v.get("tier") == 1]),
                "tier_2": len([k for k, v in LANGUAGE_CAPABILITIES.items() if v.get("tier") == 2]),
                "tier_3": len([k for k, v in LANGUAGE_CAPABILITIES.items() if v.get("tier") == 3]),
                "tier_4": len([k for k, v in LANGUAGE_CAPABILITIES.items() if v.get("tier") == 4])
            }
        },
        "modes": {
            "simple": {
                "description": "Fast, reliable translation for longer texts",
                "timeout": f"{SIMPLE_MODE_TIMEOUT}s",
                "suitable_for": f"Texts > {SIMPLE_MODE_THRESHOLD} characters"
            },
            "enhanced": {
                "description": "Deep philological analysis for shorter texts", 
                "timeout": f"{ENHANCED_MODE_TIMEOUT}s",
                "suitable_for": f"Texts ≤ {SIMPLE_MODE_THRESHOLD} characters"
            }
        },
        "supported_languages": {
            code: {
                "name": info["name"],
                "confidence": info["confidence"],
                "tier": info["tier"],
                "period": info["period"]
            }
            for code, info in LANGUAGE_CAPABILITIES.items()
        }
    }

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify router is working"""
    return {
        "status": "SUCCESS",
        "message": "Necromancer router is working",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/test",
            "/resurrect", 
            "/health",
            "/analytics"
        ]
    }
@router.get("/analytics")
async def get_translation_analytics():
    """Provide comprehensive analytics including mode usage"""
    
    def calculate_success_rate(language_pair: str) -> float:
        successes = translation_stats["success_rates"].get(language_pair, [])
        return sum(successes) / len(successes) if successes else 0.0
    
    def calculate_average_confidence(metric: str) -> float:
        scores = translation_stats["confidence_scores"].get(metric, [])
        return sum(scores) / len(scores) if scores else 0.0
    
    return {
        "overview": {
            "total_translations": translation_stats["total_translations"],
            "unique_language_pairs": len(translation_stats["language_pairs"]),
            "supported_languages": len(LANGUAGE_CAPABILITIES),
            "active_genres": len(translation_stats["genre_distribution"])
        },
        "mode_usage": dict(translation_stats["mode_usage"]),
        "language_pair_statistics": {
            pair: {
                "translation_count": count,
                "success_rate": calculate_success_rate(pair)
            }
            for pair, count in translation_stats["language_pairs"].items()
        },
        "genre_distribution": dict(translation_stats["genre_distribution"]),
        "confidence_metrics": {
            metric: {
                "average": calculate_average_confidence(metric),
                "sample_count": len(scores)
            }
            for metric, scores in translation_stats["confidence_scores"].items()
        },
        "error_patterns": dict(translation_stats["error_patterns"])
    }

@router.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "gemini_api": "available" if GEMINI_API_KEY and model else "unavailable",
            "language_capabilities": "loaded",
            "smart_mode_selection": "active",
            "analytics": "active"
        },
        "metrics": {
            "total_translations": translation_stats["total_translations"],
            "supported_languages": len(LANGUAGE_CAPABILITIES),
            "active_language_pairs": len(translation_stats["language_pairs"]),
            "mode_usage": dict(translation_stats["mode_usage"])
        }
    }
