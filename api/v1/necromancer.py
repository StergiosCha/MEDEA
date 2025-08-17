from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import json
import re

router = APIRouter()
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

import os
GEMINI_API_KEY = os.getenv("MEDEA_GEMINI_API_KEY")

# Configure Gemini
GEMINI_API_KEY = os.getenv("MEDEA_GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
else:
    model = None

# Language capabilities mapping
LANGUAGE_CAPABILITIES = {
    "lat": {"name": "Latin", "confidence": 0.9},
    "grc": {"name": "Ancient Greek", "confidence": 0.85},
    "sa": {"name": "Sanskrit", "confidence": 0.7},
    "he": {"name": "Hebrew", "confidence": 0.8},
    "arc": {"name": "Aramaic", "confidence": 0.6},
    "non": {"name": "Old Norse", "confidence": 0.6},
    "ang": {"name": "Old English", "confidence": 0.7},
    "got": {"name": "Gothic", "confidence": 0.4},
    "ara": {"name": "Arabic", "confidence": 0.75},
    "syr": {"name": "Syriac", "confidence": 0.4},
    "akk": {"name": "Akkadian", "confidence": 0.3},
    "hit": {"name": "Hittite", "confidence": 0.2},
    "cop": {"name": "Coptic", "confidence": 0.5},
    "av": {"name": "Avestan", "confidence": 0.3},
    "peo": {"name": "Old Persian", "confidence": 0.4},
    "gmh": {"name": "Middle High German", "confidence": 0.6},
    "luv": {"name": "Luvian", "confidence": 0.1},
    "pal": {"name": "Palaic", "confidence": 0.1}
}

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

def is_english_response(translation: str, target_lang: str) -> bool:
    """Detect if the LLM gave English instead of target language"""
    if not translation:
        return False
        
    translation = translation.lower().strip()
    
    # Common English words that shouldn't appear in ancient languages
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
        "goddess", "son", "destructive", "rage", "declare", "sprung", "peleus"
    }
    
    # Split into words and check for English
    words = re.findall(r'\b\w+\b', translation)
    english_word_count = sum(1 for word in words if word.lower() in common_english_words)
    
    # If more than 30% of words are common English words, it's probably English
    if len(words) > 2 and english_word_count / len(words) > 0.3:
        return True
        
    # Specific problematic phrases
    english_phrases = [
        "wrath sing", "sing goddess", "son of", "rage of", "declare o goddess",
        "this is", "the translation", "would be", "means", "in english",
        "peleus' son", "achilleus", "achilles"
    ]
    
    return any(phrase in translation for phrase in english_phrases)

def craft_aggressive_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Create aggressive anti-English prompt"""
    
    source_name = LANGUAGE_CAPABILITIES.get(source_lang, {}).get("name", source_lang)
    target_name = LANGUAGE_CAPABILITIES.get(target_lang, {}).get("name", target_lang)
    target_confidence = LANGUAGE_CAPABILITIES.get(target_lang, {}).get("confidence", 0.5)
    
    if target_confidence < 0.7:
        return f"""STOP. READ CAREFULLY.

TASK: Translate "{text}" FROM {source_name} INTO {target_name}

CRITICAL RULES:
1. DO NOT TRANSLATE TO ENGLISH
2. OUTPUT MUST BE IN {target_name} ONLY  
3. NO ENGLISH WORDS IN TRANSLATION

BAD EXAMPLE (FORBIDDEN):
"Wrath sing, goddess, of Peleus' son Achilleus" ← THIS IS ENGLISH! FORBIDDEN!

GOOD EXAMPLES:
- Old Norse: "Reiði kveð, gyðja, Peleussonar Akhilleusar"
- Aramaic: "רוגזא זמרי אלהתא ברה דפלאוס אכילאוס"  
- Gothic: "Wadi siggw, guda, Pilaiaussa sunius Akhillaiusis"

SOURCE: "{text}" ({source_name})
TARGET: {target_name}

REQUIRED JSON FORMAT:
{{
    "translated_text": "PUT_{target_name.upper()}_TEXT_HERE_NOT_ENGLISH",
    "confidence": {target_confidence},
    "alternatives": ["Alt1_in_{target_name}", "Alt2_in_{target_name}"],
    "notes": "Explain approach (English allowed here)"
}}

FORBIDDEN:
- Any English in "translated_text" field
- Phrases like "Wrath sing goddess"
- English explanations instead of translations

If you cannot generate {target_name}, use: "[CANNOT_GENERATE_{target_name.upper()}]"

TRANSLATE TO {target_name} NOW:"""
    
    else:
        return f"""Translate "{text}" FROM {source_name} TO {target_name}.

CRITICAL: Output must be in {target_name}, NOT English.

Good confidence for {target_name} - provide accurate translation.

JSON FORMAT:
{{
    "translated_text": "Text in {target_name} only",
    "confidence": {target_confidence},
    "alternatives": ["Alt1 in {target_name}", "Alt2 in {target_name}"],
    "notes": "Translation reasoning"
}}

FORBIDDEN: English in "translated_text" field."""

@router.get("/")
async def necromancer_status():
    return {
        "status": "Necromancer ready to resurrect ancient voices",
        "api_configured": GEMINI_API_KEY is not None,
        "message": "MEDEA-NEUMOUSA Necromancer Module - Anti-English Version",
        "supported_languages": list(LANGUAGE_CAPABILITIES.keys())
    }

@router.post("/resurrect")
async def resurrect_language(request: TranslationRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    # Validate language codes
    if request.source_language not in LANGUAGE_CAPABILITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {request.source_language}")
    
    if request.target_language not in LANGUAGE_CAPABILITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {request.target_language}")
    
    if request.source_language == request.target_language:
        raise HTTPException(status_code=400, detail="Source and target languages must be different")
    
    try:
        # Use aggressive prompt
        prompt = craft_aggressive_prompt(request.text, request.source_language, request.target_language)
        
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000
            )
        )
        
        # Parse response
        try:
            # Clean potential markdown
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
        except json.JSONDecodeError:
            result = {
                "translated_text": response.text,
                "confidence": 0.3,
                "alternatives": [],
                "notes": "Raw response - JSON parsing failed"
            }
        
        # Get translation
        translation = result.get("translated_text", "")
        
        # VALIDATION: Check if response is English
        if is_english_response(translation, request.target_language):
            print(f"🔮 ENGLISH DETECTED: '{translation}' for target {request.target_language}")
            translation = f"[ENGLISH_DETECTED_IN_{request.target_language.upper()}_FIELD]"
            result["confidence"] = 0.1
            result["notes"] = f"LLM gave English instead of {request.target_language}"
        
        return {
            "source_text": request.text,
            "resurrected_text": translation,
            "source_language": LANGUAGE_CAPABILITIES.get(request.source_language, {}).get("name", request.source_language),
            "target_language": LANGUAGE_CAPABILITIES.get(request.target_language, {}).get("name", request.target_language),
            "confidence": result.get("confidence", 0.5),
            "alternatives": result.get("alternatives", []),
            "spiritual_notes": result.get("notes", ""),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Necromancy failed: {str(e)}")