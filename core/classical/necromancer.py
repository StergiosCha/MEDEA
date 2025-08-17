"""
MEDEA-NEUMOUSA: Necromancer Module - FIXED VERSION
Advanced classical language translator that speaks with the voices of the dead

"Ἡ Μήδεια τῶν νεκρῶν φωνὰς ἀνίστησι"
"Medea resurrects the voices of the dead"
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from app.config import settings, LANGUAGE_NAMES
from services.llm_service import llm_service

logger = logging.getLogger("MEDEA.Necromancer")


class DeadLanguage(str, Enum):
    """The tongues of the departed - Languages commanded by the Necromancer"""
    # Classical Foundation - The Pillars
    LATIN = "lat"
    ANCIENT_GREEK = "grc"
    
    # Ancient Near Eastern - Sacred Voices
    HEBREW = "he"
    ARAMAIC = "arc"
    AKKADIAN = "akk"
    SYRIAC = "syr"
    
    # Ancient European - Northern Spirits
    GOTHIC = "got"
    OLD_ENGLISH = "ang"
    OLD_NORSE = "non"
    MIDDLE_HIGH_GERMAN = "gmh"
    
    # Anatolian & Indo-European - Primordial Tongues
    HITTITE = "hit"
    SANSKRIT = "sa"
    AVESTAN = "av"
    OLD_PERSIAN = "peo"
    LUVIAN = "luv"
    PALAIC = "pal"
    
    # Late Antique - Transitional Voices
    COPTIC = "cop"
    ARABIC = "ara"


@dataclass
class NecromancyResult:
    """Result from summoning ancient voices"""
    source_text: str
    resurrected_text: str
    source_language: str
    target_language: str
    confidence: float
    alternative_voices: List[str]
    spiritual_notes: str
    morphological_essence: Optional[str] = None
    etymological_lineage: Optional[List[str]] = None
    cultural_resonance: Optional[str] = None
    necromantic_method: str = "divine_channeling"
    uncertainty_notes: Optional[str] = None
    source_period: Optional[str] = None
    target_register: Optional[str] = None


class Necromancer:
    """
    MEDEA's Necromancer - Master of Dead Languages
    Uses shared LLM service with fallbacks and realistic expectations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.spirits_awakened = False
        self.language_capabilities = self._initialize_language_capabilities()
        
    def _initialize_language_capabilities(self) -> Dict[DeadLanguage, Dict[str, Any]]:
        """Initialize realistic language capability assessments"""
        return {
            # High confidence - extensive training data
            DeadLanguage.LATIN: {
                "confidence": 0.9, 
                "corpus": "extensive", 
                "notes": "Classical Latin well-supported",
                "periods": ["Early", "Classical", "Late", "Medieval"],
                "registers": ["Literary", "Legal", "Religious", "Inscriptional"]
            },
            DeadLanguage.ANCIENT_GREEK: {
                "confidence": 0.85, 
                "corpus": "extensive", 
                "notes": "Classical Greek well-supported",
                "periods": ["Archaic", "Classical", "Hellenistic", "Byzantine"],
                "registers": ["Epic", "Dramatic", "Philosophical", "Biblical"]
            },
            DeadLanguage.HEBREW: {
                "confidence": 0.8, 
                "corpus": "good", 
                "notes": "Biblical Hebrew well-documented",
                "periods": ["Biblical", "Mishnaic", "Medieval"],
                "registers": ["Biblical", "Liturgical", "Legal", "Poetic"]
            },
            
            # Medium confidence - some training data
            DeadLanguage.ARAMAIC: {
                "confidence": 0.6, 
                "corpus": "limited", 
                "notes": "Biblical Aramaic possible, other dialects uncertain",
                "periods": ["Old", "Imperial", "Middle", "Late"],
                "registers": ["Biblical", "Legal", "Targum", "Liturgical"]
            },
            DeadLanguage.OLD_ENGLISH: {
                "confidence": 0.7, 
                "corpus": "moderate", 
                "notes": "Anglo-Saxon texts available",
                "periods": ["Early", "Classical", "Late"],
                "registers": ["Epic", "Legal", "Religious", "Chronicle"]
            },
            DeadLanguage.OLD_NORSE: {
                "confidence": 0.6, 
                "corpus": "limited", 
                "notes": "Saga texts available but complex",
                "periods": ["Old", "Classical", "Late"],
                "registers": ["Eddic", "Skaldic", "Saga", "Legal"]
            },
            DeadLanguage.ARABIC: {
                "confidence": 0.75, 
                "corpus": "good", 
                "notes": "Classical Arabic supported",
                "periods": ["Classical", "Abbasid", "Medieval"],
                "registers": ["Quranic", "Poetic", "Philosophical", "Scientific"]
            },
            DeadLanguage.SANSKRIT: {
                "confidence": 0.7, 
                "corpus": "good", 
                "notes": "Vedic and Classical Sanskrit available",
                "periods": ["Vedic", "Classical", "Epic", "Puranic"],
                "registers": ["Vedic", "Epic", "Kavya", "Shastra"]
            },
            
            # Lower confidence - minimal training data
            DeadLanguage.AKKADIAN: {
                "confidence": 0.3, 
                "corpus": "minimal", 
                "notes": "Requires significant reconstruction",
                "periods": ["Old", "Middle", "Neo"],
                "registers": ["Royal", "Legal", "Literary", "Administrative"]
            },
            DeadLanguage.HITTITE: {
                "confidence": 0.2, 
                "corpus": "minimal", 
                "notes": "Fragmentary, specialist knowledge required",
                "periods": ["Old", "Middle", "New"],
                "registers": ["Royal", "Religious", "Legal"]
            },
            DeadLanguage.GOTHIC: {
                "confidence": 0.4, 
                "corpus": "minimal", 
                "notes": "Limited to Biblical texts",
                "periods": ["Wulfilian"],
                "registers": ["Biblical"]
            },
            DeadLanguage.SYRIAC: {
                "confidence": 0.4, 
                "corpus": "limited", 
                "notes": "Some liturgical texts",
                "periods": ["Early", "Classical", "Late"],
                "registers": ["Biblical", "Liturgical", "Theological"]
            },
            
            # Very low confidence - specialist reconstruction needed
            DeadLanguage.LUVIAN: {
                "confidence": 0.1, 
                "corpus": "fragmentary", 
                "notes": "Extremely limited, requires Anatolian specialist",
                "periods": ["Cuneiform", "Hieroglyphic"],
                "registers": ["Royal", "Religious"]
            },
            DeadLanguage.PALAIC: {
                "confidence": 0.1, 
                "corpus": "fragmentary", 
                "notes": "Extremely limited corpus",
                "periods": ["Hittite Period"],
                "registers": ["Religious"]
            },
            DeadLanguage.AVESTAN: {
                "confidence": 0.3, 
                "corpus": "limited", 
                "notes": "Zoroastrian texts, complex grammar",
                "periods": ["Old", "Young"],
                "registers": ["Religious", "Liturgical"]
            },
            DeadLanguage.OLD_PERSIAN: {
                "confidence": 0.4, 
                "corpus": "limited", 
                "notes": "Royal inscriptions mainly",
                "periods": ["Achaemenid"],
                "registers": ["Royal", "Administrative"]
            },
            DeadLanguage.MIDDLE_HIGH_GERMAN: {
                "confidence": 0.6, 
                "corpus": "moderate", 
                "notes": "Medieval German literature",
                "periods": ["Early", "Classical", "Late"],
                "registers": ["Epic", "Courtly", "Religious", "Legal"]
            },
            DeadLanguage.COPTIC: {
                "confidence": 0.5, 
                "corpus": "limited", 
                "notes": "Christian Egyptian texts",
                "periods": ["Early", "Classical", "Late"],
                "registers": ["Biblical", "Liturgical", "Monastic"]
            }
        }
        
    async def awaken_spirits(self) -> None:
        """Awaken the spirits of ancient languages"""
        try:
            await llm_service.initialize()
            self.spirits_awakened = True
            logger.info("🔮 Necromancer spirits awakened via shared LLM service")
        except Exception as e:
            logger.error(f"💀 Failed to awaken spirits: {e}")
            raise
    
    def _get_language_capability(self, lang: DeadLanguage) -> Dict[str, Any]:
        """Get realistic capability assessment for a language"""
        return self.language_capabilities.get(lang, {
            "confidence": 0.1, 
            "corpus": "unknown", 
            "notes": "Insufficient training data",
            "periods": ["Unknown"],
            "registers": ["Unknown"]
        })
    
    def _craft_realistic_incantation(
        self, 
        text: str, 
        source_lang: DeadLanguage, 
        target_lang: DeadLanguage
    ) -> str:
        """Craft incantation based on realistic AI capabilities"""
        
        source_name = LANGUAGE_NAMES.get(source_lang.value, source_lang.value)
        target_name = LANGUAGE_NAMES.get(target_lang.value, target_lang.value)
        source_cap = self._get_language_capability(source_lang)
        target_cap = self._get_language_capability(target_lang)
        
        # For low-confidence target languages, use a more aggressive prompt
        if target_cap["confidence"] < 0.7:
            return f"""STOP. READ THIS CAREFULLY.

TASK: Translate "{text}" FROM {source_name} INTO {target_name}

CRITICAL RULES:
1. DO NOT TRANSLATE TO ENGLISH
2. DO NOT USE "translated_text" - USE "translation" 
3. OUTPUT MUST BE IN {target_name} LANGUAGE ONLY
4. NO ENGLISH WORDS IN THE TRANSLATION FIELD

BAD EXAMPLE (FORBIDDEN):
{{"translated_text": "Wrath sing, goddess..."}} ← WRONG! This is English!

GOOD EXAMPLE FOR OLD NORSE:
{{"translation": "Reiði syng, gyðja, Peleusar sonar Akhilleusar"}}

GOOD EXAMPLE FOR ARAMAIC: 
{{"translation": "רגזא זמרי אלהתא די בר פלאוס אכילאוס"}}

SOURCE: "{text}" ({source_name})
TARGET: {target_name} 

YOUR JOB: Convert the {source_name} text into {target_name}. Not English. Not explanations. Just {target_name}.

EXACT JSON FORMAT REQUIRED:
{{
    "translation": "PUT_{target_name.upper()}_TEXT_HERE_NOT_ENGLISH",
    "confidence": {target_cap["confidence"]},
    "method": "reconstructed", 
    "alternatives": ["Alt1_in_{target_name}", "Alt2_in_{target_name}"],
    "linguistic_notes": "Explain your approach (this can be English)",
    "uncertainty_notes": "What you're unsure about (this can be English)"
}}

FORBIDDEN RESPONSES:
- "Wrath sing, goddess..." (This is English!)
- "translated_text" field name
- Any English in the "translation" field
- Explanations instead of translations

If you cannot produce {target_name}, use: "[CANNOT_GENERATE_{target_name.upper()}]"

TRANSLATE NOW TO {target_name}:"""
        
        else:
            # Use standard prompt for high-confidence languages  
            return f"""TRANSLATE FROM {source_name} TO {target_name}

SOURCE: "{text}"
DIRECTION: {source_name} → {target_name}

CRITICAL: Do NOT translate to English. Output must be in {target_name}.

REQUIRED JSON (no other format accepted):
{{
    "translation": "Text in {target_name} only",
    "confidence": {target_cap["confidence"]},
    "method": "direct",
    "alternatives": ["Alt1 in {target_name}", "Alt2 in {target_name}"],
    "linguistic_notes": "Reasoning (English OK here)",
    "uncertainty_notes": "Uncertainties (English OK here)"
}}

FORBIDDEN: Using "translated_text" field or English in "translation" field."""

    def _is_english_response(self, translation: str, target_lang: DeadLanguage) -> bool:
        """Detect if the LLM gave an English response instead of target language"""
        if not translation:
            return False
            
        translation = translation.lower().strip()
        
        # Check for common English words that shouldn't appear in ancient languages
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
            "goddess", "son", "destructive", "rage", "declare", "sprung"
        }
        
        # Split into words and check for English
        words = translation.replace(",", " ").replace(".", " ").replace("'", " ").split()
        english_word_count = sum(1 for word in words if word.lower() in common_english_words)
        
        # If more than 30% of words are common English words, it's probably English
        if len(words) > 2 and english_word_count / len(words) > 0.3:
            return True
            
        # Specific checks for known problematic phrases
        english_phrases = [
            "wrath sing", "sing goddess", "son of", "rage of", "declare o goddess",
            "this is", "the translation", "would be", "means", "in english"
        ]
        
        if any(phrase in translation for phrase in english_phrases):
            return True
            
        # Language-specific validation - if target language should have specific patterns
        return self._lacks_target_language_patterns(translation, target_lang)

    def _lacks_target_language_patterns(self, translation: str, target_lang: DeadLanguage) -> bool:
        """Check if translation lacks expected patterns for the target language"""
        if len(translation) < 10:  # Short texts are harder to validate
            return False
            
        if target_lang == DeadLanguage.OLD_NORSE:
            # Old Norse should have þ, ð, or common Norse patterns
            norse_indicators = ["þ", "ð", "æ", "ø", "ek ", "er ", "at ", "ok ", "til ", "af "]
            return not any(indicator in translation for indicator in norse_indicators)
            
        elif target_lang == DeadLanguage.ARAMAIC:
            # Aramaic should have Semitic patterns
            aramaic_indicators = ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "ana ", "di ", "bar ", "min "]
            return not any(indicator in translation for indicator in aramaic_indicators)
            
        elif target_lang == DeadLanguage.GOTHIC:
            # Gothic should have characteristic patterns
            gothic_indicators = ["ik ", "þu ", "is ", "si ", "weis ", "jus ", "ains ", "jah "]
            return not any(indicator in translation for indicator in gothic_indicators)
            
        elif target_lang == DeadLanguage.SANSKRIT:
            # Sanskrit should have Devanagari or romanized Sanskrit patterns
            sanskrit_indicators = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ", "ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "्"]
            romanized_sanskrit = ["tat", "aham", "tvam", "sa", "sā", "te", "me", "svāhā", "oṃ"]
            has_devanagari = any(char in translation for char in sanskrit_indicators)
            has_romanized = any(pattern in translation for pattern in romanized_sanskrit)
            return not (has_devanagari or has_romanized)
            
        return False

    def _should_attempt_translation(self, source_lang: DeadLanguage, target_lang: DeadLanguage) -> bool:
        """Determine if translation should be attempted based on capabilities"""
        source_cap = self._get_language_capability(source_lang)
        target_cap = self._get_language_capability(target_lang)
        avg_confidence = (source_cap["confidence"] + target_cap["confidence"]) / 2
        return avg_confidence >= 0.4

    async def resurrect_language(
        self,
        text: str,
        source_lang: DeadLanguage,
        target_lang: DeadLanguage
    ) -> NecromancyResult:
        """Resurrect text from one dead language to another"""
        
        if not self.spirits_awakened:
            await self.awaken_spirits()
        
        source_cap = self._get_language_capability(source_lang)
        target_cap = self._get_language_capability(target_lang)
        avg_confidence = (source_cap["confidence"] + target_cap["confidence"]) / 2
        
        # Check if translation should be attempted
        if not self._should_attempt_translation(source_lang, target_lang):
            return NecromancyResult(
                source_text=text,
                resurrected_text="[TRANSLATION NOT ATTEMPTED - INSUFFICIENT TRAINING DATA]",
                source_language=LANGUAGE_NAMES.get(source_lang.value, source_lang.value),
                target_language=LANGUAGE_NAMES.get(target_lang.value, target_lang.value),
                confidence=avg_confidence,
                alternative_voices=[],
                spiritual_notes=f"Languages {source_lang.value} ↔ {target_lang.value} require specialized scholarly tools. Current AI models lack sufficient training data for reliable translation.",
                uncertainty_notes="Both languages have insufficient digital corpus for AI training",
                necromantic_method="scholarly_recommendation"
            )
        
        try:
            incantation = self._craft_realistic_incantation(text, source_lang, target_lang)
            
            logger.info(f"🔮 Channeling spirits: {source_lang} → {target_lang} for '{text[:50]}...'")
            
            # Use shared LLM service
            response = await llm_service.generate_completion(
                prompt=incantation,
                temperature=getattr(settings, 'necromancer_temperature', 0.3),
                max_tokens=getattr(settings, 'necromancer_max_tokens', 2000)
            )
            
            # Parse the necromantic result
            try:
                # Clean potential markdown formatting
                response_text = response.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                spirit_data = json.loads(response_text)
                
                # Fix wrong field names from non-compliant LLM responses
                if "translated_text" in spirit_data and "translation" not in spirit_data:
                    spirit_data["translation"] = spirit_data.pop("translated_text")
                    logger.warning(f"🔮 LLM used wrong field name 'translated_text', corrected to 'translation'")
                
                if "notes" in spirit_data and "linguistic_notes" not in spirit_data:
                    spirit_data["linguistic_notes"] = spirit_data.pop("notes")
                
                # VALIDATION: Check if translation is actually in target language
                translation = spirit_data.get("translation", "")
                if self._is_english_response(translation, target_lang):
                    logger.warning(f"🔮 Spirit gave English instead of {target_lang.value}, forcing fallback")
                    spirit_data["translation"] = f"[ENGLISH_DETECTED_IN_{target_lang.value}_FIELD]"
                    spirit_data["confidence"] = 0.1
                    spirit_data["method"] = "failed_direction"
                    spirit_data["uncertainty_notes"] = f"LLM output English instead of {target_lang.value}"
                
                return NecromancyResult(
                    source_text=text,
                    resurrected_text=spirit_data.get("translation", ""),
                    source_language=LANGUAGE_NAMES.get(source_lang.value, source_lang.value),
                    target_language=LANGUAGE_NAMES.get(target_lang.value, target_lang.value),
                    confidence=spirit_data.get("confidence", avg_confidence),
                    alternative_voices=spirit_data.get("alternatives", []),
                    spiritual_notes=spirit_data.get("linguistic_notes", ""),
                    morphological_essence=spirit_data.get("linguistic_notes"),
                    etymological_lineage=[],
                    cultural_resonance=spirit_data.get("scholarly_recommendation"),
                    uncertainty_notes=spirit_data.get("uncertainty_notes"),
                    source_period=spirit_data.get("source_period"),
                    target_register=spirit_data.get("target_register"),
                    necromantic_method=spirit_data.get("method", "llm_channeling")
                )
                
            except json.JSONDecodeError as e:
                logger.warning(f"🔮 Spirit spoke in non-JSON format, parsing as text: {e}")
                # Fallback: treat as plain resurrection
                return NecromancyResult(
                    source_text=text,
                    resurrected_text=response,
                    source_language=LANGUAGE_NAMES.get(source_lang.value, source_lang.value),
                    target_language=LANGUAGE_NAMES.get(target_lang.value, target_lang.value),
                    confidence=avg_confidence * 0.7,  # Lower confidence for unparsed response
                    alternative_voices=[],
                    spiritual_notes="The spirits spoke, but their message format was unclear",
                    uncertainty_notes="Response could not be parsed as structured data",
                    necromantic_method="raw_channeling"
                )
            
        except Exception as e:
            logger.error(f"💀 Necromantic ritual failed: {e}")
            # Return error result instead of raising
            return NecromancyResult(
                source_text=text,
                resurrected_text=f"[NECROMANTIC RITUAL FAILED: {str(e)}]",
                source_language=LANGUAGE_NAMES.get(source_lang.value, source_lang.value),
                target_language=LANGUAGE_NAMES.get(target_lang.value, target_lang.value),
                confidence=0.0,
                alternative_voices=[],
                spiritual_notes=f"The ritual was disrupted by dark forces: {str(e)}",
                uncertainty_notes="Complete failure - seek alternative methods",
                necromantic_method="failed_ritual"
            )
    
    async def mass_resurrection(
        self,
        texts: List[str],
        source_lang: DeadLanguage,
        target_lang: DeadLanguage
    ) -> List[NecromancyResult]:
        """Perform mass resurrection of multiple texts"""
        
        # Add delay between requests to avoid rate limiting
        results = []
        for i, text in enumerate(texts):
            try:
                result = await self.resurrect_language(text, source_lang, target_lang)
                results.append(result)
                
                # Small delay between requests
                if i < len(texts) - 1:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"💀 Failed to resurrect text {i}: {e}")
                # Continue with other texts
                continue
        
        return results
    
    async def divine_linguistic_connection(
        self,
        word1: str,
        lang1: DeadLanguage,
        word2: str,
        lang2: DeadLanguage
    ) -> Dict[str, Any]:
        """Divine the spiritual connection between words across ancient languages"""
        
        lang1_cap = self._get_language_capability(lang1)
        lang2_cap = self._get_language_capability(lang2)
        
        incantation = f"""As a historical linguist, analyze the relationship between these ancient words:

Word 1: "{word1}" in {LANGUAGE_NAMES.get(lang1.value, lang1.value)}
Word 2: "{word2}" in {LANGUAGE_NAMES.get(lang2.value, lang2.value)}

Language Capabilities:
- {lang1.value}: {lang1_cap['confidence']:.1f} confidence
- {lang2.value}: {lang2_cap['confidence']:.1f} confidence

Analyze their relationship through:
1. Etymological connection (cognates)
2. Borrowing relationships  
3. Phonetic correspondences
4. Semantic evolution
5. Historical contact

RESPOND AS JSON:
{{
    "relationship": "cognate|borrowing|parallel_evolution|coincidental|insufficient_data",
    "confidence": 0.75,
    "analysis": "Detailed linguistic analysis",
    "sound_changes": ["Specific phonetic transformations"],
    "semantic_evolution": "How meanings changed",
    "historical_context": "Cultural/historical factors",
    "uncertainty": "What aspects are uncertain"
}}

Be honest about limitations when language data is insufficient."""

        try:
            response = await llm_service.generate_completion(
                prompt=incantation,
                temperature=0.2,
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
            
        except Exception as e:
            logger.error(f"💀 Failed to divine linguistic connection: {e}")
            return {
                "relationship": "error",
                "confidence": 0.0,
                "analysis": f"Divination failed: {str(e)}",
                "sound_changes": [],
                "semantic_evolution": "Unknown due to error",
                "historical_context": "Could not be determined",
                "uncertainty": "Complete failure"
            }
    
    def get_commanded_languages(self) -> List[Dict[str, str]]:
        """Get all language pairs under necromantic command with realistic assessments"""
        pairs = []
        languages = list(DeadLanguage)
        
        for source in languages:
            for target in languages:
                if source != target:
                    source_cap = self._get_language_capability(source)
                    target_cap = self._get_language_capability(target)
                    avg_confidence = (source_cap["confidence"] + target_cap["confidence"]) / 2
                    
                    pairs.append({
                        "source": source.value,
                        "target": target.value,
                        "source_name": LANGUAGE_NAMES.get(source.value, source.value),
                        "target_name": LANGUAGE_NAMES.get(target.value, target.value),
                        "confidence": avg_confidence,
                        "feasible": avg_confidence >= 0.4,
                        "spiritual_connection": self._assess_linguistic_connection(source, target),
                        "notes": f"Source: {source_cap['notes']}, Target: {target_cap['notes']}"
                    })
        
        return pairs
    
    def _assess_linguistic_connection(self, lang1: DeadLanguage, lang2: DeadLanguage) -> str:
        """Assess the spiritual connection between two languages"""
        
        # Indo-European family connections
        ie_languages = {
            DeadLanguage.LATIN, DeadLanguage.ANCIENT_GREEK, DeadLanguage.SANSKRIT,
            DeadLanguage.GOTHIC, DeadLanguage.OLD_ENGLISH, DeadLanguage.OLD_NORSE,
            DeadLanguage.HITTITE, DeadLanguage.AVESTAN, DeadLanguage.OLD_PERSIAN,
            DeadLanguage.MIDDLE_HIGH_GERMAN
        }
        
        # Semitic family connections  
        semitic_languages = {
            DeadLanguage.HEBREW, DeadLanguage.ARAMAIC, DeadLanguage.AKKADIAN,
            DeadLanguage.SYRIAC, DeadLanguage.ARABIC
        }
        
        # Anatolian subfamily
        anatolian_languages = {
            DeadLanguage.HITTITE, DeadLanguage.LUVIAN, DeadLanguage.PALAIC
        }
        
        if lang1 in anatolian_languages and lang2 in anatolian_languages:
            return "anatolian_kinship"
        elif lang1 in ie_languages and lang2 in ie_languages:
            return "indo_european_kinship"
        elif lang1 in semitic_languages and lang2 in semitic_languages:
            return "semitic_kinship"
        elif (lang1 == DeadLanguage.LATIN and lang2 == DeadLanguage.ANCIENT_GREEK) or \
             (lang1 == DeadLanguage.ANCIENT_GREEK and lang2 == DeadLanguage.LATIN):
            return "classical_brotherhood"
        elif lang1 == DeadLanguage.COPTIC and lang2 in {DeadLanguage.ANCIENT_GREEK, DeadLanguage.ARABIC}:
            return "cultural_contact"
        else:
            return "distant_echoes"
    
    async def banish_spirits(self) -> None:
        """Banish the summoned spirits (cleanup)"""
        self.spirits_awakened = False
        logger.info("🌙 Necromantic spirits return to the underworld")
    
    def get_necromancer_status(self) -> Dict[str, Any]:
        """Get the current status of the Necromancer"""
        high_confidence_langs = sum(1 for cap in self.language_capabilities.values() 
                                  if cap["confidence"] >= 0.7)
        medium_confidence_langs = sum(1 for cap in self.language_capabilities.values() 
                                    if 0.4 <= cap["confidence"] < 0.7)
        low_confidence_langs = sum(1 for cap in self.language_capabilities.values() 
                                 if cap["confidence"] < 0.4)
        
        return {
            "spirits_awakened": self.spirits_awakened,
            "total_languages": len(self.language_capabilities),
            "high_confidence_languages": high_confidence_langs,
            "medium_confidence_languages": medium_confidence_langs,
            "low_confidence_languages": low_confidence_langs,
            "realistic_translation_pairs": sum(1 for source in DeadLanguage 
                                             for target in DeadLanguage 
                                             if source != target and self._should_attempt_translation(source, target)),
            "necromantic_method": "shared_llm_service",
            "power_source": "Nine Muses + Medea's Sorcery + Scholarly Honesty",
            "purpose": "Destroyer of Talos, Resurrector of Ancient Voices, Keeper of Linguistic Truth"
        }

    def get_language_details(self, lang: DeadLanguage) -> Dict[str, Any]:
        """Get detailed information about a specific language's capabilities"""
        cap = self._get_language_capability(lang)
        return {
            "language": lang.value,
            "name": LANGUAGE_NAMES.get(lang.value, lang.value),
            "confidence": cap["confidence"],
            "corpus_quality": cap["corpus"],
            "notes": cap["notes"],
            "available_periods": cap.get("periods", ["Unknown"]),
            "available_registers": cap.get("registers", ["Unknown"]),
            "recommended_for_translation": cap["confidence"] >= 0.4,
            "requires_specialist": cap["confidence"] < 0.3
        }