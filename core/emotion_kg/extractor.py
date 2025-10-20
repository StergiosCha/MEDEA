"""
MEDEA-NEUMOUSA: LLM-Powered Emotion Knowledge Graph Extractor
Extract emotional content using shared LLM service with sentiment-based node sizing
"""
import os
import json
import asyncio
import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

# ✅ USE SHARED LLM SERVICE
from services.llm_service import llm_service

load_dotenv()

logger = logging.getLogger("MEDEA.EmotionKG")

@dataclass
class EmotionEntity:
    """Represents an extracted emotion with LLM-analyzed properties"""
    id: str
    label: str
    emotion_type: str
    intensity: float  # 0.0 to 1.0 (affects node size)
    sentiment_score: float  # -1.0 to 1.0 (affects node color)
    context: str
    triggers: List[str] = field(default_factory=list)
    manifestations: List[str] = field(default_factory=list)
    associated_entities: List[str] = field(default_factory=list)

@dataclass
class EmotionRelation:
    """Relationship between emotions or emotion triggers"""
    source: str
    target: str
    relation_type: str  # causes, leads_to, conflicts_with, amplifies, etc.
    strength: float  # 0.0 to 1.0
    description: str

@dataclass
class EmotionGraph:
    """Complete emotion knowledge graph with visualization data"""
    entities: List[EmotionEntity] = field(default_factory=list)
    relations: List[EmotionRelation] = field(default_factory=list)
    overall_sentiment: float = 0.0
    dominant_emotions: List[str] = field(default_factory=list)
    sentiment_distribution: Dict[str, float] = field(default_factory=dict)
    visualization_data: Dict[str, Any] = field(default_factory=dict)

class EmotionKGExtractor:
    """LLM-powered emotion knowledge graph extractor using shared service"""
    
    def __init__(self):
        # Check if LLM service is available
        if not llm_service.api_key:
            logger.warning("LLM service not configured for emotion extraction")
    
    def create_emotion_extraction_prompt(self, text: str) -> str:
        """Create prompt for extracting emotions using LLM"""
        return f"""Analyze this text for emotional content and return ONLY valid JSON.

Extract ALL emotions present, including negative ones like disgust, hatred, contempt, etc.

TEXT TO ANALYZE:
{text}

Return ONLY this JSON structure with NO additional text:
{{
    "emotions": [
        {{
            "id": "emotion_1",
            "label": "disappointment",
            "emotion_type": "primary_emotion",
            "intensity": 0.8,
            "sentiment_score": -0.6,
            "context": "brief context where emotion appears",
            "triggers": ["failure", "unmet expectations"],
            "manifestations": ["sighing", "drooping shoulders"],
            "associated_entities": ["person", "situation"]
        }}
    ],
    "relations": [
        {{
            "source": "emotion_1",
            "target": "emotion_2", 
            "relation_type": "leads_to",
            "strength": 0.9,
            "description": "disappointment leads to sadness"
        }}
    ],
    "overall_sentiment": -0.2,
    "dominant_emotions": ["disappointment"],
    "sentiment_distribution": {{
        "positive": 0.2,
        "negative": 0.7,
        "neutral": 0.1
    }}
}}

CRITICAL INTENSITY RULES - INTENSITY = HOW MUCH THIS EMOTION DOMINATES THE TEXT:

- intensity 0.9-1.0: This emotion COMPLETELY DOMINATES the text
  * Mentioned repeatedly throughout
  * Central theme of the entire text
  * Most important emotional element
  * Examples: Main character's overwhelming grief, pervasive fear in horror story

- intensity 0.7-0.8: This emotion is HIGHLY PROMINENT in the text
  * Mentioned multiple times
  * Major theme but not the only one
  * Significantly influences the narrative
  * Examples: Strong anger that drives plot, recurring joy in celebration scene

- intensity 0.5-0.6: This emotion is MODERATELY PRESENT
  * Mentioned several times or emphasized once
  * Important but not central
  * One of multiple emotional themes
  * Examples: Background sadness, supporting character's fear

- intensity 0.3-0.4: This emotion is MINOR in the text
  * Mentioned briefly or once
  * Not a major theme
  * Small part of emotional landscape
  * Examples: Fleeting annoyance, brief moment of surprise

- intensity 0.1-0.2: This emotion is BARELY DETECTABLE
  * Mentioned very briefly or implied
  * Minimal impact on text
  * Almost negligible presence
  * Examples: Subtle hint of unease, passing moment of contentment

IMPORTANT: Do NOT consider the "strength" of the emotion type itself. Consider ONLY how much space/emphasis this emotion takes up in the text.

Examples:
- If "mild disappointment" is the main theme throughout the text = intensity 0.9
- If "overwhelming rage" is mentioned once briefly = intensity 0.2
- If "gentle contentment" pervades the entire story = intensity 0.9
- If "devastating grief" appears in one sentence = intensity 0.3

sentiment_score: -1.0 to 1.0 (separate from intensity - affects color only)
- Very negative emotions: -1.0 to -0.5 (disgust, hatred, rage)
- Negative emotions: -0.5 to -0.1 (sadness, fear, anger)
- Neutral: -0.1 to 0.1
- Positive emotions: 0.1 to 0.5 (contentment, mild joy)
- Very positive emotions: 0.5 to 1.0 (ecstasy, bliss, love)

Return ONLY JSON, no markdown, no explanations"""

    async def extract_emotion_graph(self, text: str) -> EmotionGraph:
        """Extract emotion knowledge graph from text using LLM service"""
        
        # ✅ INITIALIZE LLM SERVICE IF NEEDED
        if not llm_service._initialized:
            await llm_service.initialize()
        
        if not llm_service.api_key:
            logger.warning("LLM service not available, using fallback")
            return await self._fallback_extraction(text)
        
        try:
            # Create extraction prompt
            prompt = self.create_emotion_extraction_prompt(text)
            
            logger.info(f"🎭 Extracting emotions from text ({len(text)} chars)")
            
            # ✅ CALL SHARED LLM SERVICE (with automatic model fallback)
            response_text = await llm_service.generate_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=4000
            )
            
            # Parse JSON response
            json_data = self._parse_json_response(response_text)
            
            if not json_data:
                logger.warning("Failed to parse LLM response, using fallback")
                return await self._fallback_extraction(text)
            
            # Convert to internal format
            emotions = self._convert_to_emotion_entities(json_data.get('emotions', []))
            relations = self._convert_to_emotion_relations(json_data.get('relations', []))
            
            logger.info(f"✅ Extracted {len(emotions)} emotions, {len(relations)} relations")
            
            # Create visualization data
            visualization_data = self._create_visualization_data(emotions, relations)
            
            return EmotionGraph(
                entities=emotions,
                relations=relations,
                overall_sentiment=json_data.get('overall_sentiment', 0.0),
                dominant_emotions=json_data.get('dominant_emotions', []),
                sentiment_distribution=json_data.get('sentiment_distribution', {}),
                visualization_data=visualization_data
            )
            
        except Exception as e:
            logger.error(f"Emotion extraction failed: {e}")
            return await self._fallback_extraction(text)

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM with error handling"""
        try:
            # Clean up response
            json_text = response_text.strip()
            
            # Remove markdown code blocks
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            elif json_text.startswith('```'):
                json_text = json_text[3:]
            
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            
            json_text = json_text.strip()
            
            # Parse JSON
            return json.loads(json_text)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            
            # Try to extract JSON with regex
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    json_content = json_match.group(0)
                    # Fix common JSON issues
                    json_content = re.sub(r',(\s*[}\]])', r'\1', json_content)
                    return json.loads(json_content)
                except:
                    pass
            
            return None

    def _convert_to_emotion_entities(self, emotions_data: List[Dict]) -> List[EmotionEntity]:
        """Convert JSON emotion data to EmotionEntity objects"""
        entities = []
        
        for i, emotion_data in enumerate(emotions_data):
            try:
                entity = EmotionEntity(
                    id=emotion_data.get('id', f"emotion_{i}"),
                    label=emotion_data.get('label', 'unknown'),
                    emotion_type=emotion_data.get('emotion_type', 'emotion'),
                    intensity=max(0.0, min(1.0, emotion_data.get('intensity', 0.5))),
                    sentiment_score=max(-1.0, min(1.0, emotion_data.get('sentiment_score', 0.0))),
                    context=emotion_data.get('context', ''),
                    triggers=emotion_data.get('triggers', []),
                    manifestations=emotion_data.get('manifestations', []),
                    associated_entities=emotion_data.get('associated_entities', [])
                )
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to convert emotion entity: {e}")
                continue
        
        return entities

    def _convert_to_emotion_relations(self, relations_data: List[Dict]) -> List[EmotionRelation]:
        """Convert JSON relation data to EmotionRelation objects"""
        relations = []
        
        for relation_data in relations_data:
            try:
                relation = EmotionRelation(
                    source=relation_data.get('source', ''),
                    target=relation_data.get('target', ''),
                    relation_type=relation_data.get('relation_type', 'relates_to'),
                    strength=max(0.0, min(1.0, relation_data.get('strength', 0.5))),
                    description=relation_data.get('description', '')
                )
                relations.append(relation)
            except Exception as e:
                logger.warning(f"Failed to convert emotion relation: {e}")
                continue
        
        return relations

    def _create_visualization_data(self, emotions: List[EmotionEntity], relations: List[EmotionRelation]) -> Dict[str, Any]:
        """Create visualization data with dominance-based node sizing"""
        nodes = []
        links = []
        
        # Create nodes with dominance-based sizing and sentiment-based coloring
        for emotion in emotions:
            # Node size based ONLY on dominance in text (intensity score)
            min_size = 15
            max_size = 70
            # Linear scaling: more dominant emotions get bigger nodes
            node_size = int(min_size + (emotion.intensity * (max_size - min_size)))
            
            # Color based on sentiment score (separate from size)
            node_color = self._get_sentiment_color(emotion.sentiment_score)
            
            # Border based on emotion type
            border_color = self._get_emotion_type_color(emotion.emotion_type)
            
            # Font size scales with dominance too
            font_size = max(10, int(10 + (emotion.intensity * 8)))
            
            node = {
                'id': emotion.id,
                'label': emotion.label,
                'size': node_size,
                'color': node_color,
                'borderColor': border_color,
                'borderWidth': 3,
                'intensity': emotion.intensity,
                'sentiment_score': emotion.sentiment_score,
                'emotion_type': emotion.emotion_type,
                'context': emotion.context,
                'triggers': emotion.triggers,
                'manifestations': emotion.manifestations,
                'font': {
                    'size': font_size,
                    'color': '#ffffff' if emotion.sentiment_score < -0.3 else '#000000'
                }
            }
            nodes.append(node)
        
        # Create links with strength-based styling
        for relation in relations:
            # Only add links if both nodes exist
            source_exists = any(n['id'] == relation.source for n in nodes)
            target_exists = any(n['id'] == relation.target for n in nodes)
            
            if source_exists and target_exists:
                link = {
                    'source': relation.source,
                    'target': relation.target,
                    'label': relation.relation_type,
                    'weight': relation.strength,
                    'width': max(1, int(relation.strength * 5)),
                    'color': self._get_relation_color(relation.relation_type),
                    'description': relation.description,
                    'arrows': 'to'
                }
                links.append(link)
        
        return {
            'network': {
                'nodes': nodes,
                'links': links
            }
        }

    def _get_sentiment_color(self, sentiment_score: float) -> str:
        """Get color based on sentiment score"""
        if sentiment_score >= 0.5:
            return '#00ff00'  # Bright green for very positive
        elif sentiment_score >= 0.1:
            return '#7fff00'  # Yellow-green for positive
        elif sentiment_score >= -0.1:
            return '#ffff00'  # Yellow for neutral
        elif sentiment_score >= -0.5:
            return '#ff7f00'  # Orange for negative
        else:
            return '#ff0000'  # Red for very negative

    def _get_emotion_type_color(self, emotion_type: str) -> str:
        """Get border color based on emotion type"""
        type_colors = {
            'primary_emotion': '#ffffff',
            'complex_emotion': '#00ffff',
            'mood': '#ff00ff',
            'emotion': '#cccccc'
        }
        return type_colors.get(emotion_type, '#cccccc')

    def _get_relation_color(self, relation_type: str) -> str:
        """Get color for relation type"""
        relation_colors = {
            'causes': '#ff4444',
            'leads_to': '#4444ff',
            'amplifies': '#ff8800',
            'diminishes': '#00ff88',
            'conflicts_with': '#ff0088',
            'triggers': '#8800ff',
            'follows': '#888888',
            'relates_to': '#cccccc'
        }
        return relation_colors.get(relation_type, '#cccccc')

    async def _fallback_extraction(self, text: str) -> EmotionGraph:
        """Fallback emotion extraction based on text prominence"""
        logger.info("Using fallback emotion extraction (keyword-based)")
        
        # Analyze text for emotion word frequency and emphasis
        emotion_patterns = {
            'anger': ['anger', 'angry', 'rage', 'furious', 'mad', 'irritated'],
            'disgust': ['disgust', 'disgusted', 'revolting', 'sickening', 'repulsive'],
            'hatred': ['hate', 'hatred', 'loathe', 'despise', 'detest'],
            'love': ['love', 'adore', 'cherish', 'treasure', 'devoted'],
            'joy': ['joy', 'happy', 'delighted', 'cheerful', 'elated'],
            'fear': ['fear', 'afraid', 'scared', 'terrified', 'anxious'],
            'sadness': ['sad', 'sadness', 'grief', 'sorrow', 'melancholy'],
            'disappointment': ['disappointed', 'disappointment', 'let down', 'failed'],
            'excitement': ['excited', 'excitement', 'thrilled', 'exhilarated'],
            'surprise': ['surprised', 'shocking', 'unexpected', 'amazed']
        }
        
        entities = []
        text_lower = text.lower()
        text_length = len(text)
        
        for emotion, patterns in emotion_patterns.items():
            # Count occurrences and calculate dominance
            total_matches = 0
            for pattern in patterns:
                total_matches += text_lower.count(pattern)
            
            if total_matches > 0:
                # Calculate dominance based on frequency and text length
                dominance = min(1.0, (total_matches * 100) / text_length)
                # Boost dominance if emotion appears multiple times
                if total_matches > 1:
                    dominance = min(1.0, dominance * (1 + total_matches * 0.2))
                
                # Assign sentiment scores
                sentiment_map = {
                    'anger': -0.7, 'disgust': -0.8, 'hatred': -0.9,
                    'fear': -0.6, 'sadness': -0.7, 'disappointment': -0.5,
                    'love': 0.9, 'joy': 0.8, 'excitement': 0.7, 'surprise': 0.1
                }
                
                entity = EmotionEntity(
                    id=f"emotion_{len(entities)}",
                    label=emotion,
                    emotion_type="primary_emotion",
                    intensity=dominance,  # Based on text dominance
                    sentiment_score=sentiment_map.get(emotion, 0.0),
                    context=f"Detected {total_matches} occurrences in text",
                    triggers=[],
                    manifestations=[],
                    associated_entities=[]
                )
                entities.append(entity)
        
        # Calculate overall sentiment
        overall_sentiment = sum(e.sentiment_score * e.intensity for e in entities) / sum(e.intensity for e in entities) if entities else 0.0
        
        # Create visualization data
        visualization_data = self._create_visualization_data(entities, [])
        
        logger.info(f"Fallback extraction: {len(entities)} emotions detected")
        
        return EmotionGraph(
            entities=entities,
            relations=[],
            overall_sentiment=overall_sentiment,
            dominant_emotions=[e.label for e in sorted(entities, key=lambda x: x.intensity, reverse=True)[:3]],
            sentiment_distribution={
                'positive': len([e for e in entities if e.sentiment_score > 0.1]) / len(entities) if entities else 0,
                'negative': len([e for e in entities if e.sentiment_score < -0.1]) / len(entities) if entities else 0,
                'neutral': len([e for e in entities if -0.1 <= e.sentiment_score <= 0.1]) / len(entities) if entities else 0
            },
            visualization_data=visualization_data
        )

    def create_rdf_from_emotions(self, emotion_graph: EmotionGraph) -> str:
        """Create RDF representation of emotion graph"""
        rdf_lines = [
            "@prefix emo: <http://www.example.org/emotion#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            ""
        ]
        
        # Add emotion entities
        for emotion in emotion_graph.entities:
            rdf_lines.extend([
                f"emo:{emotion.id} a emo:Emotion ;",
                f'    rdfs:label "{emotion.label}" ;',
                f"    emo:emotionType \"{emotion.emotion_type}\" ;",
                f"    emo:textDominance {emotion.intensity:.2f} ;",
                f"    emo:sentimentScore {emotion.sentiment_score:.2f} ;",
                f'    emo:context "{emotion.context.replace(chr(34), chr(39))}" .',
                ""
            ])
        
        # Add relations
        for relation in emotion_graph.relations:
            rdf_lines.extend([
                f"emo:{relation.source} emo:{relation.relation_type} emo:{relation.target} ;",
                f"    emo:relationStrength {relation.strength:.2f} ;",
                f'    emo:description "{relation.description.replace(chr(34), chr(39))}" .',
                ""
            ])
        
        return "\n".join(rdf_lines)

    def get_status(self) -> Dict[str, Any]:
        """Get emotion extractor status"""
        llm_status = llm_service.get_status()
        
        return {
            "api_configured": llm_status["api_configured"],
            "model_ready": llm_status["initialized"],
            "available_models": llm_status.get("available_models", []),
            "primary_model": llm_status.get("primary_model"),
            "features": [
                "LLM-powered emotion extraction with multi-model fallback",
                "Dominance-based node sizing (how much emotion dominates text)",
                "Support for all emotions including negative ones",
                "Emotion relationship analysis", 
                "Interactive visualization data",
                "RDF knowledge graph output",
                "Automatic fallback to keyword-based extraction"
            ],
            "llm_service": llm_status
        }

# Export functions
def export_emotion_graph_json(emotion_graph: EmotionGraph) -> str:
    """Export emotion graph as JSON"""
    data = {
        'emotions': [
            {
                'id': e.id,
                'label': e.label,
                'emotion_type': e.emotion_type,
                'intensity': e.intensity,
                'sentiment_score': e.sentiment_score,
                'context': e.context,
                'triggers': e.triggers,
                'manifestations': e.manifestations,
                'associated_entities': e.associated_entities
            }
            for e in emotion_graph.entities
        ],
        'relations': [
            {
                'source': r.source,
                'target': r.target,
                'relation_type': r.relation_type,
                'strength': r.strength,
                'description': r.description
            }
            for r in emotion_graph.relations
        ],
        'overall_sentiment': emotion_graph.overall_sentiment,
        'dominant_emotions': emotion_graph.dominant_emotions,
        'sentiment_distribution': emotion_graph.sentiment_distribution,
        'visualization_data': emotion_graph.visualization_data
    }
    
    return json.dumps(data, indent=2, ensure_ascii=False)

def create_emotion_analysis_report(emotion_graph: EmotionGraph) -> str:
    """Create text report of emotion analysis"""
    report = []
    report.append("EMOTION KNOWLEDGE GRAPH ANALYSIS")
    report.append("=" * 40)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL ANALYSIS")
    report.append("-" * 20)
    report.append(f"Total emotions detected: {len(emotion_graph.entities)}")
    report.append(f"Overall sentiment score: {emotion_graph.overall_sentiment:.2f}")
    report.append(f"Dominant emotions: {', '.join(emotion_graph.dominant_emotions)}")
    report.append("")
    
    # Sentiment distribution
    report.append("SENTIMENT DISTRIBUTION")
    report.append("-" * 20)
    dist = emotion_graph.sentiment_distribution
    report.append(f"Positive: {dist.get('positive', 0):.1%}")
    report.append(f"Negative: {dist.get('negative', 0):.1%}")
    report.append(f"Neutral: {dist.get('neutral', 0):.1%}")
    report.append("")
    
    # Individual emotions
    report.append("DETECTED EMOTIONS (by dominance)")
    report.append("-" * 20)
    
    # Sort by intensity (highest first)
    sorted_emotions = sorted(emotion_graph.entities, key=lambda x: x.intensity, reverse=True)
    
    for emotion in sorted_emotions:
        report.append(f"• {emotion.label.upper()}")
        report.append(f"  Dominance: {emotion.intensity:.2f} | Sentiment: {emotion.sentiment_score:.2f}")
        report.append(f"  Type: {emotion.emotion_type}")
        if emotion.context:
            report.append(f"  Context: {emotion.context[:100]}...")
        if emotion.triggers:
            report.append(f"  Triggers: {', '.join(emotion.triggers[:3])}")
        report.append("")
    
    # Relations
    if emotion_graph.relations:
        report.append("EMOTION RELATIONSHIPS")
        report.append("-" * 20)
        for relation in emotion_graph.relations:
            report.append(f"• {relation.source} --[{relation.relation_type}]--> {relation.target}")
            report.append(f"  Strength: {relation.strength:.2f}")
            if relation.description:
                report.append(f"  {relation.description}")
            report.append("")
    
    return "\n".join(report)
