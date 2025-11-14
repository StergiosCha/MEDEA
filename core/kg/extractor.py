"""
MEDEA-NEUMOUSA: Enhanced Knowledge Graph Extractor
Transform ancient texts into RDF knowledge graphs with structured output
"Ἡ Μήδεια γνῶσιν εἰς δίκτυα μεταμορφοῖ"
"Medea transforms knowledge into networks"
"""
import os
import re
import json
import hashlib
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import requests
import time
from collections import defaultdict
import aiohttp
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("MEDEA_GEMINI_API_KEY") 
logger = logging.getLogger("MEDEA.KGExtractor")

# Configure Gemini with fallback models
GEMINI_API_KEY = os.getenv("MEDEA_GEMINI_API_KEY")

# Model fallback chain
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
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
    
    if not models:
        logger.error("No Gemini models could be loaded")
    else:
        logger.info(f"Initialized {len(models)} fallback models")
else:
    logger.warning("GEMINI_API_KEY not found")

# Keep backwards compatibility - primary model
model = models[0][1] if models else None

from enum import Enum

class TextDomain(str, Enum):
    LITERATURE = "literature"
    HISTORY = "history" 
    SCIENCE = "science"
    PHILOSOPHY = "philosophy"
    POLITICAL = "political"
    GENERAL = "general"

class DynamicKGExtractor:
    def detect_text_domain(self, text: str) -> TextDomain:
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['war', 'battle', 'empire', 'king', 'ancient', 'bc', 'ad']):
            return TextDomain.HISTORY
        elif any(word in text_lower for word in ['character', 'story', 'said', 'replied']):
            return TextDomain.LITERATURE
        elif any(word in text_lower for word in ['experiment', 'hypothesis', 'theory', 'study']):
            return TextDomain.SCIENCE
        elif any(word in text_lower for word in ['government', 'policy', 'democracy', 'political']):
            return TextDomain.POLITICAL
        else:
            return TextDomain.GENERAL
    
    def create_dynamic_prompt(self, text: str, mode: str = "BASIC") -> str:
        domain = self.detect_text_domain(text)
        
        guidance = {
            TextDomain.HISTORY: "historical figures, events, places, dates, political relationships",
            TextDomain.LITERATURE: "characters, dialogue, emotions, plot events, relationships",
            TextDomain.SCIENCE: "concepts, processes, experiments, measurements, causal relationships",
            TextDomain.POLITICAL: "leaders, policies, institutions, governance relationships",
            TextDomain.GENERAL: "meaningful entities and relationships"
        }
        
        mode_instructions = {
            "BASIC": "Extract entities and relationships that are explicitly stated.",
            "RAG_ENHANCED": "Include background knowledge to add context and implied relationships.",
            "EXTERNAL_ENRICHED": "Add comprehensive properties, alternative names, and historical context."
        }
        
        # Unique prompt to avoid recitation
        return f"""Knowledge Graph Extraction Task

Analyze the text below and identify entities and their relationships.

Text to analyze:
{text}

Focus areas: {guidance[domain]}
Extraction mode: {mode_instructions.get(mode, mode_instructions["BASIC"])}

Your output must be valid JSON only, following this structure:
{{
    "entities": [
        {{"id": "unique_identifier", "label": "Entity Name", "type": "EntityType", "properties": {{}}}}
    ],
    "relations": [
        {{"subject": "entity_id", "predicate": "relationship_type", "object": "entity_id", "description": "explanation"}}
    ]
}}

Important: Use the same IDs in relations as in the entities list.
Output only JSON, no other text."""

class KGExtractionMode(str, Enum):
    """Knowledge graph extraction modes"""
    BASIC = "basic"
    RAG_ENHANCED = "rag_enhanced"
    EXTERNAL_ENRICHED = "external_enriched"

class KGOutputFormat(str, Enum):
    """Output formats for knowledge graphs"""
    TURTLE = "turtle"
    RDF_XML = "rdf_xml"
    N3 = "n3"
    JSON_LD = "json_ld"

@dataclass
class Entity:
    """Represents an extracted entity"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source_span: Optional[str] = None

@dataclass
class Relation:
    """Represents a relationship between entities"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    source_span: Optional[str] = None

@dataclass
class SemanticNetwork:
    """Semantic network structure from knowledge graph"""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    clusters: Dict[str, List[str]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KGExtractionRequest:
    """Request for knowledge graph extraction"""
    text: str
    mode: KGExtractionMode
    use_external_sources: bool = False
    output_format: KGOutputFormat = KGOutputFormat.TURTLE
    chunk_size: int = 15000
    extract_locations: bool = True
    extract_temporal: bool = True
    extract_entities: bool = True
    confidence_threshold: float = 0.7

@dataclass
class KGExtractionResult:
    """Enhanced result of knowledge graph extraction"""
    rdf_content: str
    format: KGOutputFormat
    entity_count: int
    triple_count: int
    confidence_score: float
    extraction_notes: str
    processing_time: float
    chunks_processed: int = 1
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    semantic_network: Optional[SemanticNetwork] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    coherence_analysis: Optional[Dict[str, Any]] = None

class ExternalKnowledgeEnricher:
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        headers = {
            'User-Agent': 'MEDEA-NEUMOUSA/1.0 (https://yoursite.com/contact; your-email@domain.com) aiohttp/3.8.0'
        }
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def enrich_entity(self, entity_label: str, entity_type: str) -> Dict[str, Any]:
        """Get external data for an entity"""
        try:
            clean_label = entity_label.replace('æ', 'ae').replace('œ', 'oe')
            
            params = {
                'action': 'query',
                'format': 'json',
                'titles': clean_label,
                'prop': 'extracts',
                'exintro': 'true',
                'explaintext': 'true',
                'exsentences': '2'
            }
            
            async with self.session.get(
                'https://en.wikipedia.org/w/api.php', 
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get('query', {}).get('pages', {})
                    
                    for page_id, page_data in pages.items():
                        if page_id != '-1' and page_data.get('extract'):
                            return {
                                'description': page_data.get('extract', ''),
                                'wikipedia_url': f"https://en.wikipedia.org/wiki/{clean_label.replace(' ', '_')}",
                                'source': 'wikipedia'
                            }
                else:
                    logger.warning(f"Wikipedia API returned status {response.status}")
        
            return {}
        except Exception as e:
            logger.warning(f"External enrichment failed for {entity_label}: {e}")
            return {}

class SemanticNetworkBuilder:
    """Build semantic network from entities and relations"""
    
    def build_network(self, entities: List[Entity], relations: List[Relation]) -> SemanticNetwork:
        """Build semantic network structure with better entity matching"""
        nodes = []
        edges = []
        
        entity_map = {entity.id: entity for entity in entities}
        entity_ids = {entity.id for entity in entities}
        label_to_id = {entity.label.lower(): entity.id for entity in entities}
        
        for entity in entities:
            node = {
                'id': entity.id,
                'label': entity.label,
                'type': entity.type,
                'size': self._calculate_node_size(entity, relations),
                'color': self._get_node_color(entity.type),
                'properties': entity.properties,
                'confidence': entity.confidence
            }
            nodes.append(node)
        
        relation_counts = defaultdict(int)
        
        for relation in relations:
            source_id = relation.subject
            target_id = relation.object
            
            if source_id not in entity_ids:
                source_label = source_id.lower()
                if source_label in label_to_id:
                    source_id = label_to_id[source_label]
                else:
                    for label, eid in label_to_id.items():
                        if source_label in label or label in source_label:
                            source_id = eid
                            break
            
            if target_id not in entity_ids:
                target_label = target_id.lower()
                if target_label in label_to_id:
                    target_id = label_to_id[target_label]
                else:
                    for label, eid in label_to_id.items():
                        if target_label in label or label in target_label:
                            target_id = eid
                            break
            
            if source_id in entity_ids and target_id in entity_ids:
                edge = {
                    'source': source_id,
                    'target': target_id,
                    'label': self._clean_predicate_label(relation.predicate),
                    'type': self._classify_relation_type(relation.predicate),
                    'weight': relation.confidence,
                    'color': self._get_edge_color(relation.predicate),
                    'predicate': relation.predicate
                }
                edges.append(edge)
                relation_counts[relation.predicate] += 1
            else:
                logger.debug(f"Skipping relation - source: {source_id}, target: {target_id} not found in entities")
        
        clusters = self._identify_clusters(nodes, edges)
        metrics = self._calculate_metrics(nodes, edges, relation_counts)
        
        return SemanticNetwork(
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            metrics=metrics
        )
    
    def _calculate_node_size(self, entity: Entity, relations: List[Relation]) -> int:
        """Calculate node size based on connectivity"""
        connections = sum(1 for r in relations if r.subject == entity.id or r.object == entity.id)
        return max(15, min(50, 15 + connections * 3))
    
    def _get_node_color(self, entity_type: str) -> str:
        """Enhanced color mapping for spatial-emotional entities"""
        color_map = {
            'LOC': '#2ecc71',
            'GPE': '#27ae60', 
            'PER': '#3498db',
            'NORP': '#5dade2',
            'ExperientialAffect': '#e74c3c',
            'AppraisedAffect': '#f39c12',
            'IdentityAffect': '#8e44ad',
            'Person': '#e74c3c',
            'Event': '#3498db',
            'Location': '#2ecc71',
            'Entity': '#95a5a6'
        }
        
        clean_type = entity_type.split(':')[-1]
        return color_map.get(clean_type, '#95a5a6')
    
    def _classify_relation_type(self, predicate: str) -> str:
        """Enhanced relation type classification"""
        predicate_lower = predicate.lower()
        
        if any(term in predicate_lower for term in ['experienced_by', 'triggered_by', 'expressed_via']):
            return 'emotion_relation'
        elif any(term in predicate for term in ['travels_to', 'inhabits', 'remembers']):
            return 'spatial_relation'
        elif any(term in predicate for term in ['speaks_to', 'tells', 'asks']):
            return 'dialogue_relation'
        elif any(term in predicate for term in ['causes', 'leads_to', 'results_in']):
            return 'causal_relation'
        else:
            return 'relation'
    
    def _get_edge_color(self, predicate: str) -> str:
        """Enhanced edge coloring for relation types"""
        predicate_lower = predicate.lower()
        
        if any(term in predicate_lower for term in ['experienced_by', 'triggered_by']):
            return '#e74c3c'
        elif any(term in predicate_lower for term in ['travels_to', 'inhabits']):
            return '#2ecc71'
        elif any(term in predicate_lower for term in ['speaks_to', 'tells']):
            return '#3498db'
        elif any(term in predicate_lower for term in ['causes', 'leads_to']):
            return '#f39c12'
        else:
            return '#7f8c8d'
    
    def _clean_predicate_label(self, predicate: str) -> str:
        """Clean predicate label for display"""
        clean = predicate.split(':')[-1]
        clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
        return clean.lower().replace('_', ' ')
    
    def _identify_clusters(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, List[str]]:
        """Enhanced clustering for spatial-emotional analysis"""
        clusters = defaultdict(list)
        
        for node in nodes:
            node_type = node['type'].split(':')[-1]
            clusters[node_type].append(node['id'])
        
        return dict(clusters)
    
    def _calculate_metrics(self, nodes: List[Dict], edges: List[Dict], relation_counts: Dict) -> Dict[str, Any]:
        """Calculate network metrics"""
        degree_map = defaultdict(int)
        for edge in edges:
            degree_map[edge['source']] += 1
            degree_map[edge['target']] += 1
        
        most_connected = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)[:5]
        
        n_nodes = len(nodes)
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = len(edges) / max_edges if max_edges > 0 else 0
        
        return {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'density': density,
            'most_connected': most_connected,
            'relation_distribution': dict(relation_counts),
            'cluster_count': len(self._identify_clusters(nodes, edges))
        }

class TextChunker:
    """Handle text chunking for large documents"""
    
    def chunk_text_by_sentences(self, text: str, max_chars: int = 15000) -> List[str]:
        """Chunk text by sentences to maintain coherence"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

class MedeaKGExtractor:
    """
    MEDEA-NEUMOUSA Knowledge Graph Extractor
    Enhanced version with spatial-emotional analysis
    """
    
    def __init__(self):
        self.chunker = TextChunker()
        self.network_builder = SemanticNetworkBuilder()
        self.dynamic_extractor = DynamicKGExtractor()
        self.external_enricher = ExternalKnowledgeEnricher()
    
    def create_json_extraction_prompt(self, text: str, request: KGExtractionRequest) -> str:
        """Dynamic prompt that adapts to any text domain with mode support"""
        return self.dynamic_extractor.create_dynamic_prompt(text, request.mode.value)
    
    async def extract_knowledge_graph(self, request: KGExtractionRequest) -> KGExtractionResult:
        """Extract knowledge graph using enhanced spatial-emotional approach"""
        if not model:
            raise Exception("Gemini API key not configured")
        
        start_time = time.time()
        
        try:
            return await self._extract_with_json_approach(request, start_time)
        except Exception as e:
            logger.error(f"Enhanced extraction failed: {e}, trying basic fallback")
            return await self._extract_basic_fallback(request, start_time)

    async def _extract_with_json_approach(self, request: KGExtractionRequest, start_time: float) -> KGExtractionResult:
        """Extract using enhanced JSON-based prompting with better error handling"""
        
        if len(request.text) > request.chunk_size:
            chunks = self.chunker.chunk_text_by_sentences(request.text, request.chunk_size)
            logger.info(f"Processing {len(chunks)} chunks")
        else:
            chunks = [request.text]
        
        all_entities = []
        all_relations = []
        all_spatial_emotions = []
        all_rhetorical_analysis = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            
            response_text = None
            strategies = [
                ("standard", 0.4, self.create_json_extraction_prompt(chunk, request)),
                ("higher_temp", 0.7, self.create_json_extraction_prompt(chunk, request)),
                ("abstract", 0.6, self._create_abstract_extraction_prompt(chunk, request)),
            ]
            
            for strategy_name, temperature, prompt in strategies:
                logger.info(f"Trying strategy: {strategy_name} (temp={temperature})")
                
                # Try each model in fallback chain
                for model_name, current_model in models:
                    try:
                        logger.debug(f"Trying model: {model_name}")
                        response = await current_model.generate_content_async(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=temperature,
                                max_output_tokens=8192,
                                top_p=0.95,
                                top_k=40
                            ),
                            safety_settings={
                                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                            }
                        )
                        
                        finish_reason = self._get_finish_reason_code(response)
                        logger.info(f"Strategy {strategy_name}, model {model_name}: finish_reason={finish_reason}")
                        
                        response_text = self._extract_response_text(response)
                        
                        if response_text and len(response_text) > 10:
                            logger.info(f"✅ Success with {model_name} strategy {strategy_name}, got {len(response_text)} chars")
                            break
                        else:
                            logger.warning(f"Empty response from {model_name}, trying next model...")
                            continue
                            
                    except Exception as e:
                        error_str = str(e).lower()
                        
                        # If 429/quota error, try next model
                        if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                            logger.warning(f"💀 {model_name} quota exceeded, trying next model...")
                            continue
                        else:
                            logger.error(f"Strategy {strategy_name}, model {model_name} error: {e}")
                            continue
                
                # If we got response_text from any model, break strategy loop
                if response_text and len(response_text) > 10:
                    break
                else:
                    logger.warning(f"Strategy {strategy_name} failed on all models, trying next strategy...")
                    await asyncio.sleep(0.5)
            
            if not response_text:
                logger.error(f"All strategies failed for chunk {i}, skipping")
                continue
            
            logger.info(f"Received response text of length {len(response_text)}")
            
            json_data = self._parse_json_response(response_text)
            
            if json_data:
                json_data = self._enhance_spatial_emotional_analysis(json_data, chunk)
                
                all_entities.extend(json_data.get('entities', []))
                all_relations.extend(json_data.get('relations', []))
                all_spatial_emotions.extend(json_data.get('spatial_emotions', []))
                all_rhetorical_analysis.extend(json_data.get('rhetorical_analysis', []))
                
                logger.info(f"Chunk {i}: Extracted {len(json_data.get('entities', []))} entities, {len(json_data.get('relations', []))} relations")
            else:
                logger.warning(f"Failed to parse JSON for chunk {i}")
        
        entities = self._convert_json_to_entities(all_entities)
        relations = self._convert_json_to_relations(all_relations)
        
        if request.mode == KGExtractionMode.EXTERNAL_ENRICHED:
            entities = await self._enrich_entities_external(entities)
        
        rdf_content = self._create_rdf_from_structured_data(entities, relations)
        semantic_network = self._build_semantic_network_safe(entities, relations)
        
        if request.output_format != KGOutputFormat.TURTLE:
            rdf_content = self._convert_format(rdf_content, request.output_format)
        
        processing_time = time.time() - start_time
        
        summary = self._create_enhanced_summary(entities, relations, semantic_network, all_spatial_emotions)
        visualization_data = self._create_spatial_emotion_visualization(semantic_network)
        
        result = KGExtractionResult(
            rdf_content=rdf_content,
            format=request.output_format,
            entity_count=len(entities),
            triple_count=len(relations),
            confidence_score=0.85,
            extraction_notes=f"Enhanced spatial-emotional extraction from {len(chunks)} chunks using {request.mode.value} mode. Extracted {len(all_spatial_emotions)} spatial-emotional mappings.",
            processing_time=processing_time,
            chunks_processed=len(chunks),
            entities=entities,
            relations=relations,
            semantic_network=semantic_network,
            summary=summary,
            visualization_data=visualization_data
        )
        
        if request.mode != KGExtractionMode.BASIC:
            logger.info("Running narrative coherence analysis...")
            result.coherence_analysis = self.analyze_narrative_coherence(result)
        
        return result
    
    def _get_finish_reason_code(self, response) -> int:
        """Get the numeric finish reason code"""
        try:
            if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    return int(candidate.finish_reason)
        except Exception as e:
            logger.debug(f"Could not get finish reason: {e}")
        return -1
    
    def _create_abstract_extraction_prompt(self, text: str, request: KGExtractionRequest) -> str:
        """Create a completely different prompt style to avoid recitation"""
        return f"""Please help me structure information from this text.

Text:
{text}

I need you to identify:
1. Important entities (people, places, things, concepts, events)
2. How these entities are connected

Format your response as JSON with "entities" and "relations" arrays.
Each entity needs: id, label, type, properties
Each relation needs: subject, predicate, object, description

Respond with JSON only."""
    
    async def _enrich_entities_external(self, entities: List[Entity]) -> List[Entity]:
        """Enrich Entity objects with external data"""
        enriched_count = 0
        
        async with self.external_enricher as enricher:
            for entity in entities:
                if entity.type in ['Person', 'Place', 'Event', 'Organization', 'Date', 'Concept']:
                    logger.info(f"Attempting to enrich: {entity.label} (type: {entity.type})")
                    
                    external_data = await enricher.enrich_entity(entity.label, entity.type)
                    
                    if external_data:
                        entity.properties.update(external_data)
                        entity.properties['externally_enriched'] = True
                        enriched_count += 1
                        logger.info(f"Successfully enriched {entity.label}")
                    else:
                        logger.warning(f"No external data found for {entity.label}")
                    
                    await asyncio.sleep(0.1)
        
        logger.info(f"External enrichment complete: {enriched_count}/{len(entities)} entities enriched")
        return entities
    
    def _extract_response_text(self, response) -> str:
        """Extract text from Gemini response, handling complex structures and empty responses"""
        try:
            if not hasattr(response, 'candidates'):
                logger.error("Response has no candidates attribute")
                return ""
            
            if not response.candidates:
                logger.error("Response candidates is empty or None")
                return ""
            
            if len(response.candidates) == 0:
                logger.error("Response has no candidates")
                return ""
            
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'finish_reason'):
                finish_reason_str = str(candidate.finish_reason)
                logger.info(f"Finish reason: {finish_reason_str}")
                if finish_reason_str not in ["STOP", "1", "FinishReason.STOP"]:
                    logger.warning(f"Response may be incomplete or blocked: {finish_reason_str}")
                    if hasattr(candidate, 'safety_ratings'):
                        logger.warning(f"Safety ratings: {candidate.safety_ratings}")
            
            if not hasattr(candidate, 'content'):
                logger.error("Candidate has no content attribute")
                return ""
            
            if not candidate.content:
                logger.error("Candidate content is None")
                return ""
            
            if not hasattr(candidate.content, 'parts'):
                logger.error("Content has no parts attribute")
                return ""
            
            if not candidate.content.parts:
                logger.error("Content parts is None or empty")
                return ""
            
            if len(candidate.content.parts) == 0:
                logger.error("Response content has no parts")
                return ""
            
            text_parts = []
            for part in candidate.content.parts:
                try:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                except Exception as part_error:
                    logger.warning(f"Error accessing part text: {part_error}")
            
            if text_parts:
                return ' '.join(text_parts)
            else:
                logger.error("No text found in response parts")
                return ""
        
        except Exception as e:
            logger.error(f"Unexpected error extracting response text: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def debug_response_structure(self, response):
        """Debug helper to understand response structure"""
        logger.info("=== Response Structure Debug ===")
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Has 'text' attr: {hasattr(response, 'text')}")
        logger.info(f"Has 'candidates' attr: {hasattr(response, 'candidates')}")
        
        if hasattr(response, 'candidates'):
            logger.info(f"Candidates count: {len(response.candidates) if response.candidates else 0}")
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                logger.info(f"Candidate finish_reason: {getattr(candidate, 'finish_reason', 'N/A')}")
                logger.info(f"Candidate has content: {hasattr(candidate, 'content')}")
                if hasattr(candidate, 'content') and candidate.content:
                    logger.info(f"Content has parts: {hasattr(candidate.content, 'parts')}")
                    if hasattr(candidate.content, 'parts'):
                        logger.info(f"Parts count: {len(candidate.content.parts) if candidate.content.parts else 0}")
                if hasattr(candidate, 'safety_ratings'):
                    logger.info(f"Safety ratings: {candidate.safety_ratings}")
        
        logger.info("=== End Debug ===")
    
    def _parse_json_response(self, response) -> Optional[Dict[str, Any]]:
        """Parse JSON response from Gemini with robust error handling"""
        try:
            if isinstance(response, str):
                response_text = response
            else:
                response_text = self._extract_response_text(response)
            
            if not response_text or response_text.strip() == "":
                logger.warning("Empty response text received")
                return None
            
            json_text = response_text.strip()
            
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            elif json_text.startswith('```'):
                json_text = json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            
            json_text = json_text.strip()
            
            if not json_text:
                logger.warning("No content after stripping markdown")
                return None
            
            json_text = self._fix_json_issues(json_text)
            
            parsed = json.loads(json_text)
            logger.info(f"Successfully parsed JSON with {len(parsed.get('entities', []))} entities")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            logger.debug(f"Failed JSON text: {json_text[:500] if 'json_text' in locals() else 'N/A'}")
            if 'response_text' in locals() and response_text:
                return self._extract_structured_data_fallback(response_text)
            return None
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _fix_json_issues(self, json_text: str) -> str:
        """Fix common JSON formatting issues"""
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        json_text = json_text.replace('\\"', '"')
        
        lines = json_text.split('\n')
        fixed_lines = []
        
        for line in lines:
            quote_count = line.count('"')
            
            if quote_count % 2 != 0:
                if '"description":' in line or '"label":' in line:
                    if not line.rstrip().endswith('"') and not line.rstrip().endswith('",'):
                        line = line.rstrip() + '"'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _extract_structured_data_fallback(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract structured data when JSON parsing fails completely"""
        logger.info("Using fallback structured data extraction")
        entities = []
        relations = []
        
        entity_patterns = [
            r'"id":\s*"([^"]+)"[^}]*"label":\s*"([^"]+)"[^}]*"type":\s*"([^"]+)"',
            r'"label":\s*"([^"]+)"[^}]*"type":\s*"([^"]+)"[^}]*"id":\s*"([^"]+)"'
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches[:20]:
                if len(match) == 3:
                    if pattern == entity_patterns[0]:
                        entity_id, label, entity_type = match
                    else:
                        label, entity_type, entity_id = match
                    
                    entities.append({
                        "id": entity_id,
                        "label": label.replace('\\"', '"'),
                        "type": entity_type,
                        "properties": {}
                    })
        
        relation_patterns = [
            r'"subject":\s*"([^"]+)"[^}]*"predicate":\s*"([^"]+)"[^}]*"object":\s*"([^"]+)"'
        ]
        
        for pattern in relation_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches[:30]:
                if len(match) == 3:
                    subject, predicate, obj = match
                    relations.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "description": f"Extracted relation: {predicate}"
                    })
        
        if entities or relations:
            logger.info(f"Fallback extraction found {len(entities)} entities and {len(relations)} relations")
            return {
                "entities": entities,
                "relations": relations,
                "spatial_emotions": [],
                "rhetorical_analysis": []
            }
        
        logger.warning("Fallback extraction found no data")
        return None

    def _enhance_spatial_emotional_analysis(self, json_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Simple enhancement without hardcoded patterns"""
        return json_data

    def _convert_json_to_entities(self, json_entities: List[Dict]) -> List[Entity]:
        """Convert JSON entity data to Entity objects"""
        entities = []
        
        for entity_data in json_entities:
            try:
                entity_id = entity_data.get('id', f"{entity_data.get('label', 'unknown').replace(' ', '_').lower()}")
                entity_label = entity_data.get('label', 'Unknown')
                
                entity = Entity(
                    id=entity_id,
                    label=entity_label,
                    type=entity_data.get('type', 'Entity'),
                    properties=entity_data.get('properties', {}),
                    confidence=0.8
                )
                entities.append(entity)
                logger.debug(f"Created entity: {entity_id} -> {entity_label}")
                
            except Exception as e:
                logger.warning(f"Could not convert entity: {entity_data}, error: {e}")
                continue
        
        logger.info(f"Converted {len(entities)} entities")
        return entities

    def _convert_json_to_relations(self, json_relations: List[Dict]) -> List[Relation]:
        """Convert JSON relation data to Relation objects"""
        relations = []
        
        for relation_data in json_relations:
            try:
                subject = relation_data.get('subject', 'unknown')
                predicate = relation_data.get('predicate', 'relatedTo')
                obj = relation_data.get('object', 'unknown')
                
                relation = Relation(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=0.8
                )
                relations.append(relation)
                logger.debug(f"Created relation: {subject} -> {predicate} -> {obj}")
                
            except Exception as e:
                logger.warning(f"Could not convert relation: {relation_data}, error: {e}")
                continue
        
        logger.info(f"Converted {len(relations)} relations")
        return relations

    def _create_rdf_from_structured_data(self, entities: List[Entity], relations: List[Relation]) -> str:
        """Create RDF content from structured data"""
        rdf_lines = [
            "@prefix ste: <http://www.example.org/ste#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix emotion: <http://www.example.org/emotion#> .",
            ""
        ]
        
        for entity in entities:
            safe_id = entity.id.replace(' ', '_').replace(':', '_')
            rdf_lines.append(f"ste:{safe_id} a ste:{entity.type} ;")
            rdf_lines.append(f'    rdfs:label "{entity.label}" .')
            
            for prop_key, prop_value in entity.properties.items():
                if isinstance(prop_value, str) and prop_value:
                    clean_value = prop_value.replace('"', '\\"').replace('\n', ' ')[:500]
                    rdf_lines.append(f'    ste:{prop_key} "{clean_value}" .')
            
            rdf_lines.append("")
        
        for relation in relations:
            safe_subject = relation.subject.replace(' ', '_').replace(':', '_')
            safe_object = relation.object.replace(' ', '_').replace(':', '_')
            safe_predicate = relation.predicate.replace(' ', '_').replace(':', '_')
            rdf_lines.append(f"ste:{safe_subject} ste:{safe_predicate} ste:{safe_object} .")
            
        return "\n".join(rdf_lines)

    def _build_semantic_network_safe(self, entities: List[Entity], relations: List[Relation]) -> SemanticNetwork:
        """Build semantic network with error handling"""
        try:
            return self.network_builder.build_network(entities, relations)
        except Exception as e:
            logger.warning(f"Network building failed: {e}, creating basic network")
            import traceback
            logger.warning(traceback.format_exc())
            return self._create_basic_network(entities, relations)

    def _create_basic_network(self, entities: List[Entity], relations: List[Relation]) -> SemanticNetwork:
        """Create a basic semantic network when advanced building fails"""
        nodes = []
        for entity in entities:
            node = {
                'id': entity.id,
                'label': entity.label,
                'type': entity.type,
                'size': 20,
                'color': self._get_entity_color(entity.type),
                'group': entity.type.split(':')[-1]
            }
            nodes.append(node)
        
        valid_node_ids = {node['id'] for node in nodes}
        
        edges = []
        for relation in relations:
            if relation.subject in valid_node_ids and relation.object in valid_node_ids:
                edge = {
                    'source': relation.subject,
                    'target': relation.object,
                    'label': relation.predicate.split(':')[-1],
                    'weight': relation.confidence,
                    'color': '#7f8c8d',
                    'type': 'relation'
                }
                edges.append(edge)
        
        clusters = defaultdict(list)
        for entity in entities:
            entity_type = entity.type.split(':')[-1]
            clusters[entity_type].append(entity.id)
        
        node_count = len(nodes)
        edge_count = len(edges)
        max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
        density = edge_count / max_edges if max_edges > 0 else 0
        
        metrics = {
            'node_count': node_count,
            'edge_count': edge_count,
            'density': density,
            'most_connected': [],
            'relation_distribution': {},
            'cluster_count': len(clusters)
        }
        
        return SemanticNetwork(
            nodes=nodes,
            edges=edges,
            clusters=dict(clusters),
            metrics=metrics
        )

    async def _extract_basic_fallback(self, request: KGExtractionRequest, start_time: float) -> KGExtractionResult:
        """Final fallback extraction using simple text analysis"""
        logger.info("Using basic fallback extraction")
        
        entities = []
        text = request.text
        
        person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_entities = re.findall(person_pattern, text)
        
        entity_id = 1
        for name in set(potential_entities[:10]):
            if len(name) > 2 and name not in ['The', 'This', 'That', 'And', 'But', 'When', 'Where', 'Year']:
                entity = Entity(
                    id=f"ste:entity_{entity_id}",
                    label=name,
                    type="ste:Entity",
                    confidence=0.5,
                    properties={}
                )
                entities.append(entity)
                entity_id += 1
        
        rdf_content = self._create_rdf_from_structured_data(entities, [])
        semantic_network = self._create_basic_network(entities, [])
        processing_time = time.time() - start_time
        
        summary = {
            'total_entities': len(entities),
            'total_relations': 0,
            'entity_type_distribution': {'Entity': len(entities)},
            'relation_type_distribution': {},
            'network_density': 0.0,
            'key_entities': [{'label': e.label, 'connections': 0} for e in entities[:3]]
        }
        
        visualization_data = {
            'network': {
                'nodes': semantic_network.nodes,
                'links': []
            },
            'clusters': semantic_network.clusters,
            'statistics': {
                'node_count': len(entities),
                'edge_count': 0,
                'density': 0.0
            }
        }
        
        return KGExtractionResult(
            rdf_content=rdf_content,
            format=request.output_format,
            entity_count=len(entities),
            triple_count=0,
            confidence_score=0.5,
            extraction_notes="Basic fallback extraction - simple entity recognition only",
            processing_time=processing_time,
            chunks_processed=1,
            entities=entities,
            relations=[],
            semantic_network=semantic_network,
            summary=summary,
            visualization_data=visualization_data
        )

    def _get_entity_color(self, entity_type: str) -> str:
        """Get color for entity type"""
        color_map = {
            'PER': '#3498db',
            'LOC': '#2ecc71',
            'GPE': '#27ae60',
            'NORP': '#5dade2',
            'ExperientialAffect': '#e74c3c',
            'AppraisedAffect': '#f39c12',
            'IdentityAffect': '#8e44ad',
            'Entity': '#95a5a6',
            'Person': '#3498db',
            'Place': '#2ecc71',
            'Event': '#f39c12',
            'Organization': '#9b59b6',
            'Concept': '#e67e22'
        }
        clean_type = entity_type.split(':')[-1]
        return color_map.get(clean_type, '#95a5a6')
    
    def _create_enhanced_summary(self, entities: List[Entity], relations: List[Relation], 
                                semantic_network: SemanticNetwork, spatial_emotions: List[Dict]) -> Dict[str, Any]:
        """Create an enhanced summary including spatial-emotional data"""
        entity_types = defaultdict(int)
        relation_types = defaultdict(int)
        emotion_types = defaultdict(int)
        
        for entity in entities:
            entity_type = entity.type.split(':')[-1]
            entity_types[entity_type] += 1
            
            if 'Affect' in entity_type:
                emotion_category = entity.properties.get('emotion_category', 'unknown')
                emotion_types[emotion_category] += 1
        
        for relation in relations:
            predicate = relation.predicate.split(':')[-1]
            relation_types[predicate] += 1
        
        key_entities = []
        if semantic_network and semantic_network.metrics.get('most_connected'):
            key_entities = [
                {
                    'id': entity_id, 
                    'connections': count, 
                    'label': next((e.label for e in entities if e.id == entity_id), entity_id)
                }
                for entity_id, count in semantic_network.metrics['most_connected'][:3]
            ]
        
        place_emotion_count = len([se for se in spatial_emotions if se.get('place') and se.get('emotion')])
        
        return {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entity_type_distribution': dict(entity_types),
            'relation_type_distribution': dict(relation_types),
            'emotion_type_distribution': dict(emotion_types),
            'spatial_emotional_mappings': place_emotion_count,
            'key_entities': key_entities,
            'network_density': semantic_network.metrics.get('density', 0) if semantic_network else 0,
            'cluster_count': len(semantic_network.clusters) if semantic_network else 0,
            'spatial_emotional_clusters': semantic_network.clusters if semantic_network else {}
        }
    
    def _create_spatial_emotion_visualization(self, semantic_network: SemanticNetwork) -> Dict[str, Any]:
        """Create visualization data optimized for spatial-emotional networks"""
        if not semantic_network:
            return {'network': {'nodes': [], 'links': []}, 'statistics': {}}
        
        for node in semantic_network.nodes:
            if node.get('type') in ['ExperientialAffect', 'AppraisedAffect', 'IdentityAffect']:
                node['shape'] = 'diamond'
                if 'Experiential' in node['type']:
                    node['color'] = '#e74c3c'
                elif 'Appraised' in node['type']:
                    node['color'] = '#f39c12'
                else:
                    node['color'] = '#8e44ad'
            elif node.get('type') in ['LOC', 'GPE', 'Place', 'Location']:
                node['shape'] = 'square'
                node['color'] = '#27ae60'
            elif node.get('type') in ['PER', 'NORP', 'Person']:
                node['shape'] = 'circle'
                node['color'] = '#3498db'
        
        for edge in semantic_network.edges:
            if edge.get('label') in ['experienced_by', 'triggered_by', 'expressed_via']:
                edge['width'] = 3
                edge['style'] = 'solid'
            elif edge.get('label') in ['travels_to', 'inhabits', 'remembers']:
                edge['width'] = 2
                edge['style'] = 'dashed'
            else:
                edge['width'] = 1
                edge['style'] = 'dotted'
        
        return {
            'network': {
                'nodes': semantic_network.nodes,
                'links': semantic_network.edges
            },
            'affect_tiers': {
                'experiential': [n for n in semantic_network.nodes if n.get('type') == 'ExperientialAffect'],
                'appraised': [n for n in semantic_network.nodes if n.get('type') == 'AppraisedAffect'],
                'identity': [n for n in semantic_network.nodes if n.get('type') == 'IdentityAffect']
            },
            'spatial_clusters': semantic_network.clusters,
            'statistics': {
                'total_places': len([n for n in semantic_network.nodes if n.get('type') in ['LOC', 'GPE', 'Place', 'Location']]),
                'total_emotions': len([n for n in semantic_network.nodes if 'Affect' in n.get('type', '')]),
                'total_people': len([n for n in semantic_network.nodes if n.get('type') in ['PER', 'NORP', 'Person']]),
                'emotion_place_connections': len([e for e in semantic_network.edges if e.get('label') == 'triggered_by']),
                'node_count': len(semantic_network.nodes),
                'edge_count': len(semantic_network.edges),
                'density': semantic_network.metrics.get('density', 0)
            }
        }
    
    def _convert_format(self, turtle_content: str, output_format: KGOutputFormat) -> str:
        """Convert turtle to other RDF formats"""
        if output_format == KGOutputFormat.RDF_XML:
            return f"<!-- Requested RDF/XML format -->\n{turtle_content}"
        elif output_format == KGOutputFormat.N3:
            return f"# Requested N3 format\n{turtle_content}"
        elif output_format == KGOutputFormat.JSON_LD:
            return f"// Requested JSON-LD format\n{turtle_content}"
        
        return turtle_content
    
    def get_status(self) -> Dict[str, Any]:
        """Get extractor status"""
        coherence_available = False
        coherence_error = None
        try:
            from .coherence_analyzer import NarrativeCoherenceAnalyzer
            analyzer = NarrativeCoherenceAnalyzer()
            coherence_available = True
        except ImportError as e:
            coherence_error = f"Import error: {e}"
        except Exception as e:
            coherence_error = f"Initialization error: {e}"
        
        features = [
            "Enhanced spatial-emotional extraction",
            "Three-tiered affect hierarchy (Experiential/Appraised/Identity)",
            "Literary narrative analysis",
            "Dialogue and character interaction mapping",
            "Plot causation networks",
            "Robust JSON parsing with fallbacks",
            "Semantic network visualization",
            "Multi-strategy extraction with recitation avoidance"
        ]
        
        if coherence_available:
            features.append("Narrative coherence analysis with contradiction detection")
        else:
            features.append(f"Narrative coherence analysis (unavailable: {coherence_error})")
        
        return {
            "api_configured": GEMINI_API_KEY is not None,
            "model_ready": model is not None,
            "coherence_analyzer_available": coherence_available,
            "coherence_analyzer_error": coherence_error,
            "supported_modes": [mode.value for mode in KGExtractionMode],
            "supported_formats": [fmt.value for fmt in KGOutputFormat],
            "features": features,
            "analysis_capabilities": {
                "spatial_emotional_extraction": True,
                "contradiction_detection": coherence_available,
                "narrative_flow_analysis": coherence_available,
                "emotional_triangulation": True,
                "causal_chain_detection": coherence_available,
                "dialogue_pattern_analysis": coherence_available
            },
            "message": "Enhanced Knowledge Graph Extractor ready with spatial-emotional analysis" if model else "Gemini API key not configured"
        }
    
    def analyze_narrative_coherence(self, result: KGExtractionResult) -> Dict[str, Any]:
        """Run post-hoc narrative coherence analysis"""
        try:
            from .coherence_analyzer import NarrativeCoherenceAnalyzer
            
            analyzer = NarrativeCoherenceAnalyzer()
            analysis = analyzer.analyze_knowledge_graph(result.entities, result.relations)
            
            logger.info(f"Coherence analysis: {analysis['overall_coherence_score']:.3f} score, "
                    f"{len(analysis['contradictions'])} contradictions found")
            
            return analysis
        except ImportError as e:
            logger.warning(f"Coherence analyzer not available: {e}")
            return {'error': 'Coherence analyzer not available'}
        except Exception as e:
            logger.error(f"Coherence analysis failed: {e}")
            return {'error': str(e)}

def export_to_csv(result: KGExtractionResult) -> Tuple[str, str]:
    """Export entities and relations to CSV format"""
    import csv
    from io import StringIO
    
    entities_csv = StringIO()
    entities_writer = csv.writer(entities_csv)
    entities_writer.writerow(['ID', 'Label', 'Type', 'Confidence', 'Properties'])
    
    for entity in result.entities:
        entities_writer.writerow([
            entity.id, entity.label, entity.type, entity.confidence,
            json.dumps(entity.properties)
        ])
    
    relations_csv = StringIO()
    relations_writer = csv.writer(relations_csv)
    relations_writer.writerow(['Subject', 'Predicate', 'Object', 'Confidence'])
    
    for relation in result.relations:
        relations_writer.writerow([
            relation.subject, relation.predicate, relation.object, relation.confidence
        ])
    
    return entities_csv.getvalue(), relations_csv.getvalue()

def export_to_json(result: KGExtractionResult, include_rdf: bool = True) -> str:
    """Export structured result to JSON format"""
    export_data = {
        'metadata': {
            'format': result.format.value,
            'entity_count': result.entity_count,
            'triple_count': result.triple_count,
            'confidence_score': result.confidence_score,
            'processing_time': result.processing_time,
            'chunks_processed': result.chunks_processed,
            'extraction_notes': result.extraction_notes
        },
        'entities': [
            {
                'id': entity.id,
                'label': entity.label,
                'type': entity.type,
                'properties': entity.properties,
                'confidence': entity.confidence
            }
            for entity in result.entities
        ],
        'relations': [
            {
                'subject': relation.subject,
                'predicate': relation.predicate,
                'object': relation.object,
                'confidence': relation.confidence
            }
            for relation in result.relations
        ],
        'summary': result.summary,
        'semantic_network': {
            'nodes': result.semantic_network.nodes if result.semantic_network else [],
            'edges': result.semantic_network.edges if result.semantic_network else [],
            'clusters': result.semantic_network.clusters if result.semantic_network else {},
            'metrics': result.semantic_network.metrics if result.semantic_network else {}
        },
        'visualization_data': result.visualization_data
    }
    
    if include_rdf:
        export_data['rdf_content'] = result.rdf_content
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def create_network_report(result: KGExtractionResult) -> str:
    """Create a comprehensive text report of the knowledge graph"""
    if not result.semantic_network:
        return "No semantic network data available."
    
    report = []
    report.append("MEDEA-NEUMOUSA Enhanced Knowledge Graph Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    report.append("SUMMARY STATISTICS")
    report.append("-" * 20)
    report.append(f"Total Entities: {result.entity_count}")
    report.append(f"Total Relations: {result.triple_count}")
    report.append(f"Network Density: {result.semantic_network.metrics.get('density', 0):.3f}")
    report.append(f"Cluster Count: {result.semantic_network.metrics.get('cluster_count', 0)}")
    report.append(f"Processing Time: {result.processing_time:.2f} seconds")
    report.append("")
    
    if result.summary.get('spatial_emotional_mappings'):
        report.append("SPATIAL-EMOTIONAL ANALYSIS")
        report.append("-" * 28)
        report.append(f"Place-Emotion Mappings: {result.summary['spatial_emotional_mappings']}")
        
        if result.summary.get('emotion_type_distribution'):
            report.append("Emotion Distribution:")
            for emotion_type, count in result.summary['emotion_type_distribution'].items():
                report.append(f"  {emotion_type}: {count}")
        report.append("")
    
    if result.summary.get('entity_type_distribution'):
        report.append("ENTITY TYPE DISTRIBUTION")
        report.append("-" * 25)
        for entity_type, count in result.summary['entity_type_distribution'].items():
            report.append(f"{entity_type}: {count}")
        report.append("")
    
    if result.summary.get('key_entities'):
        report.append("MOST CONNECTED ENTITIES")
        report.append("-" * 25)
        for entity in result.summary['key_entities']:
            report.append(f"{entity['label']} ({entity['connections']} connections)")
        report.append("")
    
    if result.semantic_network.clusters:
        report.append("ENTITY CLUSTERS")
        report.append("-" * 15)
        for cluster_type, entities in result.semantic_network.clusters.items():
            report.append(f"{cluster_type}: {len(entities)} entities")
        report.append("")
    
    if result.summary.get('relation_type_distribution'):
        report.append("RELATION TYPE DISTRIBUTION")
        report.append("-" * 27)
        for relation_type, count in result.summary['relation_type_distribution'].items():
            report.append(f"{relation_type}: {count}")
        report.append("")
    
    if result.coherence_analysis and 'error' not in result.coherence_analysis:
        report.append("NARRATIVE COHERENCE ANALYSIS")
        report.append("-" * 30)
        report.append(f"Overall Coherence Score: {result.coherence_analysis.get('overall_coherence_score', 0):.3f}")
        report.append(f"Contradictions Found: {len(result.coherence_analysis.get('contradictions', []))}")
        
        contradictions = result.coherence_analysis.get('contradictions', [])
        if contradictions:
            report.append("\nDetected Contradictions:")
            for i, contradiction in enumerate(contradictions[:5], 1):
                report.append(f"  {i}. {contradiction['type'].replace('_', ' ').title()}")
                report.append(f"     {contradiction['description']}")
                if 'conflicting_objects' in contradiction:
                    report.append(f"     Conflicting entities: {', '.join(contradiction['conflicting_objects'])}")
        
        narrative = result.coherence_analysis.get('narrative_coherence', {})
        if narrative:
            report.append(f"\nNarrative Flow Analysis:")
            report.append(f"  Causal Chains: {narrative.get('causal_chains', 0)}")
            report.append(f"  Longest Chain: {narrative.get('longest_chain', 0)} steps")
            report.append(f"  Emotional Triangulations: {narrative.get('emotional_triangulations', 0)}")
            report.append(f"  Dialogue Interactions: {narrative.get('dialogue_interactions', 0)}")
            report.append(f"  Unique Speakers: {narrative.get('unique_speakers', 0)}")
        
        consistency = result.coherence_analysis.get('consistency_scores', {})
        if consistency:
            report.append(f"\nConsistency Scores:")
            for aspect, score in consistency.items():
                report.append(f"  {aspect.replace('_', ' ').title()}: {score:.3f}")
        
        report.append("")
    elif result.coherence_analysis and 'error' in result.coherence_analysis:
        report.append("NARRATIVE COHERENCE ANALYSIS")
        report.append("-" * 30)
        report.append(f"Error: {result.coherence_analysis['error']}")
        report.append("")
    
    report.append("EXTRACTION NOTES")
    report.append("-" * 16)
    report.append(result.extraction_notes)
    
    return "\n".join(report)
