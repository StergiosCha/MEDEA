"""
MEDEA-NEUMOUSA: Semantic Similarity Analysis - NECROMANCER PATTERN
Find hidden connections between ancient texts
"""
import asyncio
import logging
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from typing import Any, Dict

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("MEDEA.Semantic")

# Configure Gemini with fallback
GEMINI_API_KEY = os.getenv("MEDEA_GEMINI_API_KEY")

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
        except:
            pass

model = models[0][1] if models else None

@dataclass
class TextSimilarity:
    """Similarity between two texts"""
    text1: str
    text2: str
    similarity_score: float
    semantic_connections: List[str]
    shared_themes: List[str]
    linguistic_relationship: str

@dataclass
class ClusterResult:
    """Result of text clustering analysis"""
    clusters: Dict[int, List[str]]
    cluster_themes: Dict[int, str]
    outliers: List[str]
    cluster_summaries: Dict[int, str]

def parse_llm_json_response(response_text: str) -> dict:
    """Parse JSON from LLM response with error handling - SAME AS NECROMANCER"""
    if not response_text:
        raise ValueError("Empty response")
    
    # Check for HTML error responses
    if response_text.strip().startswith('<!DOCTYPE') or '<html' in response_text:
        raise ValueError("LLM returned HTML error page")
    
    # Clean markdown formatting
    cleaned = response_text.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed. Response: {response_text[:200]}...")
        raise ValueError(f"Invalid JSON: {str(e)}")

class SemanticAnalyzer:
    """
    Analyze semantic relationships between ancient texts
    Using SAME pattern as necromancer
    """
    
    def __init__(self):
        # Check if model is available - same as necromancer
        self.model_available = model is not None
        
    async def analyze_similarity(
        self, 
        text1: str, 
        text2: str, 
        language1: str = "lat", 
        language2: str = "lat"
    ) -> TextSimilarity:
        """Analyze semantic similarity between two ancient texts with fallback"""
        
        if not models:
            logger.error("No Gemini models available")
            return TextSimilarity(
                text1=text1,
                text2=text2,
                similarity_score=0.0,
                semantic_connections=["No models available"],
                shared_themes=["Configuration error"],
                linguistic_relationship="configuration_error"
            )
        
        prompt = f"""You are a classical scholar. Compare these ancient texts and return ONLY valid JSON.

TEXT 1 ({language1}): {text1}

TEXT 2 ({language2}): {text2}

Return exactly this JSON format with no other text:

{{
    "similarity_score": 0.75,
    "semantic_connections": ["connection1", "connection2"],
    "shared_themes": ["theme1", "theme2"], 
    "linguistic_relationship": "thematic_similarity"
}}

Valid linguistic_relationship values: direct_quotation, allusion, parallel_tradition, thematic_similarity, unrelated

Only return the JSON object, nothing else."""

        # Try each model in fallback chain
        for model_name, current_model in models:
            try:
                response = await current_model.generate_content_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1000
                    )
                )
                
                # Parse JSON with error handling
                result = parse_llm_json_response(response.text)
                
                logger.info(f"✅ Similarity analysis succeeded with {model_name}")
                return TextSimilarity(
                    text1=text1,
                    text2=text2,
                    similarity_score=float(result.get("similarity_score", 0.5)),
                    semantic_connections=result.get("semantic_connections", []),
                    shared_themes=result.get("shared_themes", []),
                    linguistic_relationship=result.get("linguistic_relationship", "unknown")
                )
                
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                    logger.warning(f"💀 {model_name} quota exceeded, trying next model...")
                    continue
                else:
                    logger.error(f"❌ {model_name} error: {e}, trying next model...")
                    continue
        
        # All models failed
        logger.error("All models failed for similarity analysis")
        return TextSimilarity(
            text1=text1,
            text2=text2,
            similarity_score=0.0,
            semantic_connections=["All models failed"],
            shared_themes=["Analysis unavailable"],
            linguistic_relationship="analysis_failed"
        )
    
    async def analyze_multiple_texts(
        self, 
        texts: List[Dict[str, str]]  # [{"text": "...", "language": "lat", "title": "..."}]
    ) -> List[Tuple[int, int, TextSimilarity]]:
        """Analyze similarity between multiple texts (pairwise)"""
        
        results = []
        
        # Generate all pairs
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                text1_data = texts[i]
                text2_data = texts[j]
                
                similarity = await self.analyze_similarity(
                    text1_data["text"],
                    text2_data["text"],
                    text1_data.get("language", "lat"),
                    text2_data.get("language", "lat")
                )
                
                results.append((i, j, similarity))
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.5)
        
        return results
    
    async def cluster_texts(
        self, 
        texts: List[Dict[str, str]], 
        num_clusters: int = 3
    ) -> ClusterResult:
        """Cluster texts by semantic similarity"""
        
        # Get pairwise similarities
        similarities = await self.analyze_multiple_texts(texts)
        
        # Build similarity matrix
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i, j, sim in similarities:
            similarity_matrix[i, j] = sim.similarity_score
            similarity_matrix[j, i] = sim.similarity_score
        
        # Fill diagonal
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Convert to distance matrix for clustering
        distance_matrix = 1 - similarity_matrix
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=min(num_clusters, n),
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Organize results
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[idx]["text"])
        
        # Generate cluster themes using SAME PATTERN AS NECROMANCER
        cluster_themes = {}
        cluster_summaries = {}
        
        for cluster_id, cluster_texts in clusters.items():
            theme_prompt = f"""Analyze this cluster of ancient texts and identify their common theme. Return ONLY JSON:

TEXTS:
{chr(10).join([f"- {text[:100]}..." for text in cluster_texts])}

Return exactly this JSON format:
{{
    "main_theme": "brief theme description",
    "detailed_summary": "scholarly analysis of what unites these texts"
}}"""
            
            # Try each model in fallback chain
            theme_found = False
            for model_name, current_model in models:
                try:
                    response = await current_model.generate_content_async(
                        theme_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=800
                        )
                    )
                    
                    theme_data = parse_llm_json_response(response.text)
                    cluster_themes[cluster_id] = theme_data.get("main_theme", f"Cluster {cluster_id}")
                    cluster_summaries[cluster_id] = theme_data.get("detailed_summary", "No summary available")
                    theme_found = True
                    break
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "quota" in error_str:
                        logger.debug(f"{model_name} quota exceeded, trying next...")
                        continue
                    else:
                        continue
            
            if not theme_found:
                logger.warning(f"Theme analysis failed for cluster {cluster_id} on all models")
                # Fallback theme naming
                if len(cluster_texts) == 1:
                    cluster_themes[cluster_id] = f"Single Text Cluster {cluster_id}"
                else:
                    cluster_themes[cluster_id] = f"Mixed Texts ({len(cluster_texts)} texts)"
                cluster_summaries[cluster_id] = "Automatic theme analysis unavailable"
        
        return ClusterResult(
            clusters=clusters,
            cluster_themes=cluster_themes,
            outliers=[],  # Could implement outlier detection
            cluster_summaries=cluster_summaries
        )
    
    async def find_textual_echoes(
        self, 
        query_text: str, 
        corpus_texts: List[Dict[str, str]], 
        threshold: float = 0.7
    ) -> List[Tuple[Dict[str, str], TextSimilarity]]:
        """Find texts in a corpus that echo the query text"""
        
        echoes = []
        
        for corpus_text in corpus_texts:
            similarity = await self.analyze_similarity(
                query_text, 
                corpus_text["text"],
                "lat",  # Default language
                corpus_text.get("language", "lat")
            )
            
            if similarity.similarity_score >= threshold:
                echoes.append((corpus_text, similarity))
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.3)
        
        # Sort by similarity score
        echoes.sort(key=lambda x: x[1].similarity_score, reverse=True)
        
        return echoes
        
    def get_status(self) -> Dict[str, Any]:
        """Get semantic analyzer status - SAME PATTERN AS NECROMANCER"""
        return {
            "api_configured": GEMINI_API_KEY is not None,
            "model_ready": model is not None,
            "message": "Semantic Oracle ready" if model else "Gemini API key not configured"
        }
