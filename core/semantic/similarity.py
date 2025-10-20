"""
MEDEA-NEUMOUSA: Semantic Similarity Analysis
Find hidden connections between ancient texts using shared LLM service
"""
import asyncio
import logging
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from dotenv import load_dotenv

# ✅ USE SHARED LLM SERVICE
from services.llm_service import llm_service

load_dotenv()

logger = logging.getLogger("MEDEA.Semantic")

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
    """Parse JSON from LLM response with error handling"""
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
    Uses shared LLM service with multi-model fallback
    """
    
    def __init__(self):
        # Check if LLM service is available
        self.model_available = llm_service.api_key is not None
        
    async def analyze_similarity(
        self, 
        text1: str, 
        text2: str, 
        language1: str = "lat", 
        language2: str = "lat"
    ) -> TextSimilarity:
        """Analyze semantic similarity between two ancient texts"""
        
        # ✅ INITIALIZE LLM SERVICE IF NEEDED
        if not llm_service._initialized:
            await llm_service.initialize()
        
        if not llm_service.api_key:
            logger.error("LLM service not configured for semantic analysis")
            return TextSimilarity(
                text1=text1,
                text2=text2,
                similarity_score=0.0,
                semantic_connections=["LLM service not configured"],
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

        try:
            logger.info(f"🔍 Analyzing similarity between texts ({len(text1)} and {len(text2)} chars)")
            
            # ✅ USE SHARED LLM SERVICE (with automatic model fallback)
            response_text = await llm_service.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse JSON with error handling
            result = parse_llm_json_response(response_text)
            
            logger.info(f"✅ Similarity score: {result.get('similarity_score', 0):.2f}")
            
            return TextSimilarity(
                text1=text1,
                text2=text2,
                similarity_score=float(result.get("similarity_score", 0.5)),
                semantic_connections=result.get("semantic_connections", []),
                shared_themes=result.get("shared_themes", []),
                linguistic_relationship=result.get("linguistic_relationship", "unknown")
            )
            
        except Exception as e:
            logger.error(f"Similarity analysis failed: {e}")
            # Simple fallback without assumptions
            return TextSimilarity(
                text1=text1,
                text2=text2,
                similarity_score=0.0,
                semantic_connections=["Analysis failed - please try again"],
                shared_themes=["Analysis unavailable"],
                linguistic_relationship="analysis_failed"
            )
    
    async def analyze_multiple_texts(
        self, 
        texts: List[Dict[str, str]]  # [{"text": "...", "language": "lat", "title": "..."}]
    ) -> List[Tuple[int, int, TextSimilarity]]:
        """Analyze similarity between multiple texts (pairwise)"""
        
        results = []
        total_pairs = (len(texts) * (len(texts) - 1)) // 2
        
        logger.info(f"🔍 Analyzing {total_pairs} text pairs")
        
        # Generate all pairs
        pair_count = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                pair_count += 1
                text1_data = texts[i]
                text2_data = texts[j]
                
                logger.info(f"Analyzing pair {pair_count}/{total_pairs}")
                
                similarity = await self.analyze_similarity(
                    text1_data["text"],
                    text2_data["text"],
                    text1_data.get("language", "lat"),
                    text2_data.get("language", "lat")
                )
                
                results.append((i, j, similarity))
                
                # Add small delay to avoid overwhelming the service
                await asyncio.sleep(0.5)
        
        logger.info(f"✅ Completed analysis of {total_pairs} pairs")
        return results
    
    async def cluster_texts(
        self, 
        texts: List[Dict[str, str]], 
        num_clusters: int = 3
    ) -> ClusterResult:
        """Cluster texts by semantic similarity"""
        
        logger.info(f"🔍 Clustering {len(texts)} texts into {num_clusters} clusters")
        
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
        
        logger.info(f"📊 Clusters formed: {len(clusters)}")
        
        # Generate cluster themes using LLM service
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
            
            try:
                # ✅ USE SHARED LLM SERVICE
                response_text = await llm_service.generate_completion(
                    prompt=theme_prompt,
                    temperature=0.3,
                    max_tokens=800
                )
                
                theme_data = parse_llm_json_response(response_text)
                cluster_themes[cluster_id] = theme_data.get("main_theme", f"Cluster {cluster_id}")
                cluster_summaries[cluster_id] = theme_data.get("detailed_summary", "No summary available")
                
                logger.info(f"✅ Cluster {cluster_id}: {cluster_themes[cluster_id]}")
                
            except Exception as e:
                logger.warning(f"Theme analysis failed for cluster {cluster_id}: {e}")
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
        
        logger.info(f"🔍 Searching for echoes in {len(corpus_texts)} texts (threshold: {threshold})")
        
        echoes = []
        
        for idx, corpus_text in enumerate(corpus_texts, 1):
            logger.info(f"Analyzing text {idx}/{len(corpus_texts)}")
            
            similarity = await self.analyze_similarity(
                query_text, 
                corpus_text["text"],
                "lat",  # Default language
                corpus_text.get("language", "lat")
            )
            
            if similarity.similarity_score >= threshold:
                echoes.append((corpus_text, similarity))
                logger.info(f"✅ Echo found! Score: {similarity.similarity_score:.2f}")
            
            # Add small delay to avoid overwhelming the service
            await asyncio.sleep(0.3)
        
        # Sort by similarity score
        echoes.sort(key=lambda x: x[1].similarity_score, reverse=True)
        
        logger.info(f"✅ Found {len(echoes)} echoes above threshold")
        return echoes
    
    async def batch_similarity_analysis(
        self,
        text_pairs: List[Tuple[str, str]],
        languages: Optional[List[Tuple[str, str]]] = None
    ) -> List[TextSimilarity]:
        """
        Batch analysis of multiple text pairs
        More efficient than calling analyze_similarity repeatedly
        """
        
        if languages is None:
            languages = [("lat", "lat")] * len(text_pairs)
        
        logger.info(f"🔍 Batch analyzing {len(text_pairs)} text pairs")
        
        results = []
        for idx, ((text1, text2), (lang1, lang2)) in enumerate(zip(text_pairs, languages), 1):
            logger.info(f"Analyzing pair {idx}/{len(text_pairs)}")
            
            similarity = await self.analyze_similarity(text1, text2, lang1, lang2)
            results.append(similarity)
            
            # Small delay between requests
            if idx < len(text_pairs):
                await asyncio.sleep(0.5)
        
        logger.info(f"✅ Completed batch analysis of {len(text_pairs)} pairs")
        return results
        
    def get_status(self) -> Dict[str, Any]:
        """Get semantic analyzer status"""
        llm_status = llm_service.get_status()
        
        return {
            "api_configured": llm_status["api_configured"],
            "model_ready": llm_status["initialized"],
            "available_models": llm_status.get("available_models", []),
            "primary_model": llm_status.get("primary_model"),
            "features": [
                "Pairwise text similarity analysis",
                "Multi-text clustering with theme detection",
                "Textual echo detection in corpora",
                "Batch similarity analysis",
                "Multi-model fallback system",
                "Automatic quota handling"
            ],
            "llm_service": llm_status,
            "message": "Semantic Oracle ready with multi-model fallback" if llm_status["initialized"] else "LLM service not configured"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Test semantic analyzer functionality"""
        
        # Initialize if needed
        if not llm_service._initialized:
            await llm_service.initialize()
        
        # Simple test
        test_text1 = "amor vincit omnia"
        test_text2 = "omnia vincit amor"
        
        try:
            result = await self.analyze_similarity(test_text1, test_text2)
            
            return {
                "status": "healthy",
                "test_passed": True,
                "similarity_score": result.similarity_score,
                "llm_service_status": llm_service.get_status()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "test_passed": False,
                "error": str(e),
                "llm_service_status": llm_service.get_status()
            }


# Convenience functions
async def compare_texts(text1: str, text2: str, lang1: str = "lat", lang2: str = "lat") -> TextSimilarity:
    """Convenience function for quick text comparison"""
    analyzer = SemanticAnalyzer()
    return await analyzer.analyze_similarity(text1, text2, lang1, lang2)


async def find_similar_texts(
    query: str, 
    corpus: List[Dict[str, str]], 
    threshold: float = 0.7
) -> List[Tuple[Dict[str, str], TextSimilarity]]:
    """Convenience function for finding similar texts"""
    analyzer = SemanticAnalyzer()
    return await analyzer.find_textual_echoes(query, corpus, threshold)


async def cluster_corpus(
    texts: List[Dict[str, str]], 
    num_clusters: int = 3
) -> ClusterResult:
    """Convenience function for clustering texts"""
    analyzer = SemanticAnalyzer()
    return await analyzer.cluster_texts(texts, num_clusters)
