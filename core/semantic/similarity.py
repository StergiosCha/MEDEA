"""
MEDEA-NEUMOUSA: Semantic Similarity Analysis
Find hidden connections between ancient texts
"""
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from services.llm_service import llm_service

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

class SemanticAnalyzer:
    """
    Analyze semantic relationships between ancient texts
    Using centralized LLM service with fallbacks
    """
    
    def __init__(self):
        # Uses shared LLM service
        pass
        
    async def analyze_similarity(
        self, 
        text1: str, 
        text2: str, 
        language1: str = "lat", 
        language2: str = "lat"
    ) -> TextSimilarity:
        """Analyze semantic similarity between two ancient texts"""
        
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
            result = await llm_service.generate_json(prompt)
            
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
        
        # Generate cluster themes using shared LLM service
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
                theme_data = await llm_service.generate_json(theme_prompt)
                cluster_themes[cluster_id] = theme_data.get("main_theme", f"Cluster {cluster_id}")
                cluster_summaries[cluster_id] = theme_data.get("detailed_summary", "No summary available")
                
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
        
        # Sort by similarity score
        echoes.sort(key=lambda x: x[1].similarity_score, reverse=True)
        
        return echoes