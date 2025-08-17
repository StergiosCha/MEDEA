"""
MEDEA-NEUMOUSA: Semantic Analysis API
Discover hidden connections in ancient texts
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging

from core.semantic.similarity import SemanticAnalyzer, TextSimilarity, ClusterResult
from services.llm_service import llm_service

logger = logging.getLogger("MEDEA.SemanticAPI")
router = APIRouter()

# Initialize semantic analyzer
semantic_analyzer = SemanticAnalyzer()

class TextInput(BaseModel):
    text: str = Field(..., description="Ancient text content")
    language: str = Field(default="lat", description="Language code")
    title: Optional[str] = Field(default=None, description="Text title or identifier")

class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="First text")
    text2: str = Field(..., description="Second text")
    language1: str = Field(default="lat", description="Language of first text")
    language2: str = Field(default="lat", description="Language of second text")

class ClusterRequest(BaseModel):
    texts: List[TextInput] = Field(..., min_items=2, max_items=10, description="Texts to cluster")
    num_clusters: int = Field(default=3, ge=2, le=5, description="Number of clusters")

class EchoRequest(BaseModel):
    query_text: str = Field(..., description="Text to find echoes for")
    corpus_texts: List[TextInput] = Field(..., min_items=1, max_items=20, description="Corpus to search")
    threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Similarity threshold")

@router.get("/")
async def semantic_status():
    """Get semantic analysis module status"""
    llm_status = llm_service.get_status()
    
    return {
        "status": "Semantic Oracle ready to find hidden connections",
        "capabilities": [
            "Text similarity analysis",
            "Semantic clustering", 
            "Textual echo detection",
            "Thematic analysis"
        ],
        "llm_service": llm_status
    }

@router.post("/similarity")
async def analyze_similarity(request: SimilarityRequest):
    """
    Analyze semantic similarity between two ancient texts
    
    Discovers shared themes, concepts, and linguistic relationships
    """
    try:
        if not llm_service.get_status()["api_configured"]:
            raise HTTPException(status_code=500, detail="Semantic oracle needs Gemini API key")
        
        result = await semantic_analyzer.analyze_similarity(
            request.text1,
            request.text2, 
            request.language1,
            request.language2
        )
        
        return {
            "similarity_analysis": {
                "similarity_score": result.similarity_score,
                "semantic_connections": result.semantic_connections,
                "shared_themes": result.shared_themes,
                "linguistic_relationship": result.linguistic_relationship
            },
            "texts": {
                "text1": result.text1,
                "text2": result.text2
            },
            "interpretation": {
                "strength": "high" if result.similarity_score > 0.8 else "medium" if result.similarity_score > 0.6 else "low",
                "recommendation": get_similarity_recommendation(result.similarity_score)
            }
        }
        
    except Exception as e:
        logger.error(f"Similarity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic analysis failed: {str(e)}")

@router.post("/cluster")
async def cluster_texts(request: ClusterRequest):
    """
    Cluster ancient texts by semantic similarity
    
    Groups texts with similar themes and analyzes cluster characteristics
    """
    try:
        if not semantic_analyzer.models:
            raise HTTPException(status_code=500, detail="Semantic oracle needs Gemini API key")
        
        # Convert to internal format
        texts = [
            {
                "text": t.text,
                "language": t.language,
                "title": t.title or f"Text {i+1}"
            }
            for i, t in enumerate(request.texts)
        ]
        
        result = await semantic_analyzer.cluster_texts(texts, request.num_clusters)
        
        # Format for frontend
        formatted_clusters = []
        for cluster_id, cluster_texts in result.clusters.items():
            formatted_clusters.append({
                "id": int(cluster_id),  # Convert numpy.int64 to int
                "theme": result.cluster_themes.get(cluster_id, f"Cluster {cluster_id}"),
                "summary": result.cluster_summaries.get(cluster_id, ""),
                "texts": cluster_texts,
                "size": len(cluster_texts)
            })
        
        return {
            "clustering_result": {
                "clusters": formatted_clusters,
                "total_clusters": len(result.clusters),
                "outliers": result.outliers
            },
            "analysis": {
                "largest_cluster": max(formatted_clusters, key=lambda x: x["size"]) if formatted_clusters else None,
                "dominant_themes": [cluster["theme"] for cluster in formatted_clusters],
                "cluster_distribution": {str(c["id"]): c["size"] for c in formatted_clusters}
            }
        }
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@router.post("/echoes")
async def find_echoes(request: EchoRequest):
    """
    Find textual echoes in a corpus
    
    Discovers texts that resonate with the query text above a similarity threshold
    """
    try:
        if not semantic_analyzer.models:
            raise HTTPException(status_code=500, detail="Semantic oracle needs Gemini API key")
        
        # Convert to internal format
        corpus_texts = [
            {
                "text": t.text,
                "language": t.language,
                "title": t.title or f"Text {i+1}"
            }
            for i, t in enumerate(request.corpus_texts)
        ]
        
        echoes = await semantic_analyzer.find_textual_echoes(
            request.query_text,
            corpus_texts,
            request.threshold
        )
        
        # Format results
        formatted_echoes = []
        for corpus_text, similarity in echoes:
            formatted_echoes.append({
                "text": corpus_text["text"],
                "title": corpus_text.get("title", "Untitled"),
                "language": corpus_text.get("language", "lat"),
                "similarity_score": similarity.similarity_score,
                "semantic_connections": similarity.semantic_connections,
                "shared_themes": similarity.shared_themes,
                "linguistic_relationship": similarity.linguistic_relationship
            })
        
        return {
            "echo_analysis": {
                "query_text": request.query_text,
                "echoes_found": len(formatted_echoes),
                "threshold_used": request.threshold,
                "echoes": formatted_echoes
            },
            "insights": {
                "strongest_echo": formatted_echoes[0] if formatted_echoes else None,
                "common_themes": get_most_common_themes([e["shared_themes"] for e in formatted_echoes]),
                "relationship_types": get_relationship_distribution([e["linguistic_relationship"] for e in formatted_echoes])
            }
        }
        
    except Exception as e:
        logger.error(f"Echo analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Echo analysis failed: {str(e)}")

def get_similarity_recommendation(score: float) -> str:
    """Get interpretation of similarity score"""
    if score > 0.9:
        return "Texts are virtually identical or directly related"
    elif score > 0.8:
        return "Strong semantic similarity - likely related sources"
    elif score > 0.7:
        return "Notable similarity - potential thematic connections"
    elif score > 0.6:
        return "Moderate similarity - some shared elements"
    else:
        return "Low similarity - distinct texts with little overlap"

def get_most_common_themes(theme_lists: List[List[str]]) -> List[str]:
    """Find most common themes across echo results"""
    from collections import Counter
    all_themes = [theme for themes in theme_lists for theme in themes]
    return [theme for theme, count in Counter(all_themes).most_common(5)]

def get_relationship_distribution(relationships: List[str]) -> Dict[str, int]:
    """Get distribution of relationship types"""
    from collections import Counter
    return dict(Counter(relationships))