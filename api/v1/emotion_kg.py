"""
MEDEA-NEUMOUSA: Emotion Knowledge Graph API
FastAPI endpoints for LLM-powered emotion extraction with sentiment-based visualization
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import asyncio
import json
import time

from core.emotion_kg.extractor import (
    EmotionKGExtractor,
    EmotionGraph,
    EmotionEntity,
    EmotionRelation,
    export_emotion_graph_json,
    create_emotion_analysis_report
)

logger = logging.getLogger("MEDEA.EmotionAPI")
router = APIRouter()

# Initialize emotion extractor
emotion_extractor = EmotionKGExtractor()

class EmotionExtractionRequest(BaseModel):
    text: str = Field(..., description="Text to extract emotions from")
    include_negative_emotions: bool = Field(default=True, description="Include negative emotions like disgust, hatred")
    sentiment_analysis_mode: str = Field(default="comprehensive", description="Mode: basic, comprehensive, deep")
    visualization_style: str = Field(default="sentiment_sized", description="Node sizing: sentiment_sized, intensity_sized, balanced")

class EmotionEntityAPI(BaseModel):
    id: str
    label: str
    emotion_type: str
    intensity: float
    sentiment_score: float
    context: str
    triggers: List[str] = []
    manifestations: List[str] = []
    associated_entities: List[str] = []

class EmotionRelationAPI(BaseModel):
    source: str
    target: str
    relation_type: str
    strength: float
    description: str

class EmotionGraphResponse(BaseModel):
    entities: List[EmotionEntityAPI]
    relations: List[EmotionRelationAPI]
    overall_sentiment: float
    dominant_emotions: List[str]
    sentiment_distribution: Dict[str, float]
    visualization_data: Dict[str, Any]
    analysis_notes: str
    processing_time: float

@router.get("/")
async def emotion_kg_status():
    """Get Emotion Knowledge Graph Extractor status"""
    status = emotion_extractor.get_status()
    
    return {
        "status": "Emotion Knowledge Graph Oracle ready - LLM-powered with sentiment analysis",
        "description": "Extract emotional content from text with sentiment-based node sizing",
        "greek_motto": "Ἡ Μήδεια τὰ πάθη διὰ νοῦ εἰς δίκτυα συνείρει",
        "english_motto": "Medea weaves emotions through intelligence into networks",
        "capabilities": [
            "LLM-powered emotion extraction using Gemini",
            "ALL emotions supported (including disgust, hatred, contempt, rage)",
            "Sentiment scoring (-1.0 to 1.0)",
            "Intensity scoring (0.0 to 1.0)",
            "Node sizing based on emotion intensity and sentiment",
            "Emotion relationship analysis",
            "Interactive visualization data",
            "RDF knowledge graph output"
        ],
        "visualization_features": {
            "node_sizing": "Based on emotion intensity (stronger emotions = bigger nodes)",
            "color_coding": "Based on sentiment score (red=negative, green=positive, yellow=neutral)",
            "border_colors": "Based on emotion type (primary, complex, mood)",
            "edge_styling": "Relationship strength affects line thickness"
        },
        "supported_emotions": {
            "positive": ["joy", "happiness", "love", "trust", "pride", "anticipation"],
            "negative": ["anger", "sadness", "fear", "disgust", "hatred", "contempt", "rage", "fury"],
            "neutral": ["surprise", "curiosity"],
            "complex": ["shame", "guilt", "envy", "jealousy", "nostalgia", "melancholy"]
        },
        "extractor_status": status
    }

@router.post("/extract", response_model=EmotionGraphResponse)
async def extract_emotion_graph(request: EmotionExtractionRequest):
    """
    Extract emotion knowledge graph from text with sentiment-based visualization
    
    Uses LLM to identify ALL emotions including negative ones (disgust, hatred, etc.)
    and creates a knowledge graph with sentiment-based node sizing.
    """
    try:
        start_time = time.time()
        
        # Validate input
        if len(request.text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text too short for emotion analysis")
        
        if len(request.text) > 100000:
            raise HTTPException(status_code=400, detail="Text too long. Maximum 100k characters.")
        
        logger.info(f"Emotion extraction request: {len(request.text)} chars, mode: {request.sentiment_analysis_mode}")
        
        # Extract emotion graph using LLM
        emotion_graph = await emotion_extractor.extract_emotion_graph(request.text)
        
        processing_time = time.time() - start_time
        
        # Convert to API response format
        entities_api = [
            EmotionEntityAPI(
                id=entity.id,
                label=entity.label,
                emotion_type=entity.emotion_type,
                intensity=entity.intensity,
                sentiment_score=entity.sentiment_score,
                context=entity.context,
                triggers=entity.triggers,
                manifestations=entity.manifestations,
                associated_entities=entity.associated_entities
            )
            for entity in emotion_graph.entities
        ]
        
        relations_api = [
            EmotionRelationAPI(
                source=relation.source,
                target=relation.target,
                relation_type=relation.relation_type,
                strength=relation.strength,
                description=relation.description
            )
            for relation in emotion_graph.relations
        ]
        
        # Generate analysis notes
        analysis_notes = f"LLM-powered emotion extraction completed. Found {len(emotion_graph.entities)} emotions with overall sentiment {emotion_graph.overall_sentiment:.2f}. "
        
        if emotion_graph.dominant_emotions:
            analysis_notes += f"Dominant emotions: {', '.join(emotion_graph.dominant_emotions)}. "
        
        negative_emotions = [e for e in emotion_graph.entities if e.sentiment_score < -0.1]
        if negative_emotions:
            analysis_notes += f"Detected {len(negative_emotions)} negative emotions including: {', '.join([e.label for e in negative_emotions[:3]])}."
        
        return EmotionGraphResponse(
            entities=entities_api,
            relations=relations_api,
            overall_sentiment=emotion_graph.overall_sentiment,
            dominant_emotions=emotion_graph.dominant_emotions,
            sentiment_distribution=emotion_graph.sentiment_distribution,
            visualization_data=emotion_graph.visualization_data,
            analysis_notes=analysis_notes,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Emotion extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")

@router.post("/extract-file")
async def extract_emotions_from_file(
    file: UploadFile = File(..., description="Text file to analyze emotions from"),
    export_format: str = Form(default="json", description="Export format: json, report, rdf")
):
    """Extract emotions from uploaded text file"""
    try:
        # Check file type
        if not file.content_type or not file.content_type.startswith('text/'):
            raise HTTPException(status_code=400, detail="Only text files are supported")
        
        # Read and decode file
        content = await file.read()
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = content.decode('latin-1')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file")
        
        # Validate size
        if len(text) > 200000:
            raise HTTPException(status_code=400, detail="File too large. Maximum 200k characters")
        
        if len(text.strip()) < 50:
            raise HTTPException(status_code=400, detail="File content too short for emotion analysis")
        
        # Extract emotions
        emotion_graph = await emotion_extractor.extract_emotion_graph(text)
        
        # Generate output based on format
        if export_format == "json":
            content_data = export_emotion_graph_json(emotion_graph)
            media_type = "application/json"
            file_extension = "json"
        elif export_format == "report":
            content_data = create_emotion_analysis_report(emotion_graph)
            media_type = "text/plain"
            file_extension = "txt"
        elif export_format == "rdf":
            content_data = emotion_extractor.create_rdf_from_emotions(emotion_graph)
            media_type = "text/turtle"
            file_extension = "ttl"
        else:
            raise HTTPException(status_code=400, detail="Invalid export format")
        
        filename = f"emotion_analysis_{file.filename}.{file_extension}"
        
        return Response(
            content=content_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Emotion-Count": str(len(emotion_graph.entities)),
                "X-Overall-Sentiment": str(emotion_graph.overall_sentiment),
                "X-Dominant-Emotions": ",".join(emotion_graph.dominant_emotions)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File emotion extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")

@router.get("/examples")
async def get_emotion_examples():
    """Get example texts for emotion analysis"""
    return {
        "examples": [
            {
                "name": "Mixed Emotions - Love and Hate",
                "text": "I loved her with all my heart, but her betrayal filled me with such rage and disgust that I could barely look at her. The joy we once shared turned to bitter resentment.",
                "expected_emotions": ["love", "rage", "disgust", "joy", "resentment"],
                "expected_sentiment": -0.2
            },
            {
                "name": "Intense Negative Emotions",
                "text": "The corrupt politician's lies filled me with absolute disgust and contempt. I felt sick to my stomach watching him manipulate the crowd with his hatred and fear-mongering.",
                "expected_emotions": ["disgust", "contempt", "hatred", "fear"],
                "expected_sentiment": -0.8
            },
            {
                "name": "Complex Emotional Journey", 
                "text": "At first I was terrified of the presentation, my hands shaking with anxiety. But as I spoke, confidence grew and I felt pride in my achievement. The audience's applause filled me with pure joy.",
                "expected_emotions": ["fear", "anxiety", "confidence", "pride", "joy"],
                "expected_sentiment": 0.4
            }
        ],
        "tips": [
            "Include descriptive emotional language for better detection",
            "Mention physical manifestations (tears, trembling, etc.)",
            "Describe emotional triggers and causes",
            "Use both positive and negative emotions for rich analysis",
            "Longer texts provide more detailed emotional networks"
        ]
    }

@router.get("/health")
async def health_check():
    """Health check for emotion extraction service"""
    status = emotion_extractor.get_status()
    
    return {
        "status": "healthy" if status.get("api_configured") else "configuration_required",
        "api_configured": status.get("api_configured", False),
        "model_ready": status.get("model_ready", False),
        "version": "1.0.0-emotion-kg"
    }