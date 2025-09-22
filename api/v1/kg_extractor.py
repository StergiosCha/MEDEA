"""
MEDEA-NEUMOUSA: Enhanced Knowledge Graph Extractor API
Transform ancient texts into RDF knowledge graphs with structured output and semantic networks
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import asyncio
import json

from core.kg.extractor import (
    MedeaKGExtractor, 
    KGExtractionRequest, 
    KGExtractionMode, 
    KGOutputFormat,
    KGExtractionResult,
    Entity,
    Relation,
    SemanticNetwork,
    export_to_json,
    export_to_csv,
    create_network_report
)

logger = logging.getLogger("MEDEA.KGAPI")
router = APIRouter()

# Initialize KG extractor
kg_extractor = MedeaKGExtractor()

class KGExtractionRequestAPI(BaseModel):
    text: str = Field(..., description="Ancient text to extract knowledge from")
    mode: KGExtractionMode = Field(default=KGExtractionMode.BASIC, description="Extraction mode")
    use_external_sources: bool = Field(default=False, description="Use external knowledge sources")
    output_format: KGOutputFormat = Field(default=KGOutputFormat.TURTLE, description="Output RDF format")
    chunk_size: int = Field(default=15000, ge=1000, le=50000, description="Text chunk size for processing")
    extract_locations: bool = Field(default=True, description="Extract location information")
    extract_temporal: bool = Field(default=True, description="Extract temporal information")
    extract_entities: bool = Field(default=True, description="Extract named entities")
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Confidence threshold")
    include_semantic_network: bool = Field(default=True, description="Include semantic network analysis")
    include_visualization_data: bool = Field(default=True, description="Include visualization data")

class BatchExtractionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to process")
    mode: KGExtractionMode = Field(default=KGExtractionMode.BASIC)
    use_external_sources: bool = Field(default=False)
    output_format: KGOutputFormat = Field(default=KGOutputFormat.TURTLE)

class EntityAPI(BaseModel):
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = {}
    confidence: float = 0.0

class RelationAPI(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0

class SemanticNetworkAPI(BaseModel):
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    clusters: Dict[str, List[str]] = {}
    metrics: Dict[str, Any] = {}



class ValidationRequest(BaseModel):
    rdf_content: str = Field(..., description="RDF content to validate")
    format: KGOutputFormat = Field(default=KGOutputFormat.TURTLE, description="RDF format")

class KGExtractionResponseAPI(BaseModel):
    # Basic extraction results
    rdf_content: str
    format: str
    entity_count: int
    triple_count: int
    confidence_score: float
    extraction_notes: str
    processing_time: float
    chunks_processed: int
    
    # Enhanced structured results
    entities: List[EntityAPI] = []
    relations: List[RelationAPI] = []
    semantic_network: Optional[SemanticNetworkAPI] = None
    summary: Dict[str, Any] = {}
    visualization_data: Dict[str, Any] = {}
    coherence_analysis: Optional[Dict[str, Any]] = None 

@router.get("/")
async def kg_status():
    """Get Knowledge Graph Extractor status"""
    status = kg_extractor.get_status()
    
    return {
        "status": "Enhanced Knowledge Graph Oracle ready to transform texts into structured networks",
        "capabilities": [
            "RDF knowledge graph extraction",
            "Structured entity and relation extraction",
            "Semantic network analysis",
            "Multi-mode processing (Basic, RAG-enhanced, External-enriched)", 
            "Ancient text specialization",
            "Multiple output formats",
            "Large text chunking support",
            "Visualization data generation",
            "Network clustering and metrics"
        ],
        "extraction_modes": {
            "basic": "Standard knowledge extraction from text",
            "rag_enhanced": "Enhanced with entity recognition and analysis",
            "external_enriched": "Enriched with external knowledge sources"
        },
        "output_formats": ["turtle", "rdf_xml", "n3", "json_ld"],
        "export_formats": ["json", "csv", "rdf", "report"],
        "extractor_status": status
    }

# QUICK FIX: Replace your /extract endpoint with this version

@router.post("/extract", response_model=KGExtractionResponseAPI)
async def extract_knowledge_graph(request: KGExtractionRequestAPI):
    """
    Extract RDF knowledge graph from ancient text with fallback handling
    """
    try:
        # Check if extractor is configured
        status = kg_extractor.get_status()
        if not status.get("api_configured", False):
            raise HTTPException(status_code=500, detail="Knowledge Graph Oracle needs Gemini API key")
        
        # Convert API request to internal format
        kg_request = KGExtractionRequest(
            text=request.text,
            mode=request.mode,
            use_external_sources=request.use_external_sources,
            output_format=request.output_format,
            chunk_size=request.chunk_size,
            extract_locations=request.extract_locations,
            extract_temporal=request.extract_temporal,
            extract_entities=request.extract_entities,
            confidence_threshold=request.confidence_threshold
        )
        
        # Validate text length
        if len(request.text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Text too short for meaningful knowledge extraction")
        
        if len(request.text) > 200000:
            raise HTTPException(status_code=400, detail="Text too long. Please use file upload for large texts.")
        
        logger.info(f"Knowledge extraction request: {request.mode.value} mode, {len(request.text)} chars")
        
        # Try extraction with fallback
        result = None
        try:
            # First try: normal extraction
            result = await kg_extractor.extract_knowledge_graph(kg_request)
        except Exception as e:
            logger.warning(f"Normal extraction failed: {e}, trying simple approach")
            try:
                # Second try: simple extraction without complex RDF parsing
                result = await extract_simple_fallback(kg_request)
            except Exception as e2:
                logger.error(f"Both extraction methods failed: {e2}")
                raise HTTPException(status_code=500, detail=f"Knowledge extraction failed: {str(e2)}")
        
        # Convert to API response format
        entities_api = [
            EntityAPI(
                id=entity.id,
                label=entity.label,
                type=entity.type,
                properties=entity.properties,
                confidence=entity.confidence
            )
            for entity in result.entities
        ] if result.entities else []
        
        relations_api = [
            RelationAPI(
                subject=relation.subject,
                predicate=relation.predicate,
                object=relation.object,
                confidence=relation.confidence
            )
            for relation in result.relations
        ] if result.relations else []
        
        semantic_network_api = None
        if request.include_semantic_network and result.semantic_network:
            semantic_network_api = SemanticNetworkAPI(
                nodes=result.semantic_network.nodes,
                edges=result.semantic_network.edges,
                clusters=result.semantic_network.clusters,
                metrics=result.semantic_network.metrics
            )
        
        visualization_data = result.visualization_data if request.include_visualization_data else {}
        
        # Extract coherence analysis from result
        coherence_analysis = getattr(result, 'coherence_analysis', None)
        
        return KGExtractionResponseAPI(
            rdf_content=result.rdf_content,
            format=result.format.value,
            entity_count=result.entity_count,
            triple_count=result.triple_count,
            confidence_score=result.confidence_score,
            extraction_notes=result.extraction_notes,
            processing_time=result.processing_time,
            chunks_processed=result.chunks_processed,
            entities=entities_api,
            relations=relations_api,
            semantic_network=semantic_network_api,
            summary=result.summary,
            visualization_data=visualization_data,
            coherence_analysis=coherence_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph extraction failed: {str(e)}")
async def extract_simple_fallback(request: KGExtractionRequest):
    """Simple fallback extraction that creates basic structured data"""
    start_time = time.time()
    
    # Create minimal entities and relations from text analysis
    text = request.text
    
    # Simple entity extraction using basic patterns
    entities = []
    relations = []
    
    # Extract simple entities using regex patterns
    person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    potential_entities = re.findall(person_pattern, text)
    
    # Filter and create entities
    entity_id = 1
    for potential in set(potential_entities[:10]):  # Limit to 10 entities
        if len(potential) > 2 and potential not in ['The', 'This', 'That', 'And', 'But']:
            entity = Entity(
                id=f"ste:entity_{entity_id}",
                label=potential,
                type="ste:Entity",
                confidence=0.6,
                properties={}
            )
            entities.append(entity)
            entity_id += 1
    
    # Create simple RDF content
    rdf_lines = [
        "@prefix ste: <http://www.example.org/ste#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        ""
    ]
    
    for entity in entities:
        rdf_lines.extend([
            f"{entity.id} a ste:Entity ;",
            f'    rdfs:label "{entity.label}" .',
            ""
        ])
    
    rdf_content = '\n'.join(rdf_lines)
    
    # Create basic semantic network
    semantic_network = SemanticNetwork(
        nodes=[{
            'id': entity.id,
            'label': entity.label,
            'type': entity.type,
            'size': 20,
            'color': '#95a5a6',
            'group': 'Entity'
        } for entity in entities],
        edges=[],
        clusters={'Entity': [e.id for e in entities]},
        metrics={
            'node_count': len(entities),
            'edge_count': 0,
            'density': 0.0
        }
    )
    
    # Create summary
    summary = {
        'total_entities': len(entities),
        'total_relations': len(relations),
        'entity_type_distribution': {'Entity': len(entities)},
        'relation_type_distribution': {},
        'network_density': 0.0,
        'key_entities': [{'label': e.label, 'connections': 0} for e in entities[:3]]
    }
    
    # Create visualization data
    visualization_data = {
        'network': {
            'nodes': semantic_network.nodes,
            'links': []
        }
    }
    
    processing_time = time.time() - start_time
    
    return KGExtractionResult(
        rdf_content=rdf_content,
        format=request.output_format,
        entity_count=len(entities),
        triple_count=len(relations),
        confidence_score=0.6,
        extraction_notes="Simple fallback extraction - basic entity recognition",
        processing_time=processing_time,
        chunks_processed=1,
        entities=entities,
        relations=relations,
        semantic_network=semantic_network,
        summary=summary,
        visualization_data=visualization_data
    )
@router.post("/extract-file")
async def extract_from_file(
    file: UploadFile = File(..., description="Text file to extract knowledge from"),
    mode: KGExtractionMode = Form(default=KGExtractionMode.BASIC),
    use_external_sources: bool = Form(default=False),
    output_format: KGOutputFormat = Form(default=KGOutputFormat.TURTLE),
    chunk_size: int = Form(default=15000),
    extract_locations: bool = Form(default=True),
    extract_temporal: bool = Form(default=True),
    extract_entities: bool = Form(default=True),
    include_semantic_network: bool = Form(default=True),
    export_format: str = Form(default="json", description="Export format: json, csv, rdf, report")
):
    """
    📜 Extract knowledge graph from uploaded text file with multiple export formats
    
    Upload an ancient text file and extract structured RDF knowledge graph with
    semantic network analysis and multiple export format options.
    """
    try:
        # Check file type
        if not file.content_type or not file.content_type.startswith('text/'):
            raise HTTPException(status_code=400, detail="Only text files are supported")
        
        # Read file content
        content = await file.read()
        
        # Decode text
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = content.decode('latin-1')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file. Please ensure it's a valid text file.")
        
        # Validate file size
        if len(text) > 500000:  # 500k character limit for file uploads
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 500k characters")
        
        if len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="File content too short for meaningful extraction")
        
        # Create extraction request
        kg_request = KGExtractionRequest(
            text=text,
            mode=mode,
            use_external_sources=use_external_sources,
            output_format=output_format,
            chunk_size=chunk_size,
            extract_locations=extract_locations,
            extract_temporal=extract_temporal,
            extract_entities=extract_entities
        )
        
        logger.info(f"📜 Enhanced file extraction: {file.filename}, {len(text)} chars, {mode.value} mode")
        
        # Perform extraction
        result = await kg_extractor.extract_knowledge_graph(kg_request)
        
        # Generate output based on export format
        if export_format == "json":
            content_data = export_to_json(result, include_rdf=True)
            media_type = "application/json"
            file_extension = "json"
        elif export_format == "csv":
            entities_csv, relations_csv = export_to_csv(result)
            # For CSV, return entities file (relations would need separate endpoint)
            content_data = entities_csv
            media_type = "text/csv"
            file_extension = "csv"
        elif export_format == "rdf":
            content_data = result.rdf_content
            media_type = "text/turtle"
            file_extension = output_format.value
        elif export_format == "report":
            content_data = create_network_report(result)
            media_type = "text/plain"
            file_extension = "txt"
        else:
            raise HTTPException(status_code=400, detail="Invalid export format. Use: json, csv, rdf, report")
        
        # Return as downloadable file
        filename = f"medea_kg_{file.filename}_{mode.value}.{file_extension}"
        
        return Response(
            content=content_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Entity-Count": str(result.entity_count),
                "X-Triple-Count": str(result.triple_count),
                "X-Confidence-Score": str(result.confidence_score),
                "X-Processing-Time": str(result.processing_time),
                "X-Chunks-Processed": str(result.chunks_processed),
                "X-Network-Density": str(result.semantic_network.metrics.get('density', 0) if result.semantic_network else 0)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💀 Enhanced file extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"File extraction failed: {str(e)}")

@router.post("/extract-batch")
async def extract_batch_texts(request: BatchExtractionRequest):
    """
    🔄 Batch extract knowledge graphs from multiple texts with semantic network analysis
    
    Process multiple ancient texts and return combined structured knowledge graph
    with enhanced semantic network analysis.
    """
    try:
        if len(request.texts) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 texts per batch")
        
        if not request.texts or all(len(text.strip()) < 50 for text in request.texts):
            raise HTTPException(status_code=400, detail="All texts too short for extraction")
        
        # Check extractor status
        status = kg_extractor.get_status()
        if not status.get("api_configured", False):
            raise HTTPException(status_code=500, detail="Knowledge Graph Oracle needs Gemini API key")
        
        all_entities = []
        all_relations = []
        combined_rdf_parts = []
        total_processing_time = 0
        
        for i, text in enumerate(request.texts, 1):
            if len(text.strip()) < 50:
                continue
                
            logger.info(f"🔄 Processing batch text {i}/{len(request.texts)}")
            
            kg_request = KGExtractionRequest(
                text=text,
                mode=request.mode,
                use_external_sources=request.use_external_sources,
                output_format=request.output_format,
                chunk_size=15000,
                extract_locations=True,
                extract_temporal=True,
                extract_entities=True
            )
            
            try:
                result = await kg_extractor.extract_knowledge_graph(kg_request)
                
                # Collect entities and relations
                all_entities.extend(result.entities)
                all_relations.extend(result.relations)
                
                # Collect RDF content (without prefixes)
                rdf_lines = result.rdf_content.split('\n')
                content_lines = [line for line in rdf_lines if not line.strip().startswith('@prefix')]
                combined_rdf_parts.append('\n'.join(content_lines))
                
                total_processing_time += result.processing_time
                
                # Small delay between texts
                if i < len(request.texts):
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Failed to process text {i}: {e}")
                continue
        
        if not all_entities:
            raise HTTPException(status_code=500, detail="No texts could be processed successfully")
        
        # Build combined semantic network
        network_builder = kg_extractor.network_builder
        combined_semantic_network = network_builder.build_network(all_entities, all_relations)
        
        # Create combined RDF
        prefixes = """@prefix ste: <http://www.example.org/ste#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dbp: <http://dbpedia.org/ontology/> .
@prefix geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> .

"""
        
        combined_rdf = prefixes + "\n\n".join(combined_rdf_parts)
        
        # Create combined summary
        summary = kg_extractor._create_summary(all_entities, all_relations, combined_semantic_network)
        
        # Create visualization data
        visualization_data = kg_extractor._create_visualization_data(combined_semantic_network)
        
        return {
            "rdf_content": combined_rdf,
            "format": request.output_format.value,
            "entity_count": len(all_entities),
            "triple_count": len(all_relations),
            "confidence_score": sum(e.confidence for e in all_entities) / len(all_entities) if all_entities else 0,
            "extraction_notes": f"Batch processed {len([t for t in request.texts if len(t.strip()) >= 50])} texts using {request.mode.value} mode",
            "processing_time": total_processing_time,
            "chunks_processed": len(request.texts),
            "entities": [
                {
                    "id": entity.id,
                    "label": entity.label,
                    "type": entity.type,
                    "properties": entity.properties,
                    "confidence": entity.confidence
                }
                for entity in all_entities
            ],
            "relations": [
                {
                    "subject": relation.subject,
                    "predicate": relation.predicate,
                    "object": relation.object,
                    "confidence": relation.confidence
                }
                for relation in all_relations
            ],
            "semantic_network": {
                "nodes": combined_semantic_network.nodes,
                "edges": combined_semantic_network.edges,
                "clusters": combined_semantic_network.clusters,
                "metrics": combined_semantic_network.metrics
            },
            "summary": summary,
            "visualization_data": visualization_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💀 Enhanced batch extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch extraction failed: {str(e)}")

class BatchExtractionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to process")
    mode: KGExtractionMode = Field(default=KGExtractionMode.BASIC)
    use_external_sources: bool = Field(default=False)
    output_format: KGOutputFormat = Field(default=KGOutputFormat.TURTLE)

@router.get("/export/{extraction_id}")
async def export_extraction(
    extraction_id: str,
    format: str = "json",
    include_rdf: bool = True,
    include_network: bool = True
):
    """
    📤 Export previously extracted knowledge graph in various formats
    
    Note: This is a placeholder endpoint. In a real implementation, you would
    store extraction results with IDs and retrieve them here.
    """
    # This would typically retrieve stored results from a database
    return HTTPException(status_code=501, detail="Export endpoint requires result storage implementation")

@router.get("/analyze/{extraction_id}")
async def analyze_semantic_network(extraction_id: str):
    """
    🔍 Perform advanced semantic network analysis on extracted knowledge graph
    
    Returns detailed network metrics, centrality measures, and clustering analysis.
    """
    # This would typically retrieve stored results and perform advanced analysis
    return HTTPException(status_code=501, detail="Analysis endpoint requires result storage implementation")

@router.post("/compare")
async def compare_extractions(extraction_ids: List[str]):
    """
    ⚖️ Compare multiple knowledge graph extractions
    
    Analyze similarities, differences, and overlaps between different text extractions.
    """
    return HTTPException(status_code=501, detail="Comparison endpoint requires result storage implementation")

@router.get("/modes")
async def get_extraction_modes():
    """
    📋 Get available extraction modes and their descriptions
    """
    return {
        "modes": {
            "basic": {
                "name": "Basic Extraction",
                "description": "Standard knowledge extraction from text using linguistic patterns",
                "features": [
                    "Entity recognition", 
                    "Event extraction", 
                    "Relationship mapping",
                    "Basic semantic network analysis"
                ],
                "best_for": "General ancient texts, inscriptions, basic historical documents",
                "output_includes": ["RDF triples", "Entity list", "Relation list", "Basic network"]
            },
            "rag_enhanced": {
                "name": "RAG-Enhanced Extraction", 
                "description": "Enhanced extraction with entity analysis and context awareness",
                "features": [
                    "Advanced entity recognition", 
                    "Context-aware extraction", 
                    "Improved accuracy",
                    "Enhanced semantic clustering"
                ],
                "best_for": "Complex historical narratives, literary texts, detailed chronicles",
                "output_includes": ["RDF triples", "Entity list", "Relation list", "Enhanced network", "Clusters"]
            },
            "external_enriched": {
                "name": "External Knowledge Enriched",
                "description": "Enriched with external knowledge sources (Wikidata, DBpedia)",
                "features": [
                    "External data integration", 
                    "Enhanced entity information", 
                    "Cross-reference validation",
                    "Comprehensive network metrics"
                ],
                "best_for": "Research applications, scholarly analysis, comprehensive knowledge extraction",
                "output_includes": ["RDF triples", "Entity list", "Relation list", "Full network", "External links", "Validation scores"]
            }
        },
        "output_formats": {
            "turtle": "Turtle/TTL format - Human readable RDF",
            "rdf_xml": "RDF/XML format - Standard XML-based RDF",
            "n3": "Notation3 format - Compact RDF notation",
            "json_ld": "JSON-LD format - JSON-based linked data"
        },
        "export_formats": {
            "json": "Complete structured data including entities, relations, and network",
            "csv": "Tabular format for entities and relations",
            "rdf": "Pure RDF content in specified format",
            "report": "Human-readable analysis report"
        }
    }

@router.get("/examples")
async def get_extraction_examples():
    """
    📚 Get example inputs and outputs for enhanced knowledge graph extraction
    """
    return {
        "examples": [
            {
                "name": "Roman Battle - Basic Mode",
                "input": "In the year 31 BC, Augustus defeated Mark Antony at the Battle of Actium. This victory established Augustus as the first Roman Emperor and ended the Roman Republic.",
                "mode": "basic",
                "expected_entities": [
                    {"label": "Augustus", "type": "Person"},
                    {"label": "Mark Antony", "type": "Person"},
                    {"label": "Battle of Actium", "type": "Event"},
                    {"label": "Actium", "type": "Location"},
                    {"label": "Roman Republic", "type": "PoliticalEntity"}
                ],
                "expected_relations": [
                    {"subject": "Augustus", "predicate": "defeated", "object": "Mark Antony"},
                    {"subject": "Battle of Actium", "predicate": "hasAgent", "object": "Augustus"},
                    {"subject": "Battle of Actium", "predicate": "hasTime", "object": "31 BC"},
                    {"subject": "Battle of Actium", "predicate": "hasLocation", "object": "Actium"}
                ],
                "sample_network": {
                    "nodes": 5,
                    "edges": 8,
                    "clusters": {"Person": 2, "Event": 1, "Location": 1, "PoliticalEntity": 1}
                }
            },
            {
                "name": "Greek Philosophy - RAG Enhanced",
                "input": "Socrates taught Plato in Athens. Plato founded the Academy around 387 BC. Aristotle was a student of Plato and later tutored Alexander the Great.",
                "mode": "rag_enhanced",
                "expected_entities": [
                    {"label": "Socrates", "type": "Person"},
                    {"label": "Plato", "type": "Person"},
                    {"label": "Aristotle", "type": "Person"},
                    {"label": "Alexander the Great", "type": "Person"},
                    {"label": "Athens", "type": "Location"},
                    {"label": "Academy", "type": "Institution"}
                ],
                "expected_relations": [
                    {"subject": "Socrates", "predicate": "taught", "object": "Plato"},
                    {"subject": "Plato", "predicate": "founded", "object": "Academy"},
                    {"subject": "Aristotle", "predicate": "studiedUnder", "object": "Plato"},
                    {"subject": "Aristotle", "predicate": "tutored", "object": "Alexander the Great"}
                ],
                "enhanced_features": [
                    "Teacher-student relationship clustering",
                    "Temporal sequence analysis",
                    "Geographic context integration"
                ]
            }
        ],
        "semantic_network_features": {
            "node_analysis": [
                "Size based on connectivity",
                "Color coding by entity type",
                "Confidence scoring",
                "Property enrichment"
            ],
            "edge_analysis": [
                "Relationship type classification",
                "Weight based on confidence",
                "Directional relationships",
                "Temporal ordering"
            ],
            "clustering": [
                "Entity type grouping",
                "Temporal period clustering",
                "Geographic region grouping",
                "Thematic relationship clusters"
            ],
            "metrics": [
                "Network density",
                "Centrality measures",
                "Clustering coefficients",
                "Path length analysis"
            ]
        },
        "visualization_data": {
            "formats": [
                "D3.js force-directed network",
                "Hierarchical tree structure",
                "Clustered node groups",
                "Timeline visualization"
            ],
            "interactive_features": [
                "Node hover details",
                "Edge filtering",
                "Cluster highlighting",
                "Zoom and pan controls"
            ]
        },
        "tips": [
            "Use Basic mode for simple texts and inscriptions",
            "Use RAG-enhanced for complex narratives and literary works", 
            "Use External-enriched when you need comprehensive entity information",
            "Include dates, places, and names for better extraction",
            "Longer texts are automatically chunked for processing",
            "Semantic networks reveal hidden relationships in the text",
            "Export to JSON for programmatic analysis",
            "Export to CSV for spreadsheet analysis",
            "Use visualization data for interactive displays"
        ]
    }

@router.get("/visualization/templates")
async def get_visualization_templates():
    """
    🎨 Get visualization templates and code examples for semantic networks
    """
    return {
        "d3_force_network": {
            "description": "Force-directed graph layout for exploring entity relationships",
            "data_format": "nodes and links arrays",
            "features": ["Interactive dragging", "Zoom and pan", "Node clustering", "Edge filtering"],
            "sample_code": """
// D3.js force simulation example
const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2));
            """
        },
        "hierarchical_tree": {
            "description": "Tree layout for hierarchical entity relationships",
            "data_format": "nested hierarchy object",
            "features": ["Collapsible nodes", "Entity type grouping", "Property display"],
            "sample_code": """
// D3.js tree layout example
const tree = d3.tree().size([height, width]);
const root = d3.hierarchy(hierarchyData);
tree(root);
            """
        },
        "timeline_view": {
            "description": "Temporal visualization of events and relationships",
            "data_format": "time-indexed events array",
            "features": ["Chronological ordering", "Event clustering", "Period highlighting"],
            "sample_code": """
// D3.js timeline example
const xScale = d3.scaleTime()
    .domain(d3.extent(events, d => d.date))
    .range([0, width]);
            """
        },
        "network_metrics": {
            "description": "Statistical dashboard for network analysis",
            "data_format": "metrics object",
            "features": ["Centrality measures", "Clustering stats", "Density plots"],
            "charts": ["Bar charts", "Pie charts", "Line graphs", "Scatter plots"]
        }
    }

@router.post("/validate")
async def validate_extraction_result(request: ValidationRequest):
    """
    ✅ Validate RDF content and analyze semantic network structure
    
    Validates RDF syntax and provides quality metrics for the knowledge graph.
    """
    try:
        # Parse and validate the RDF content
        parser = kg_extractor.rdf_parser
        entities, relations = parser.parse_turtle(request.rdf_content)
        
        if not entities and not relations:
            return {
                "valid": False,
                "errors": ["No valid RDF triples found in content"],
                "suggestions": ["Check RDF syntax", "Ensure proper prefixes are defined"]
            }
        
        # Build semantic network for analysis
        network = kg_extractor.network_builder.build_network(entities, relations)
        
        # Validation metrics
        validation_result = {
            "valid": True,
            "syntax_valid": True,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "network_metrics": network.metrics,
            "quality_score": _calculate_quality_score(entities, relations, network),
            "warnings": [],
            "suggestions": []
        }
        
        # Add warnings and suggestions
        if len(entities) < 3:
            validation_result["warnings"].append("Very few entities extracted - consider longer input text")
        
        if len(relations) < len(entities):
            validation_result["warnings"].append("Few relationships found - entities may be isolated")
        
        if network.metrics.get('density', 0) < 0.1:
            validation_result["suggestions"].append("Low network density - consider more detailed text input")
        
        return validation_result
        
    except Exception as e:
        return {
            "valid": False,
            "syntax_valid": False,
            "errors": [f"RDF parsing failed: {str(e)}"],
            "suggestions": ["Check RDF syntax", "Validate against RDF specification"]
        }

def _calculate_quality_score(entities: List, relations: List, network) -> float:
    """Calculate a quality score for the extracted knowledge graph"""
    score = 0.0
    
    # Entity diversity (0-30 points)
    entity_types = set(e.type for e in entities)
    score += min(30, len(entity_types) * 5)
    
    # Relation density (0-30 points)
    if len(entities) > 0:
        relation_ratio = len(relations) / len(entities)
        score += min(30, relation_ratio * 15)
    
    # Network connectivity (0-40 points)
    if network and network.metrics:
        density = network.metrics.get('density', 0)
        score += density * 40
    
    return min(100, score)

# Health check endpoint
@router.get("/health")
async def health_check():
    """🏥 Health check endpoint for monitoring"""
    status = kg_extractor.get_status()
    
    return {
        "status": "healthy" if status.get("api_configured") else "configuration_required",
        "api_configured": status.get("api_configured", False),
        "model_ready": status.get("model_ready", False),
        "features_available": [
            "RDF extraction",
            "Semantic network analysis", 
            "Multiple export formats",
            "Batch processing",
            "Visualization data generation"
        ],
        "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        "version": "2.0.0-enhanced"
    }