from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Debug imports
print("Starting MEDEA-NEUMOUSA...")

print("Attempting to import necromancer...")
try:
    from api.v1 import necromancer
    print("SUCCESS: Necromancer imported")
    print(f"Router exists: {necromancer.router}")
except Exception as e:
    print(f"ERROR: Failed to import necromancer: {e}")
    necromancer = None

print("Attempting to import semantic...")
try:
    from api.v1 import semantic
    print("SUCCESS: Semantic imported")
except Exception as e:
    print(f"ERROR: Failed to import semantic: {e}")
    semantic = None

print("Attempting to import kg_extractor...")
try:
    from api.v1 import kg_extractor
    print("SUCCESS: KG extractor imported")
except Exception as e:
    print(f"ERROR: Failed to import kg_extractor: {e}")
    kg_extractor = None

print("Attempting to import linguistic_distance...")
LINGUISTIC_DISTANCE_AVAILABLE = False
try:
    from api.v1 import linguistic_distance
    LINGUISTIC_DISTANCE_AVAILABLE = True
    print("SUCCESS: Linguistic distance module loaded")
except Exception as e:
    print(f"ERROR: Linguistic distance module not available: {e}")

print("Attempting to import emotion_kg...")
EMOTION_KG_AVAILABLE = False
try:
    from api.v1 import emotion_kg
    EMOTION_KG_AVAILABLE = True
    print("SUCCESS: Emotion KG module loaded")
except Exception as e:
    print(f"ERROR: Failed to import emotion_kg: {e}")

app = FastAPI(title="MEDEA-NEUMOUSA", description="The Robot Destroyer")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include API routers with debug info
print("\nRegistering routers...")

if necromancer:
    print("Registering necromancer router at /necromancer")
    app.include_router(necromancer.router, prefix="/necromancer", tags=["necromancer"])
    print("SUCCESS: Necromancer router registered")
else:
    print("SKIPPING: Necromancer not available")

if semantic:
    print("Registering semantic router at /semantic")
    app.include_router(semantic.router, prefix="/api/v1/semantic", tags=["semantic"])
    print("SUCCESS: Semantic router registered")
else:
    print("SKIPPING: Semantic not available")

if kg_extractor:
    print("Registering KG router at /kg")
    app.include_router(kg_extractor.router, prefix="/api/v1/kg", tags=["knowledge-graph"])
    print("SUCCESS: KG router registered")
else:
    print("SKIPPING: KG extractor not available")

if LINGUISTIC_DISTANCE_AVAILABLE:
    print("Registering linguistic distance router at /linguistic-distance")
    app.include_router(linguistic_distance.router, prefix="/api/v1/linguistic-distance", tags=["linguistic-distance"])
    print("SUCCESS: Linguistic distance router registered")
else:
    print("SKIPPING: Linguistic distance not available")

if EMOTION_KG_AVAILABLE:
    print("Registering emotion KG router at /api/v1/emotion-kg")
    app.include_router(emotion_kg.router, prefix="/api/v1/emotion-kg", tags=["emotion-kg"])
    print("SUCCESS: Emotion KG router registered")
else:
    print("SKIPPING: Emotion KG not available")

print("Router registration complete.\n")

# Pages
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/translate", response_class=HTMLResponse)
async def translate_page(request: Request):
    return templates.TemplateResponse("translate.html", {"request": request})

@app.get("/kg", response_class=HTMLResponse)
async def kg_page(request: Request):
    return templates.TemplateResponse("kg_extractor.html", {"request": request})

@app.get("/semantic", response_class=HTMLResponse)
async def semantic_page(request: Request):
    return templates.TemplateResponse("semantic.html", {"request": request})

@app.get("/linguistic-distance", response_class=HTMLResponse)
async def linguistic_distance_page(request: Request):
    return templates.TemplateResponse("linguistic_distance.html", {"request": request})

@app.get("/emotions", response_class=HTMLResponse)
async def emotions_page(request: Request):
    return templates.TemplateResponse("emotion_kg.html", {"request": request})

@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to see all available routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'Unknown')
            })
    
    return {
        "total_routes": len(routes),
        "routes": routes,
        "module_status": {
            "necromancer": "loaded" if necromancer else "failed",
            "semantic": "loaded" if semantic else "failed", 
            "kg_extractor": "loaded" if kg_extractor else "failed",
            "linguistic_distance": "loaded" if LINGUISTIC_DISTANCE_AVAILABLE else "failed",
            "emotion_kg": "loaded" if EMOTION_KG_AVAILABLE else "failed"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "The Sorceress Lives",
        "necromancer": "Ready" if necromancer else "Unavailable",
        "semantic": "Ready" if semantic else "Unavailable",
        "kg": "Ready" if kg_extractor else "Unavailable",
        "linguistic_distance": "Ready" if LINGUISTIC_DISTANCE_AVAILABLE else "Unavailable",
        "emotion_kg": "Ready" if EMOTION_KG_AVAILABLE else "Unavailable",
    }

@app.get("/terms", response_class=HTMLResponse)
async def terms_page():
    """Terms of Use and Attribution page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Terms of Use - MEDEA-NEUMOUSA</title>
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Crimson Text', serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2rem; background: linear-gradient(135deg, #0f1419 0%, #1a1f3a 100%); color: #e6e6e6; }
            h1, h2, h3 { font-family: 'Cinzel', serif; color: #d4af37; }
            .back-link { display: inline-block; margin-bottom: 2rem; color: #d4af37; text-decoration: none; border: 1px solid #d4af37; padding: 0.5rem 1rem; border-radius: 4px; transition: all 0.3s ease; }
            .back-link:hover { background: #d4af37; color: #1a1f3a; }
            code { background: rgba(212, 175, 55, 0.1); padding: 0.2rem 0.4rem; border-radius: 3px; color: #d4af37; font-family: monospace; }
            .citation-box { background: rgba(212, 175, 55, 0.1); border: 1px solid #d4af37; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← Back to MEDEA-NEUMOUSA</a>
        <h1>MEDEA-NEUMOUSA Terms of Use</h1>
        <p><em>"Ἡ Μήδεια τῶν νεκρῶν φωνὰς ἀνίστησι"</em><br><em>"Medea resurrects the voices of the dead"</em></p>
        <hr style="border-color: #d4af37; margin: 2rem 0;">
        <h2>Creator</h2><p><strong>MEDEA-NEUMOUSA</strong> is created by <strong>Stergios Chatzikyriakidis</strong>.</p>
        <h2>FREE TO USE</h2>
        <ul><li>Personal use</li><li>Academic research</li><li>Educational purposes</li><li>Commercial applications</li><li>Open source projects</li></ul>
        <h2>CITATION REQUIRED</h2>
        <div class="citation-box"><strong>Required Citation:</strong><br>
            <code>MEDEA-NEUMOUSA: Ancient Language Translation System<br>Created by Stergios Chatzikyriakidis</code>
        </div>
        <h2>Translation Accuracy Disclaimer</h2>
        <ul><li>Translations are computational approximations</li><li>Results should be verified by scholars for critical use</li><li>Ancient languages have limited digital training data</li><li>Some language pairs may have lower confidence scores</li><li>The system is a research tool, not a definitive authority</li></ul>
        <hr style="border-color: #d4af37; margin: 2rem 0;">
        <p style="text-align: center; font-style: italic; color: #b8860b;">"Where ancient voices rise and bronze giants fall."</p>
        <p style="text-align: center;"><strong>© 2025 Stergios Chatzikyriakidis. MEDEA-NEUMOUSA - Free to use with attribution.</strong></p>
    </body></html>
    """
