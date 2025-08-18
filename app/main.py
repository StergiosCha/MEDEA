from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from api.v1 import necromancer, semantic


app = FastAPI(title="MEDEA-NEUMOUSA", description="The Robot Destroyer")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include API routers
app.include_router(
    necromancer.router,
    prefix="/api/v1/necromancer",
    tags=["necromancer"]
)

app.include_router(
    semantic.router,
    prefix="/api/v1/semantic",
    tags=["semantic"]
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/translate", response_class=HTMLResponse)
async def translate_page(request: Request):
    return templates.TemplateResponse("translate.html", {"request": request})




# Or if you prefer a simple HTML response without a template:
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
            body {
                font-family: 'Crimson Text', serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
                background: linear-gradient(135deg, #0f1419 0%, #1a1f3a 100%);
                color: #e6e6e6;
            }
            h1, h2, h3 {
                font-family: 'Cinzel', serif;
                color: #d4af37;
            }
            .back-link {
                display: inline-block;
                margin-bottom: 2rem;
                color: #d4af37;
                text-decoration: none;
                border: 1px solid #d4af37;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                transition: all 0.3s ease;
            }
            .back-link:hover {
                background: #d4af37;
                color: #1a1f3a;
            }
            code {
                background: rgba(212, 175, 55, 0.1);
                padding: 0.2rem 0.4rem;
                border-radius: 3px;
                color: #d4af37;
                font-family: monospace;
            }
            .citation-box {
                background: rgba(212, 175, 55, 0.1);
                border: 1px solid #d4af37;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← Back to MEDEA-NEUMOUSA</a>
        
        <h1>🔮 MEDEA-NEUMOUSA Terms of Use</h1>
        
        <p><em>"Ἡ Μήδεια τῶν νεκρῶν φωνὰς ἀνίστησι"</em><br>
        <em>"Medea resurrects the voices of the dead"</em></p>
        
        <hr style="border-color: #d4af37; margin: 2rem 0;">
        
        <h2>Creator</h2>
        <p><strong>MEDEA-NEUMOUSA</strong> is created by <strong>Stergios Chatzikyriakidis</strong>.</p>
        
        <h2>✅ FREE TO USE</h2>
        <p>MEDEA-NEUMOUSA is freely available for:</p>
        <ul>
            <li>Personal use</li>
            <li>Academic research</li>
            <li>Educational purposes</li>
            <li>Commercial applications</li>
            <li>Open source projects</li>
        </ul>
        
        <h2>📝 CITATION REQUIRED</h2>
        <p>All use of MEDEA-NEUMOUSA must include proper attribution:</p>
        
        <div class="citation-box">
            <strong>Required Citation:</strong><br>
            <code>MEDEA-NEUMOUSA: Ancient Language Translation System<br>
            Created by Stergios Chatzikyriakidis</code>
        </div>
        
        <h3>Academic Citation (APA Style):</h3>
        <div class="citation-box">
            <code>Chatzikyriakidis, S. (2025). MEDEA-NEUMOUSA: AI-powered ancient language translation and semantic analysis system.</code>
        </div>
        
        <h3>Academic Citation (MLA Style):</h3>
        <div class="citation-box">
            <code>Chatzikyriakidis, Stergios. "MEDEA-NEUMOUSA: Ancient Language Translation System." Web. 2025.</code>
        </div>
        
        <h2>What You Can Do</h2>
        <ul>
            <li>✅ Translate ancient texts</li>
            <li>✅ Use translations in your work</li>
            <li>✅ Build upon the results</li>
            <li>✅ Share outputs with attribution</li>
            <li>✅ Integrate into educational curricula</li>
            <li>✅ Use for commercial projects</li>
            <li>✅ Create derivative applications</li>
        </ul>
        
        <h2>What You Must Do</h2>
        <ul>
            <li>📝 Always cite Stergios Chatzikyriakidis as creator</li>
            <li>📝 Include MEDEA-NEUMOUSA attribution</li>
            <li>📝 Maintain attribution in derivative works</li>
        </ul>
        
        <h2>What You Cannot Do</h2>
        <ul>
            <li>❌ Claim creation or ownership of MEDEA-NEUMOUSA</li>
            <li>❌ Remove or obscure attribution requirements</li>
            <li>❌ Misrepresent the source or creator</li>
        </ul>
        
        <h2>Translation Accuracy Disclaimer</h2>
        <p>MEDEA-NEUMOUSA uses AI to translate between ancient languages. While the system strives for accuracy:</p>
        <ul>
            <li>Translations are computational approximations</li>
            <li>Results should be verified by scholars for critical use</li>
            <li>Ancient languages have limited digital training data</li>
            <li>Some language pairs may have lower confidence scores</li>
            <li>The system is a research tool, not a definitive authority</li>
        </ul>
        
        <hr style="border-color: #d4af37; margin: 2rem 0;">
        
        <p style="text-align: center; font-style: italic; color: #b8860b;">
            "Where ancient voices rise and bronze giants fall."
        </p>
        
        <p style="text-align: center;">
            <strong>© 2025 Stergios Chatzikyriakidis. MEDEA-NEUMOUSA - Free to use with attribution.</strong>
        </p>
    </body>
    </html>
    """

@app.get("/semantic", response_class=HTMLResponse)
async def semantic_page(request: Request):
    return templates.TemplateResponse("semantic.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "The Sorceress Lives", "necromancer": "Ready to resurrect ancient voices", "semantic": "Ready to find hidden connections"}
