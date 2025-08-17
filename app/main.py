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

@app.get("/semantic", response_class=HTMLResponse)
async def semantic_page(request: Request):
    return templates.TemplateResponse("semantic.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "The Sorceress Lives", "necromancer": "Ready to resurrect ancient voices", "semantic": "Ready to find hidden connections"}
