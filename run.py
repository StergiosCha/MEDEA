#!/usr/bin/env python3
"""
MEDEA-NEUMOUSA Development Server
The Talos Destroyer - Classical Studies AI Platform

"ἡ Μήδεια ἀνίστησι τὰς φωνὰς τῶν νεκρῶν"
"Medea resurrects the voices of the dead"
"""
import uvicorn
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config import settings

def main():
    """Launch MEDEA-NEUMOUSA into the digital realm"""
    
    print("🔮 " + "="*60)
    print("🔮 MEDEA-NEUMOUSA: The Talos Destroyer")
    print("🔮 Classical Studies AI Platform")
    print("🔮 " + "="*60)
    print("🔮")
    print("🔮 Where ancient voices rise and bronze giants fall...")
    print("🔮")
    print(f"🔮 Starting server on http://{settings.host}:{settings.port}")
    print(f"🔮 Environment: {settings.environment}")
    print(f"🔮 Debug mode: {settings.debug}")
    print(f"🔮 Supported languages: {len(settings.supported_languages)}")
    print("🔮")
    print("🔮 Powers available:")
    print("🔮   📖 API Documentation: /docs")
    print("🔮   🔮 Necromancer: /api/v1/necromancer")
    print("🔮   🏛️ Classical Analysis: /api/v1/classical")
    print("🔮   🕸️ Semantic Networks: /api/v1/semantic")
    print("🔮   💓 Health Check: /health")
    print("🔮")
    print("🔮 " + "="*60)
    print()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # Auto-reload for development
        log_level=settings.log_level.lower(),
        access_log=True,
        reload_dirs=[str(project_root)],
        reload_includes=["*.py"],
        app_dir=str(project_root),  # Add this to help find the module
    )

if __name__ == "__main__":
    main()