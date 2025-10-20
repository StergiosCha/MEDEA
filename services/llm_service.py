"""
MEDEA-NEUMOUSA: Shared LLM Service
Centralized LLM access with model fallbacks for all modules
"""
import logging
import os
import asyncio
from typing import List, Tuple, Optional, Dict, Any
import google.generativeai as genai

logger = logging.getLogger("MEDEA.LLMService")


class LLMService:
    """
    Centralized LLM service with model fallbacks
    Used by all modules requiring LLM functionality
    """
    
    def __init__(self):
        self.api_key = os.getenv("MEDEA_GEMINI_API_KEY")
        self.models: List[Tuple[str, Any]] = []
        self._initialized = False
        
        # ✅ OFFICIAL GEMINI MODEL FALLBACK CHAIN (as of Oct 2025)
        self.model_names = [
            # Tier 1: Latest and most capable
            "gemini-2.5-flash",              # Best price-performance, thinking capable
            "gemini-2.5-pro",                # Most advanced, reasoning over complex problems
            
            # Tier 2: Fast and efficient
            "gemini-2.5-flash-lite",         # Fastest, optimized for cost-efficiency
            
            # Tier 3: Second generation workhorses (fallback)
            "gemini-2.0-flash",              # 1M token context, reliable
            "gemini-2.0-flash-lite",         # Smaller, faster 2.0
            
            # Tier 4: First generation (final fallback)
            "gemini-1.5-flash",              # Older but stable
            "gemini-1.5-pro",                # Older pro model
        ]
    
    async def initialize(self) -> None:
        """Initialize all available models"""
        if self._initialized:
            return
            
        if not self.api_key:
            logger.warning("❌ No Gemini API key configured")
            return
        
        try:
            genai.configure(api_key=self.api_key)
            
            # Load available models
            for model_name in self.model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    self.models.append((model_name, model))
                    logger.info(f"✅ Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model {model_name}: {e}")
            
            if not self.models:
                logger.error("❌ No Gemini models could be loaded")
            else:
                logger.info(f"🎯 LLM Service initialized with {len(self.models)} models")
                
            self._initialized = True
            
        except Exception as e:
            logger.error(f"💀 Failed to initialize LLM service: {e}")
            raise
    
    async def generate_completion(
        self, 
        prompt: str, 
        temperature: float = 0.2, 
        max_tokens: int = 1024,
        retry_count: int = 2  # Reduced retries per model since we have multiple models
    ) -> str:
        """
        Generate content with automatic model fallbacks
        
        Args:
            prompt: The input prompt
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum output tokens
            retry_count: Number of retries per model
            
        Returns:
            Generated text content
            
        Raises:
            Exception: If all models fail
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.models:
            raise Exception("No models available for generation")
        
        last_error = None
        
        # Try each model in the fallback chain
        for model_name, model in self.models:
            for attempt in range(retry_count):
                try:
                    logger.debug(f"🔄 Trying {model_name} (attempt {attempt + 1}/{retry_count})")
                    
                    response = await model.generate_content_async(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens
                        )
                    )
                    
                    if response.text and response.text.strip():
                        logger.info(f"✅ Success with {model_name} (attempt {attempt + 1})")
                        return response.text.strip()
                    else:
                        logger.warning(f"⚠️ Empty response from {model_name}")
                        continue
                        
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check for quota/rate limit errors
                    if any(err in error_str for err in ["quota", "429", "rate limit", "resource exhausted"]):
                        logger.warning(f"💀 {model_name} quota exceeded, trying next model...")
                        break  # Skip remaining retries for this model, move to next
                    
                    logger.warning(f"⚠️ {model_name} attempt {attempt + 1} failed: {e}")
                    last_error = e
                    
                    # Wait before retry (exponential backoff)
                    if attempt < retry_count - 1:
                        await asyncio.sleep(1.5 ** attempt)
                    continue
        
        # If all models and retries failed
        raise Exception(f"All {len(self.models)} LLM models failed after {retry_count} attempts each. Last error: {last_error}")
    
    # Alias for compatibility
    async def generate(self, *args, **kwargs) -> str:
        """Alias for generate_completion"""
        return await self.generate_completion(*args, **kwargs)
    
    async def generate_json(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate JSON content with automatic parsing and fallbacks
        
        Args:
            prompt: The input prompt (should request JSON output)
            temperature: Generation temperature (lower for JSON)
            max_tokens: Maximum output tokens
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            Exception: If generation or JSON parsing fails
        """
        response_text = await self.generate_completion(prompt, temperature, max_tokens)
        
        # Clean and parse JSON response
        import re
        import json
        
        # Remove markdown formatting
        cleaned_text = response_text
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text.replace('```', '').strip()
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            cleaned_text = json_match.group(0)
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw response: {response_text[:500]}...")
            raise Exception(f"Could not parse JSON response: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current LLM service status"""
        return {
            "initialized": self._initialized,
            "api_configured": self.api_key is not None,
            "total_models": len(self.models),
            "available_models": [name for name, _ in self.models],
            "primary_model": self.models[0][0] if self.models else None,
            "fallback_chain": self.model_names,
            "model_tiers": {
                "tier_1_latest": ["gemini-2.5-flash", "gemini-2.5-pro"],
                "tier_2_fast": ["gemini-2.5-flash-lite"],
                "tier_3_workhorse": ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
                "tier_4_legacy": ["gemini-1.5-flash", "gemini-1.5-pro"]
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Test all models with a simple prompt"""
        if not self._initialized:
            await self.initialize()
        
        health_results = {}
        test_prompt = "Return exactly: {'test': 'success'}"
        
        for model_name, model in self.models:
            try:
                start_time = asyncio.get_event_loop().time()
                response = await model.generate_content_async(
                    test_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=50
                    )
                )
                end_time = asyncio.get_event_loop().time()
                
                health_results[model_name] = {
                    "status": "healthy",
                    "response_time": round(end_time - start_time, 2),
                    "response_length": len(response.text) if response.text else 0
                }
                
            except Exception as e:
                error_str = str(e).lower()
                if any(err in error_str for err in ["quota", "429", "rate limit"]):
                    health_results[model_name] = {
                        "status": "quota_exceeded",
                        "error": "Quota exhausted"
                    }
                else:
                    health_results[model_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
        
        return health_results


# Global LLM service instance
llm_service = LLMService()
