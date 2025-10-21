"""
MEDEA-NEUMOUSA: Narrative Coherence Analyzer
Analyzes extracted knowledge graphs for logical consistency and narrative coherence
Uses LNN (Logical Neural Networks) and LLM-based analysis
"""
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional
import asyncio

# ✅ USE SHARED LLM SERVICE
from services.llm_service import llm_service

# Try to import LNN, but make it optional
try:from lnn import Model, Predicate, Variable, Implies, And, Or, Not
from collections import defaultdict
from typing import List, Dict, Any
import logging

logger = logging.getLogger("MEDEA.CoherenceAnalyzer")

class NarrativeCoherenceLNN:
    def __init__(self):
        self.model = Model()
        self._setup_predicates()
        self._define_coherence_rules()
    
    def _setup_predicates(self):
        """Define predicates for narrative analysis"""
        # Entity types
        self.Person = Predicate('Person')
        self.Place = Predicate('Place')
        self.Emotion = Predicate('Emotion')
        
        # Spatial relations
        self.travels_to = Predicate('travels_to', arity=2)
        self.lives_in = Predicate('lives_in', arity=2)
        self.goes_to = Predicate('goes_to', arity=2)
        self.inhabits = Predicate('inhabits', arity=2)
        
        # Emotional relations
        self.experiences = Predicate('experiences', arity=2)
        self.triggered_by = Predicate('triggered_by', arity=2)
        self.loves = Predicate('loves', arity=2)
        self.hates = Predicate('hates', arity=2)
        self.fears = Predicate('fears', arity=2)
        
        # Communication
        self.speaks_to = Predicate('speaks_to', arity=2)
        self.tells = Predicate('tells', arity=2)
        
        # Temporal/causal
        self.causes = Predicate('causes', arity=2)
        self.leads_to = Predicate('leads_to', arity=2)
        
        # Variables
        self.x = Variable('x')
        self.y = Variable('y')
        self.z = Variable('z')
    
    def _define_coherence_rules(self):
        """Define logical rules for narrative coherence"""
        
        # Rule 1: Emotional consistency - can't love and hate the same entity
        try:
            consistency_rule1 = Not(And(self.loves(self.x, self.y), self.hates(self.x, self.y)))
            
            # Rule 2: Communication requires both parties to be people
            communication_rule = Implies(
                Or(self.speaks_to(self.x, self.y), self.tells(self.x, self.y)),
                And(self.Person(self.x), self.Person(self.y))
            )
            
            # Add rules to model
            self.model.add_knowledge(consistency_rule1, communication_rule)
            
        except Exception as e:
            logger.warning(f"Could not add all LNN rules: {e}")
    
    def analyze_knowledge_graph(self, entities, relations) -> Dict[str, Any]:
        """Analyze extracted knowledge graph for coherence and contradictions"""
        
        try:
            # Load facts into LNN model
            self._load_facts(entities, relations)
            
            # Run inference
            self.model.infer()
            
        except Exception as e:
            logger.warning(f"LNN inference failed: {e}")
        
        # Check for contradictions (using basic logic as fallback)
        contradictions = self._find_contradictions(relations)
        
        # Analyze narrative flow
        narrative_analysis = self._analyze_narrative_flow(entities, relations)
        
        return {
            'contradictions': contradictions,
            'narrative_coherence': narrative_analysis,
            'total_relations': len(relations),
            'overall_coherence_score': self._calculate_overall_coherence(contradictions, len(relations))
        }
    
    def _load_facts(self, entities, relations):
        """Load extracted facts into LNN model"""
        
        # Load entity types
        for entity in entities:
            try:
                if entity.type == 'PER':
                    self.model.add_data({self.Person: {entity.id: True}})
                elif entity.type in ['LOC', 'GPE']:
                    self.model.add_data({self.Place: {entity.id: True}})
                elif 'Affect' in entity.type:
                    self.model.add_data({self.Emotion: {entity.id: True}})
            except Exception as e:
                logger.debug(f"Could not add entity {entity.id}: {e}")
        
        # Load relations
        predicate_map = {
            'travels_to': self.travels_to,
            'lives_in': self.lives_in,
            'goes_to': self.goes_to,
            'inhabits': self.inhabits,
            'experiences': self.experiences,
            'triggered_by': self.triggered_by,
            'loves': self.loves,
            'hates': self.hates,
            'fears': self.fears,
            'speaks_to': self.speaks_to,
            'tells': self.tells,
            'causes': self.causes,
            'leads_to': self.leads_to
        }
        
        for relation in relations:
            try:
                if relation.predicate in predicate_map:
                    pred = predicate_map[relation.predicate]
                    self.model.add_data({pred: {(relation.subject, relation.object): True}})
            except Exception as e:
                logger.debug(f"Could not add relation {relation.predicate}: {e}")
    
    def _find_contradictions(self, relations) -> List[Dict[str, Any]]:
        """Find logical contradictions in the knowledge graph"""
        contradictions = []
        
        # Group relations by subject
        subject_relations = defaultdict(list)
        for relation in relations:
            subject_relations[relation.subject].append(relation)
        
        # Check for contradictions within each subject's relations
        for subject, rels in subject_relations.items():
            
            # Check love/hate contradictions
            loved_entities = {r.object for r in rels if r.predicate == 'loves'}
            hated_entities = {r.object for r in rels if r.predicate == 'hates'}
            love_hate_conflicts = loved_entities & hated_entities
            
            if love_hate_conflicts:
                contradictions.append({
                    'type': 'emotional_contradiction',
                    'subject': subject,
                    'conflicting_objects': list(love_hate_conflicts),
                    'description': f"{subject} both loves and hates the same entities"
                })
            
            # Check spatial contradictions
            residences = {r.object for r in rels if r.predicate in ['lives_in', 'inhabits']}
            if len(residences) > 1:
                contradictions.append({
                    'type': 'spatial_contradiction',
                    'subject': subject,
                    'conflicting_places': list(residences),
                    'description': f"{subject} appears to live in multiple places"
                })
        
        return contradictions
    
    def _analyze_narrative_flow(self, entities, relations) -> Dict[str, Any]:
        """Analyze narrative coherence and flow"""
        
        # Find narrative chains (A causes B, B leads to C, etc.)
        causal_chains = []
        causal_relations = [r for r in relations if r.predicate in ['causes', 'leads_to']]
        
        # Build chains
        for start_relation in causal_relations:
            chain = [start_relation]
            current_object = start_relation.object
            
            # Find continuing relations
            for next_relation in causal_relations:
                if next_relation.subject == current_object and next_relation != start_relation:
                    chain.append(next_relation)
                    current_object = next_relation.object
            
            if len(chain) > 1:
                causal_chains.append(chain)
        
        # Analyze emotional triangulations
        triangulations = self._find_emotional_triangulations(entities, relations)
        
        return {
            'causal_chains': len(causal_chains),
            'longest_chain': max([len(chain) for chain in causal_chains], default=0),
            'emotional_triangulations': len(triangulations),
            'triangulation_details': triangulations
        }
    
    def _find_emotional_triangulations(self, entities, relations) -> List[Dict]:
        """Find Person-Emotion-Place triangulations per your theoretical framework"""
        triangulations = []
        
        emotions = [e for e in entities if 'Affect' in e.type]
        
        for emotion in emotions:
            # Find experiencer
            experiencers = [r.object for r in relations 
                           if r.subject == emotion.id and r.predicate == 'experiences']
            
            # Find trigger
            triggers = [r.object for r in relations 
                       if r.subject == emotion.id and r.predicate == 'triggered_by']
            
            for experiencer in experiencers:
                for trigger in triggers:
                    trigger_entity = next((e for e in entities if e.id == trigger), None)
                    if trigger_entity and trigger_entity.type in ['LOC', 'GPE']:
                        triangulations.append({
                            'person': experiencer,
                            'emotion': emotion.label,
                            'place': trigger,
                            'affect_tier': emotion.type
                        })
        
        return triangulations
    
    def _calculate_overall_coherence(self, contradictions: List, total_relations: int) -> float:
        """Calculate overall narrative coherence score"""
        if total_relations == 0:
            return 0.0
        
        # Base score starts at 1.0
        base_score = 1.0
        
        # Penalty for contradictions
        contradiction_penalty = len(contradictions) * 0.2
        
        return max(0.0, base_score - contradiction_penalty)
# Add this class to the END of your existing coherence_analyzer.py file

class NarrativeCoherenceAnalyzer:
    """
    Main interface for narrative coherence analysis
    Uses the LNN-based analyzer as the backend
    """
    
    def __init__(self):
        self.lnn_analyzer = NarrativeCoherenceLNN()
    
    def analyze_knowledge_graph(self, entities, relations) -> Dict[str, Any]:
        """Analyze knowledge graph for narrative coherence"""
        
        # Use the LNN analyzer
        lnn_result = self.lnn_analyzer.analyze_knowledge_graph(entities, relations)
        
        # Add additional analysis
        dialogue_analysis = self._analyze_dialogue_patterns(relations)
        consistency_scores = self._calculate_consistency_scores(entities, relations)
        
        return {
            'overall_coherence_score': lnn_result['overall_coherence_score'],
            'contradictions': lnn_result['contradictions'],
            'narrative_coherence': {
                **lnn_result['narrative_coherence'],
                'dialogue_interactions': dialogue_analysis['interactions'],
                'unique_speakers': dialogue_analysis['unique_speakers']
            },
            'consistency_scores': consistency_scores
        }
    
    def _analyze_dialogue_patterns(self, relations) -> Dict[str, Any]:
        """Analyze dialogue patterns in the narrative"""
        dialogue_predicates = {'speaks_to', 'tells', 'asks', 'says_to', 'whispers_to', 'shouts_at'}
        
        dialogue_relations = [r for r in relations if r.predicate in dialogue_predicates]
        speakers = set()
        listeners = set()
        
        for relation in dialogue_relations:
            speakers.add(relation.subject)
            listeners.add(relation.object)
        
        return {
            'interactions': len(dialogue_relations),
            'unique_speakers': len(speakers),
            'unique_listeners': len(listeners)
        }
    
    def _calculate_consistency_scores(self, entities, relations) -> Dict[str, float]:
        """Calculate consistency scores for different aspects"""
        
        # Temporal consistency (simplified)
        temporal_score = 1.0  # Default to perfect if no temporal conflicts detected
        
        # Spatial consistency
        spatial_contradictions = [c for c in self.lnn_analyzer._find_contradictions(relations) 
                                if c['type'] == 'spatial_contradiction']
        spatial_score = max(0.0, 1.0 - len(spatial_contradictions) * 0.3)
        
        # Character consistency
        emotional_contradictions = [c for c in self.lnn_analyzer._find_contradictions(relations) 
                                  if c['type'] == 'emotional_contradiction']
        character_score = max(0.0, 1.0 - len(emotional_contradictions) * 0.4)
        
        return {
            'temporal_consistency': temporal_score,
            'spatial_consistency': spatial_score,
            'character_consistency': character_score
        }
    from lnn import Model, Predicate, Variable, Implies, And, Or, Not
    LNN_AVAILABLE = True
except ImportError:
    LNN_AVAILABLE = False
    logging.warning("LNN not available - using LLM-only coherence analysis")

logger = logging.getLogger("MEDEA.CoherenceAnalyzer")


class NarrativeCoherenceLNN:
    """
    Logical Neural Network-based coherence analyzer
    Only used if LNN is available
    """
    
    def __init__(self):
        if not LNN_AVAILABLE:
            logger.warning("LNN not available, LNN-based analysis disabled")
            self.model = None
            return
            
        self.model = Model()
        self._setup_predicates()
        self._define_coherence_rules()
    
    def _setup_predicates(self):
        """Define predicates for narrative analysis"""
        if not LNN_AVAILABLE:
            return
            
        # Entity types
        self.Person = Predicate('Person')
        self.Place = Predicate('Place')
        self.Emotion = Predicate('Emotion')
        
        # Spatial relations
        self.travels_to = Predicate('travels_to', arity=2)
        self.lives_in = Predicate('lives_in', arity=2)
        self.goes_to = Predicate('goes_to', arity=2)
        self.inhabits = Predicate('inhabits', arity=2)
        
        # Emotional relations
        self.experiences = Predicate('experiences', arity=2)
        self.triggered_by = Predicate('triggered_by', arity=2)
        self.loves = Predicate('loves', arity=2)
        self.hates = Predicate('hates', arity=2)
        self.fears = Predicate('fears', arity=2)
        
        # Communication
        self.speaks_to = Predicate('speaks_to', arity=2)
        self.tells = Predicate('tells', arity=2)
        
        # Temporal/causal
        self.causes = Predicate('causes', arity=2)
        self.leads_to = Predicate('leads_to', arity=2)
        
        # Variables
        self.x = Variable('x')
        self.y = Variable('y')
        self.z = Variable('z')
    
    def _define_coherence_rules(self):
        """Define logical rules for narrative coherence"""
        if not LNN_AVAILABLE or not self.model:
            return
            
        try:
            # Rule 1: Emotional consistency - can't love and hate the same entity
            consistency_rule1 = Not(And(self.loves(self.x, self.y), self.hates(self.x, self.y)))
            
            # Rule 2: Communication requires both parties to be people
            communication_rule = Implies(
                Or(self.speaks_to(self.x, self.y), self.tells(self.x, self.y)),
                And(self.Person(self.x), self.Person(self.y))
            )
            
            # Add rules to model
            self.model.add_knowledge(consistency_rule1, communication_rule)
            
        except Exception as e:
            logger.warning(f"Could not add all LNN rules: {e}")
    
    def analyze_knowledge_graph(self, entities, relations) -> Dict[str, Any]:
        """Analyze extracted knowledge graph for coherence and contradictions"""
        
        if LNN_AVAILABLE and self.model:
            try:
                # Load facts into LNN model
                self._load_facts(entities, relations)
                
                # Run inference
                self.model.infer()
                
            except Exception as e:
                logger.warning(f"LNN inference failed: {e}")
        
        # Check for contradictions (using basic logic as fallback)
        contradictions = self._find_contradictions(relations)
        
        # Analyze narrative flow
        narrative_analysis = self._analyze_narrative_flow(entities, relations)
        
        return {
            'contradictions': contradictions,
            'narrative_coherence': narrative_analysis,
            'total_relations': len(relations),
            'overall_coherence_score': self._calculate_overall_coherence(contradictions, len(relations))
        }
    
    def _load_facts(self, entities, relations):
        """Load extracted facts into LNN model"""
        if not LNN_AVAILABLE or not self.model:
            return
            
        # Load entity types
        for entity in entities:
            try:
                if entity.type == 'PER':
                    self.model.add_data({self.Person: {entity.id: True}})
                elif entity.type in ['LOC', 'GPE']:
                    self.model.add_data({self.Place: {entity.id: True}})
                elif 'Affect' in entity.type:
                    self.model.add_data({self.Emotion: {entity.id: True}})
            except Exception as e:
                logger.debug(f"Could not add entity {entity.id}: {e}")
        
        # Load relations
        predicate_map = {
            'travels_to': self.travels_to,
            'lives_in': self.lives_in,
            'goes_to': self.goes_to,
            'inhabits': self.inhabits,
            'experiences': self.experiences,
            'triggered_by': self.triggered_by,
            'loves': self.loves,
            'hates': self.hates,
            'fears': self.fears,
            'speaks_to': self.speaks_to,
            'tells': self.tells,
            'causes': self.causes,
            'leads_to': self.leads_to
        }
        
        for relation in relations:
            try:
                if relation.predicate in predicate_map:
                    pred = predicate_map[relation.predicate]
                    self.model.add_data({pred: {(relation.subject, relation.object): True}})
            except Exception as e:
                logger.debug(f"Could not add relation {relation.predicate}: {e}")
    
    def _find_contradictions(self, relations) -> List[Dict[str, Any]]:
        """Find logical contradictions in the knowledge graph"""
        contradictions = []
        
        # Group relations by subject
        subject_relations = defaultdict(list)
        for relation in relations:
            subject_relations[relation.subject].append(relation)
        
        # Check for contradictions within each subject's relations
        for subject, rels in subject_relations.items():
            
            # Check love/hate contradictions
            loved_entities = {r.object for r in rels if r.predicate == 'loves'}
            hated_entities = {r.object for r in rels if r.predicate == 'hates'}
            love_hate_conflicts = loved_entities & hated_entities
            
            if love_hate_conflicts:
                contradictions.append({
                    'type': 'emotional_contradiction',
                    'subject': subject,
                    'conflicting_objects': list(love_hate_conflicts),
                    'description': f"{subject} both loves and hates the same entities"
                })
            
            # Check spatial contradictions
            residences = {r.object for r in rels if r.predicate in ['lives_in', 'inhabits']}
            if len(residences) > 1:
                contradictions.append({
                    'type': 'spatial_contradiction',
                    'subject': subject,
                    'conflicting_places': list(residences),
                    'description': f"{subject} appears to live in multiple places"
                })
        
        return contradictions
    
    def _analyze_narrative_flow(self, entities, relations) -> Dict[str, Any]:
        """Analyze narrative coherence and flow"""
        
        # Find narrative chains (A causes B, B leads to C, etc.)
        causal_chains = []
        causal_relations = [r for r in relations if r.predicate in ['causes', 'leads_to']]
        
        # Build chains
        for start_relation in causal_relations:
            chain = [start_relation]
            current_object = start_relation.object
            
            # Find continuing relations
            for next_relation in causal_relations:
                if next_relation.subject == current_object and next_relation != start_relation:
                    chain.append(next_relation)
                    current_object = next_relation.object
            
            if len(chain) > 1:
                causal_chains.append(chain)
        
        # Analyze emotional triangulations
        triangulations = self._find_emotional_triangulations(entities, relations)
        
        return {
            'causal_chains': len(causal_chains),
            'longest_chain': max([len(chain) for chain in causal_chains], default=0),
            'emotional_triangulations': len(triangulations),
            'triangulation_details': triangulations
        }
    
    def _find_emotional_triangulations(self, entities, relations) -> List[Dict]:
        """Find Person-Emotion-Place triangulations per your theoretical framework"""
        triangulations = []
        
        emotions = [e for e in entities if 'Affect' in e.type]
        
        for emotion in emotions:
            # Find experiencer
            experiencers = [r.object for r in relations 
                           if r.subject == emotion.id and r.predicate == 'experiences']
            
            # Find trigger
            triggers = [r.object for r in relations 
                       if r.subject == emotion.id and r.predicate == 'triggered_by']
            
            for experiencer in experiencers:
                for trigger in triggers:
                    trigger_entity = next((e for e in entities if e.id == trigger), None)
                    if trigger_entity and trigger_entity.type in ['LOC', 'GPE']:
                        triangulations.append({
                            'person': experiencer,
                            'emotion': emotion.label,
                            'place': trigger,
                            'affect_tier': emotion.type
                        })
        
        return triangulations
    
    def _calculate_overall_coherence(self, contradictions: List, total_relations: int) -> float:
        """Calculate overall narrative coherence score"""
        if total_relations == 0:
            return 0.0
        
        # Base score starts at 1.0
        base_score = 1.0
        
        # Penalty for contradictions
        contradiction_penalty = len(contradictions) * 0.2
        
        return max(0.0, base_score - contradiction_penalty)


class NarrativeCoherenceAnalyzer:
    """
    Main interface for narrative coherence analysis
    Uses both LNN-based analyzer and LLM-enhanced analysis
    """
    
    def __init__(self):
        # Initialize LNN analyzer if available
        if LNN_AVAILABLE:
            self.lnn_analyzer = NarrativeCoherenceLNN()
        else:
            self.lnn_analyzer = None
            logger.info("LNN not available - using LLM-enhanced analysis only")
    
    async def analyze_knowledge_graph_async(self, entities, relations) -> Dict[str, Any]:
        """Async version of analysis (uses LLM)"""
        
        # Start with basic analysis
        base_analysis = self._get_base_analysis(entities, relations)
        
        # Enhance with LLM if we have enough data
        if len(entities) > 5 and len(relations) > 3:
            try:
                llm_analysis = await self._llm_enhanced_analysis(entities, relations)
                base_analysis['llm_enhanced_insights'] = llm_analysis
            except Exception as e:
                logger.warning(f"LLM-enhanced analysis failed: {e}")
                base_analysis['llm_enhanced_insights'] = {"error": str(e)}
        
        return base_analysis
    
    def analyze_knowledge_graph(self, entities, relations) -> Dict[str, Any]:
        """Synchronous version (no LLM enhancement)"""
        return self._get_base_analysis(entities, relations)
    
    def _get_base_analysis(self, entities, relations) -> Dict[str, Any]:
        """Get base analysis using LNN or fallback logic"""
        
        # Use LNN analyzer if available
        if self.lnn_analyzer and LNN_AVAILABLE:
            lnn_result = self.lnn_analyzer.analyze_knowledge_graph(entities, relations)
        else:
            # Fallback to basic analysis
            lnn_result = self._basic_analysis(entities, relations)
        
        # Add additional analysis
        dialogue_analysis = self._analyze_dialogue_patterns(relations)
        consistency_scores = self._calculate_consistency_scores(entities, relations, lnn_result.get('contradictions', []))
        
        return {
            'overall_coherence_score': lnn_result['overall_coherence_score'],
            'contradictions': lnn_result['contradictions'],
            'narrative_coherence': {
                **lnn_result['narrative_coherence'],
                'dialogue_interactions': dialogue_analysis['interactions'],
                'unique_speakers': dialogue_analysis['unique_speakers']
            },
            'consistency_scores': consistency_scores,
            'analysis_method': 'lnn' if (self.lnn_analyzer and LNN_AVAILABLE) else 'fallback'
        }
    
    def _basic_analysis(self, entities, relations) -> Dict[str, Any]:
        """Basic fallback analysis when LNN is not available"""
        contradictions = self._find_basic_contradictions(relations)
        narrative_flow = self._analyze_basic_narrative_flow(entities, relations)
        
        return {
            'contradictions': contradictions,
            'narrative_coherence': narrative_flow,
            'total_relations': len(relations),
            'overall_coherence_score': max(0.0, 1.0 - len(contradictions) * 0.2)
        }
    
    def _find_basic_contradictions(self, relations) -> List[Dict[str, Any]]:
        """Basic contradiction detection without LNN"""
        contradictions = []
        
        # Group by subject
        subject_relations = defaultdict(list)
        for relation in relations:
            subject_relations[relation.subject].append(relation)
        
        for subject, rels in subject_relations.items():
            # Emotional contradictions
            loved = {r.object for r in rels if r.predicate == 'loves'}
            hated = {r.object for r in rels if r.predicate == 'hates'}
            conflicts = loved & hated
            
            if conflicts:
                contradictions.append({
                    'type': 'emotional_contradiction',
                    'subject': subject,
                    'conflicting_objects': list(conflicts),
                    'description': f"{subject} both loves and hates {', '.join(conflicts)}"
                })
        
        return contradictions
    
    def _analyze_basic_narrative_flow(self, entities, relations) -> Dict[str, Any]:
        """Basic narrative flow analysis"""
        causal_rels = [r for r in relations if r.predicate in ['causes', 'leads_to']]
        
        return {
            'causal_chains': len(causal_rels),
            'longest_chain': 1 if causal_rels else 0,
            'emotional_triangulations': 0,
            'triangulation_details': []
        }
    
    async def _llm_enhanced_analysis(self, entities, relations) -> Dict[str, Any]:
        """
        ✅ USE LLM SERVICE FOR ENHANCED NARRATIVE ANALYSIS
        
        Provides deeper insights using AI reasoning
        """
        
        # Initialize LLM service if needed
        if not llm_service._initialized:
            await llm_service.initialize()
        
        # Prepare context for LLM
        entity_summary = self._summarize_entities(entities)
        relation_summary = self._summarize_relations(relations)
        
        prompt = f"""Analyze this narrative knowledge graph for coherence and quality.

ENTITIES ({len(entities)} total):
{entity_summary}

RELATIONS ({len(relations)} total):
{relation_summary}

Analyze for:
1. **Narrative Plausibility**: Do the events/relationships make logical sense?
2. **Character Consistency**: Are character actions/emotions consistent?
3. **Temporal Logic**: Is there a coherent sequence of events?
4. **Missing Elements**: What important narrative elements might be missing?
5. **Strength Assessment**: What are the strongest and weakest parts of this narrative?

Return ONLY valid JSON:
{{
    "plausibility_score": 0.85,
    "plausibility_notes": "Brief assessment of logical coherence",
    "character_consistency": "Assessment of character consistency",
    "temporal_coherence": "Assessment of event sequencing",
    "missing_elements": ["List of potentially missing narrative elements"],
    "narrative_strengths": ["Strong aspects of the narrative"],
    "narrative_weaknesses": ["Weak or problematic aspects"],
    "overall_assessment": "Brief overall narrative quality assessment"
}}"""

        try:
            # ✅ USE SHARED LLM SERVICE
            response_text = await llm_service.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse JSON response
            import json
            import re
            
            # Clean response
            cleaned = response_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Try to find JSON
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            
            analysis = json.loads(cleaned)
            
            logger.info(f"LLM enhanced analysis: plausibility={analysis.get('plausibility_score', 0):.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM enhanced analysis failed: {e}")
            return {
                "error": str(e),
                "plausibility_score": 0.0,
                "overall_assessment": "Analysis failed due to technical error"
            }
    
    def _summarize_entities(self, entities, max_items: int = 20) -> str:
        """Create a summary of entities for LLM analysis"""
        entity_types = defaultdict(list)
        
        for entity in entities[:max_items]:
            entity_type = entity.type.split(':')[-1]
            entity_types[entity_type].append(entity.label)
        
        summary_lines = []
        for entity_type, labels in entity_types.items():
            summary_lines.append(f"- {entity_type}: {', '.join(labels[:10])}")
        
        if len(entities) > max_items:
            summary_lines.append(f"- ... and {len(entities) - max_items} more entities")
        
        return "\n".join(summary_lines)
    
    def _summarize_relations(self, relations, max_items: int = 30) -> str:
        """Create a summary of relations for LLM analysis"""
        relation_lines = []
        
        for relation in relations[:max_items]:
            predicate = relation.predicate.split(':')[-1].replace('_', ' ')
            relation_lines.append(f"- {relation.subject} {predicate} {relation.object}")
        
        if len(relations) > max_items:
            relation_lines.append(f"- ... and {len(relations) - max_items} more relations")
        
        return "\n".join(relation_lines)
    
    def _analyze_dialogue_patterns(self, relations) -> Dict[str, Any]:
        """Analyze dialogue patterns in the narrative"""
        dialogue_predicates = {'speaks_to', 'tells', 'asks', 'says_to', 'whispers_to', 'shouts_at'}
        
        dialogue_relations = [r for r in relations if r.predicate in dialogue_predicates]
        speakers = set()
        listeners = set()
        
        for relation in dialogue_relations:
            speakers.add(relation.subject)
            listeners.add(relation.object)
        
        return {
            'interactions': len(dialogue_relations),
            'unique_speakers': len(speakers),
            'unique_listeners': len(listeners)
        }
    
    def _calculate_consistency_scores(self, entities, relations, contradictions) -> Dict[str, float]:
        """Calculate consistency scores for different aspects"""
        
        # Temporal consistency (simplified)
        temporal_score = 1.0  # Default to perfect if no temporal conflicts detected
        
        # Spatial consistency
        spatial_contradictions = [c for c in contradictions if c['type'] == 'spatial_contradiction']
        spatial_score = max(0.0, 1.0 - len(spatial_contradictions) * 0.3)
        
        # Character consistency
        emotional_contradictions = [c for c in contradictions if c['type'] == 'emotional_contradiction']
        character_score = max(0.0, 1.0 - len(emotional_contradictions) * 0.4)
        
        return {
            'temporal_consistency': temporal_score,
            'spatial_consistency': spatial_score,
            'character_consistency': character_score
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            'lnn_available': LNN_AVAILABLE,
            'llm_service_available': llm_service.api_key is not None,
            'analysis_modes': {
                'basic': 'Always available',
                'lnn_enhanced': 'Available' if LNN_AVAILABLE else 'Not available (LNN not installed)',
                'llm_enhanced': 'Available' if llm_service.api_key else 'Not available (no API key)'
            },
            'capabilities': [
                'Contradiction detection',
                'Narrative flow analysis',
                'Dialogue pattern analysis',
                'Consistency scoring',
                'Emotional triangulation detection'
            ]
        }


# Convenience function for synchronous use
def analyze_narrative(entities, relations) -> Dict[str, Any]:
    """Convenience function for synchronous narrative analysis"""
    analyzer = NarrativeCoherenceAnalyzer()
    return analyzer.analyze_knowledge_graph(entities, relations)


# Async convenience function
async def analyze_narrative_async(entities, relations) -> Dict[str, Any]:
    """Convenience function for async narrative analysis with LLM enhancement"""
    analyzer = NarrativeCoherenceAnalyzer()
    return await analyzer.analyze_knowledge_graph_async(entities, relations)
