from lnn import Model, Predicate, Variable, Implies, And, Or, Not
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