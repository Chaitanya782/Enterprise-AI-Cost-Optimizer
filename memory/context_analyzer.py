"""Context Analyzer for processing conversation context and relevance"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import re
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import logger


@dataclass
class ContextInsight:
    """Represents an insight derived from context analysis"""
    type: str  # 'pattern', 'preference', 'constraint', 'evolution'
    description: str
    confidence: float
    supporting_queries: List[str]
    relevance_score: float


class ContextAnalyzer:
    """Analyzes conversation context for intelligent recommendations"""

    # Context relevance weights
    RECENCY_WEIGHT = 0.4
    SIMILARITY_WEIGHT = 0.3
    INTENT_CONTINUITY_WEIGHT = 0.2
    METRIC_CONSISTENCY_WEIGHT = 0.1

    # Pattern recognition thresholds
    PATTERN_MIN_OCCURRENCES = 2
    PREFERENCE_CONFIDENCE_THRESHOLD = 0.6
    CONTEXT_RELEVANCE_THRESHOLD = 0.3

    def __init__(self):
        """Initialize context analyzer"""
        self.similarity_cache = {}
        logger.info("ContextAnalyzer initialized")

    def analyze_context_relevance(self, current_query: str,
                                  history: List[Any],
                                  max_context_age_hours: int = 24) -> Dict[str, Any]:
        """Analyze how relevant historical context is to current query"""
        if not history:
            return {
                'relevant_context': [],
                'context_score': 0.0,
                'insights': [],
                'recommended_defaults': {}
            }

        # Filter by time relevance
        cutoff_time = datetime.now() - timedelta(hours=max_context_age_hours)
        recent_history = [ctx for ctx in history if ctx.timestamp >= cutoff_time]

        # Score each historical query for relevance
        context_scores = []
        for ctx in recent_history:
            score = self._calculate_context_relevance(current_query, ctx)
            if score >= self.CONTEXT_RELEVANCE_THRESHOLD:
                context_scores.append((ctx, score))

        # Sort by relevance score
        context_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_context = [ctx for ctx, score in context_scores[:5]]  # Top 5 most relevant

        # Generate insights
        insights = self._generate_context_insights(current_query, recent_history)

        # Calculate overall context score
        overall_score = sum(score for _, score in context_scores) / len(context_scores) if context_scores else 0.0

        # Generate recommended defaults
        recommended_defaults = self._generate_contextual_defaults(current_query, relevant_context)

        return {
            'relevant_context': relevant_context,
            'context_score': overall_score,
            'insights': insights,
            'recommended_defaults': recommended_defaults,
            'context_summary': self._generate_context_summary(relevant_context)
        }

    def _calculate_context_relevance(self, current_query: str, historical_context: Any) -> float:
        """Calculate relevance score between current query and historical context"""
        current_lower = current_query.lower()
        historical_lower = historical_context.query.lower()

        # 1. Recency score (more recent = more relevant)
        hours_ago = (datetime.now() - historical_context.timestamp).total_seconds() / 3600
        recency_score = max(0, 1 - (hours_ago / 24))  # Decay over 24 hours

        # 2. Textual similarity
        similarity_score = self._calculate_text_similarity(current_lower, historical_lower)

        # 3. Intent continuity
        intent_score = 1.0 if self._has_intent_continuity(current_query, historical_context) else 0.0

        # 4. Metric consistency
        metric_score = self._calculate_metric_consistency(current_query, historical_context)

        # Weighted combination
        total_score = (
                recency_score * self.RECENCY_WEIGHT +
                similarity_score * self.SIMILARITY_WEIGHT +
                intent_score * self.INTENT_CONTINUITY_WEIGHT +
                metric_score * self.METRIC_CONSISTENCY_WEIGHT
        )

        return min(1.0, total_score)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # Simple word overlap-based similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        # Boost for exact phrase matches
        phrase_boost = 0.0
        common_phrases = [
            'cost analysis', 'roi calculation', 'task automation', 'workflow optimization',
            'llm comparison', 'provider comparison', 'budget analysis', 'implementation cost'
        ]

        for phrase in common_phrases:
            if phrase in text1 and phrase in text2:
                phrase_boost += 0.2

        return min(1.0, jaccard_similarity + phrase_boost)

    def _has_intent_continuity(self, current_query: str, historical_context: Any) -> bool:
        """Check if there's intent continuity between queries"""
        current_intent = self._extract_primary_intent(current_query)
        historical_intent = historical_context.intent

        # Direct match
        if current_intent == historical_intent:
            return True

        # Related intents
        related_intents = {
            'cost_analysis': ['roi_analysis', 'budget_planning'],
            'roi_analysis': ['cost_analysis', 'benefit_analysis'],
            'task_analysis': ['workflow_optimization', 'automation_planning'],
            'comprehensive': ['cost_analysis', 'roi_analysis', 'task_analysis']
        }

        return historical_intent in related_intents.get(current_intent, [])

    def _extract_primary_intent(self, query: str) -> str:
        """Extract primary intent from query (simplified version)"""
        query_lower = query.lower()

        cost_keywords = ['cost', 'price', 'budget', 'expensive', 'cheap', 'spend']
        roi_keywords = ['roi', 'return', 'benefit', 'value', 'profit', 'payback']
        task_keywords = ['automate', 'task', 'workflow', 'process', 'optimize']

        cost_score = sum(1 for keyword in cost_keywords if keyword in query_lower)
        roi_score = sum(1 for keyword in roi_keywords if keyword in query_lower)
        task_score = sum(1 for keyword in task_keywords if keyword in query_lower)

        if cost_score >= roi_score and cost_score >= task_score:
            return 'cost_analysis'
        elif roi_score >= task_score:
            return 'roi_analysis'
        else:
            return 'task_analysis'

    def _calculate_metric_consistency(self, current_query: str, historical_context: Any) -> float:
        """Calculate consistency of metrics between queries"""
        current_metrics = self._extract_query_metrics(current_query)
        historical_metrics = historical_context.extracted_metrics

        if not current_metrics or not historical_metrics:
            return 0.0

        # Check for overlapping metric types
        current_keys = set(current_metrics.keys())
        historical_keys = set(historical_metrics.keys())

        overlap = current_keys.intersection(historical_keys)
        total_keys = current_keys.union(historical_keys)

        if not total_keys:
            return 0.0

        # Base score from key overlap
        base_score = len(overlap) / len(total_keys)

        # Value consistency bonus
        value_consistency = 0.0
        for key in overlap:
            if isinstance(current_metrics[key], (int, float)) and isinstance(historical_metrics[key], (int, float)):
                ratio = min(current_metrics[key], historical_metrics[key]) / max(current_metrics[key],
                                                                                 historical_metrics[key])
                if ratio > 0.5:  # Values are reasonably similar
                    value_consistency += 0.2

        return min(1.0, base_score + value_consistency)

    def _extract_query_metrics(self, query: str) -> Dict[str, Any]:
        """Extract metrics from query (simplified version)"""
        metrics = {}
        query_lower = query.lower()

        # Budget patterns
        budget_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])?'
        budget_matches = re.findall(budget_pattern, query_lower)
        if budget_matches:
            amount = float(budget_matches[0][0].replace(',', ''))
            multiplier = {'k': 1000, 'm': 1000000, 'b': 1000000000}.get(budget_matches[0][1], 1)
            metrics['budget'] = amount * multiplier

        # Request patterns
        request_pattern = r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?)\s*(?:per\s*day|daily)'
        request_matches = re.findall(request_pattern, query_lower)
        if request_matches:
            metrics['daily_requests'] = float(request_matches[0].replace(',', ''))

        return metrics

    def _generate_context_insights(self, current_query: str, history: List[Any]) -> List[ContextInsight]:
        """Generate insights from context analysis"""
        insights = []

        if len(history) < 2:
            return insights

        # Pattern analysis
        intent_pattern = self._analyze_intent_patterns(history)
        if intent_pattern:
            insights.append(intent_pattern)

        # Budget evolution
        budget_evolution = self._analyze_budget_evolution(history)
        if budget_evolution:
            insights.append(budget_evolution)

        # Provider preferences
        provider_preferences = self._analyze_provider_preferences(history)
        if provider_preferences:
            insights.append(provider_preferences)

        # Use case consistency
        use_case_consistency = self._analyze_use_case_consistency(history)
        if use_case_consistency:
            insights.append(use_case_consistency)

        return insights

    def _analyze_intent_patterns(self, history: List[Any]) -> Optional[ContextInsight]:
        """Analyze intent patterns in conversation history"""
        intents = [ctx.intent for ctx in history]
        intent_counts = defaultdict(int)

        for intent in intents:
            intent_counts[intent] += 1

        if len(intent_counts) == 1:
            dominant_intent = list(intent_counts.keys())[0]
            return ContextInsight(
                type='pattern',
                description=f"Consistent focus on {dominant_intent.replace('_', ' ')}",
                confidence=0.9,
                supporting_queries=[ctx.query[:50] + "..." for ctx in history[:3]],
                relevance_score=0.8
            )

        # Check for evolution pattern
        if len(intents) >= 3:
            recent_intents = intents[-3:]
            if len(set(recent_intents)) == 1 and recent_intents[0] != intents[0]:
                return ContextInsight(
                    type='evolution',
                    description=f"Shifted focus to {recent_intents[0].replace('_', ' ')}",
                    confidence=0.7,
                    supporting_queries=[ctx.query[:50] + "..." for ctx in history[-3:]],
                    relevance_score=0.7
                )

        return None

    def _analyze_budget_evolution(self, history: List[Any]) -> Optional[ContextInsight]:
        """Analyze budget evolution patterns"""
        budget_values = []
        for ctx in history:
            for key in ['budget', 'monthly_spend', 'annual_spend']:
                if key in ctx.extracted_metrics:
                    budget_values.append((ctx.timestamp, ctx.extracted_metrics[key]))

        if len(budget_values) < 2:
            return None

        # Sort by timestamp
        budget_values.sort(key=lambda x: x[0])

        # Check for trend
        values = [v[1] for v in budget_values]
        if len(values) >= 2:
            trend = "increasing" if values[-1] > values[0] else "decreasing"
            change_percent = abs(values[-1] - values[0]) / values[0] * 100

            if change_percent > 20:  # Significant change
                return ContextInsight(
                    type='evolution',
                    description=f"Budget requirements {trend} by {change_percent:.0f}%",
                    confidence=0.8,
                    supporting_queries=[f"${v:,.0f}" for _, v in budget_values],
                    relevance_score=0.9
                )

        return None

    def _analyze_provider_preferences(self, history: List[Any]) -> Optional[ContextInsight]:
        """Analyze provider preferences"""
        all_providers = []
        for ctx in history:
            all_providers.extend(ctx.providers)

        if not all_providers:
            return None

        provider_counts = defaultdict(int)
        for provider in all_providers:
            provider_counts[provider] += 1

        if provider_counts:
            top_provider = max(provider_counts.items(), key=lambda x: x[1])
            if top_provider[1] >= self.PATTERN_MIN_OCCURRENCES:
                return ContextInsight(
                    type='preference',
                    description=f"Strong interest in {top_provider[0]}",
                    confidence=min(0.9, top_provider[1] / len(history)),
                    supporting_queries=[f"Mentioned {top_provider[1]} times"],
                    relevance_score=0.8
                )

        return None

    def _analyze_use_case_consistency(self, history: List[Any]) -> Optional[ContextInsight]:
        """Analyze use case consistency"""
        use_cases = [ctx.use_case for ctx in history if ctx.use_case]

        if not use_cases:
            return None

        use_case_counts = defaultdict(int)
        for use_case in use_cases:
            use_case_counts[use_case] += 1

        consistency_ratio = max(use_case_counts.values()) / len(use_cases)

        if consistency_ratio >= 0.7:  # 70% consistency
            dominant_use_case = max(use_case_counts.items(), key=lambda x: x[1])[0]
            return ContextInsight(
                type='pattern',
                description=f"Consistent focus on {dominant_use_case}",
                confidence=consistency_ratio,
                supporting_queries=[f"Mentioned {use_case_counts[dominant_use_case]} times"],
                relevance_score=0.8
            )

        return None

    def _generate_contextual_defaults(self, current_query: str, relevant_context: List[Any]) -> Dict[str, Any]:
        """Generate intelligent defaults based on relevant context"""
        defaults = {}

        if not relevant_context:
            return defaults

        # Aggregate metrics from relevant context
        metric_aggregations = defaultdict(list)
        for ctx in relevant_context:
            for key, value in ctx.extracted_metrics.items():
                if isinstance(value, (int, float)):
                    metric_aggregations[key].append(value)

        # Calculate averages for numerical metrics
        for key, values in metric_aggregations.items():
            if len(values) >= 2:  # Need at least 2 data points
                defaults[f'suggested_{key}'] = sum(values) / len(values)

        # Most common categorical values
        providers = []
        use_cases = []
        intents = []

        for ctx in relevant_context:
            providers.extend(ctx.providers)
            use_cases.append(ctx.use_case)
            intents.append(ctx.intent)

        if providers:
            provider_counts = defaultdict(int)
            for provider in providers:
                provider_counts[provider] += 1
            defaults['suggested_providers'] = sorted(provider_counts.keys(),
                                                     key=lambda x: provider_counts[x], reverse=True)[:3]

        if use_cases:
            use_case_counts = defaultdict(int)
            for use_case in use_cases:
                if use_case:
                    use_case_counts[use_case] += 1
            if use_case_counts:
                defaults['suggested_use_case'] = max(use_case_counts.items(), key=lambda x: x[1])[0]

        if intents:
            intent_counts = defaultdict(int)
            for intent in intents:
                intent_counts[intent] += 1
            defaults['likely_intent'] = max(intent_counts.items(), key=lambda x: x[1])[0]

        return defaults

    def _generate_context_summary(self, relevant_context: List[Any]) -> str:
        """Generate a human-readable summary of relevant context"""
        if not relevant_context:
            return "No relevant context found."

        context_count = len(relevant_context)

        # Analyze patterns
        intents = [ctx.intent for ctx in relevant_context]
        use_cases = [ctx.use_case for ctx in relevant_context if ctx.use_case]

        dominant_intent = max(set(intents), key=intents.count) if intents else "general"
        dominant_use_case = max(set(use_cases), key=use_cases.count) if use_cases else "various"

        time_span = (relevant_context[0].timestamp - relevant_context[-1].timestamp).total_seconds() / 3600

        summary_parts = [
            f"Found {context_count} relevant queries",
            f"Primary focus: {dominant_intent.replace('_', ' ')}",
            f"Main use case: {dominant_use_case}"
        ]

        if time_span > 1:
            summary_parts.append(f"Over {time_span:.1f} hours")

        return " | ".join(summary_parts)

    def get_context_based_recommendations(self, current_query: str,
                                          context_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on context analysis"""
        recommendations = []

        insights = context_analysis.get('insights', [])
        defaults = context_analysis.get('recommended_defaults', {})
        context_score = context_analysis.get('context_score', 0.0)

        # Context-aware recommendations
        if context_score > 0.7:
            recommendations.append("🧠 **Smart Context**: Building on your previous queries for personalized analysis")

        # Insight-based recommendations
        for insight in insights:
            if insight.type == 'pattern' and insight.confidence > 0.8:
                recommendations.append(f"📈 **Pattern Detected**: {insight.description}")
            elif insight.type == 'evolution' and insight.confidence > 0.7:
                recommendations.append(f"🔄 **Trend Analysis**: {insight.description}")
            elif insight.type == 'preference' and insight.confidence > 0.6:
                recommendations.append(f"⭐ **Preference**: {insight.description}")

        # Default-based recommendations
        if 'suggested_providers' in defaults:
            top_provider = defaults['suggested_providers'][0]
            recommendations.append(f"🎯 **Based on History**: Continue exploring {top_provider}")

        if 'suggested_use_case' in defaults:
            recommendations.append(f"💼 **Use Case Focus**: Optimizing for {defaults['suggested_use_case']}")

        # Budget guidance
        if 'suggested_budget' in defaults:
            budget = defaults['suggested_budget']
            recommendations.append(f"💰 **Budget Context**: Similar to your ${budget:,.0f} previous queries")

        return recommendations[:4]  # Limit to top 4 recommendations