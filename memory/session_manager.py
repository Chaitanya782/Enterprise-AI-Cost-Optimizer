"""Session Manager for handling conversation state and history"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import logger


@dataclass
class QueryContext:
    """Represents a single query context"""
    timestamp: datetime
    query: str
    intent: str
    confidence: float
    use_case: str
    extracted_metrics: Dict[str, Any]
    providers: List[str]
    analysis_results: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryContext':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class UserPreferences:
    """Stores learned user preferences"""
    preferred_providers: List[str]
    typical_use_cases: List[str]
    budget_range: Tuple[float, float]  # (min, max)
    typical_usage_patterns: Dict[str, float]
    priority_factors: List[str]  # e.g., ['cost', 'performance', 'reliability']

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        return cls(**data)


class SessionManager:
    """Manages conversation sessions and context memory"""

    MAX_HISTORY_PER_SESSION = 10
    SESSION_TIMEOUT_HOURS = 24
    MAX_SESSIONS = 100  # Memory management

    def __init__(self):
        """Initialize session manager"""
        self.sessions: Dict[str, List[QueryContext]] = {}
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.locked_use_cases = {}  # session_id -> use_case
        logger.info("SessionManager initialized")

    def lock_use_case(self, session_id: str, use_case: str):
        """Lock the use case for this session"""
        self.locked_use_cases[session_id] = use_case

    def get_locked_use_case(self, session_id: str) -> Optional[str]:
        """Get locked use case for session"""
        return self.locked_use_cases.get(session_id)
    def create_session(self, session_id: str, user_id: Optional[str] = None) -> str:
        """Create a new session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            self.session_metadata[session_id] = {
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'user_id': user_id,
                'query_count': 0
            }
            logger.info(f"Created new session: {session_id}")
        return session_id

    def add_query_context(self, session_id: str, context: QueryContext) -> None:
        """Add query context to session history"""
        if session_id not in self.sessions:
            self.create_session(session_id)

        # Add to history
        self.sessions[session_id].append(context)

        # Maintain history limit
        if len(self.sessions[session_id]) > self.MAX_HISTORY_PER_SESSION:
            self.sessions[session_id] = self.sessions[session_id][-self.MAX_HISTORY_PER_SESSION:]

        # Update metadata
        self.session_metadata[session_id]['last_activity'] = datetime.now()
        self.session_metadata[session_id]['query_count'] += 1

        # Update user preferences
        self._update_user_preferences(session_id, context)

        logger.info(f"Added query context to session {session_id}")

    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[QueryContext]:
        """Get session conversation history"""
        if session_id not in self.sessions:
            return []

        history = self.sessions[session_id]
        if limit:
            return history[-limit:]
        return history

    def get_recent_context(self, session_id: str, minutes: int = 30) -> List[QueryContext]:
        """Get recent context within specified time window"""
        if session_id not in self.sessions:
            return []

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [ctx for ctx in self.sessions[session_id] if ctx.timestamp >= cutoff_time]

    def get_user_preferences(self, session_id: str) -> Optional[UserPreferences]:
        """Get learned user preferences for session"""
        user_id = self.session_metadata.get(session_id, {}).get('user_id')
        if user_id and user_id in self.user_preferences:
            return self.user_preferences[user_id]
        return None

    def get_contextual_defaults(self, session_id: str) -> Dict[str, Any]:
        """Get intelligent defaults based on session history"""
        history = self.get_session_history(session_id, limit=5)
        if not history:
            return {}

        # Aggregate common patterns
        defaults = {}

        # Most common use case
        use_cases = [ctx.use_case for ctx in history if ctx.use_case]
        if use_cases:
            defaults['use_case'] = max(set(use_cases), key=use_cases.count)

        # Most common providers
        all_providers = []
        for ctx in history:
            all_providers.extend(ctx.providers)
        if all_providers:
            provider_counts = defaultdict(int)
            for provider in all_providers:
                provider_counts[provider] += 1
            defaults['preferred_providers'] = sorted(provider_counts.keys(),
                                                     key=lambda x: provider_counts[x], reverse=True)[:3]

        # Average metrics for common patterns
        metrics_aggregated = defaultdict(list)
        for ctx in history:
            for key, value in ctx.extracted_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_aggregated[key].append(value)

        for key, values in metrics_aggregated.items():
            if values:
                defaults[key] = sum(values) / len(values)

        # Most common intent patterns
        intents = [ctx.intent for ctx in history if ctx.intent]
        if intents:
            defaults['typical_intent'] = max(set(intents), key=intents.count)

        return defaults

    def _update_user_preferences(self, session_id: str, context: QueryContext) -> None:
        """Update user preferences based on new context"""
        user_id = self.session_metadata.get(session_id, {}).get('user_id', session_id)

        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreferences(
                preferred_providers=[],
                typical_use_cases=[],
                budget_range=(0.0, 0.0),
                typical_usage_patterns={},
                priority_factors=[]
            )

        prefs = self.user_preferences[user_id]

        # Update preferred providers
        for provider in context.providers:
            if provider not in prefs.preferred_providers:
                prefs.preferred_providers.append(provider)
        prefs.preferred_providers = prefs.preferred_providers[:5]  # Keep top 5

        # Update typical use cases
        if context.use_case and context.use_case not in prefs.typical_use_cases:
            prefs.typical_use_cases.append(context.use_case)
        prefs.typical_use_cases = prefs.typical_use_cases[:3]  # Keep top 3

        # Update budget range
        budget_values = []
        for key in ['budget', 'monthly_spend', 'annual_spend']:
            if key in context.extracted_metrics:
                budget_values.append(context.extracted_metrics[key])

        if budget_values:
            current_min, current_max = prefs.budget_range
            new_min = min(budget_values)
            new_max = max(budget_values)

            if current_min == 0 and current_max == 0:
                prefs.budget_range = (new_min, new_max)
            else:
                prefs.budget_range = (min(current_min, new_min), max(current_max, new_max))

        # Update usage patterns
        for key, value in context.extracted_metrics.items():
            if isinstance(value, (int, float)) and 'requests' in key.lower():
                if key in prefs.typical_usage_patterns:
                    # Running average
                    prefs.typical_usage_patterns[key] = (prefs.typical_usage_patterns[key] + value) / 2
                else:
                    prefs.typical_usage_patterns[key] = value

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        cutoff_time = datetime.now() - timedelta(hours=self.SESSION_TIMEOUT_HOURS)
        expired_sessions = []

        for session_id, metadata in self.session_metadata.items():
            if metadata['last_activity'] < cutoff_time:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]
            del self.session_metadata[session_id]

        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session activity"""
        if session_id not in self.sessions:
            return {}

        history = self.sessions[session_id]
        metadata = self.session_metadata[session_id]

        if not history:
            return {
                'session_id': session_id,
                'query_count': 0,
                'created_at': metadata['created_at'],
                'last_activity': metadata['last_activity']
            }

        # Analyze patterns
        intents = [ctx.intent for ctx in history]
        use_cases = [ctx.use_case for ctx in history]
        providers = []
        for ctx in history:
            providers.extend(ctx.providers)

        return {
            'session_id': session_id,
            'query_count': len(history),
            'created_at': metadata['created_at'],
            'last_activity': metadata['last_activity'],
            'dominant_intent': max(set(intents), key=intents.count) if intents else None,
            'primary_use_case': max(set(use_cases), key=use_cases.count) if use_cases else None,
            'providers_discussed': list(set(providers)),
            'average_confidence': sum(ctx.confidence for ctx in history) / len(history) if history else 0,
            'has_cost_analysis': any('costs' in ctx.analysis_results.get('analysis', {}) for ctx in history),
            'has_roi_analysis': any('roi' in ctx.analysis_results.get('analysis', {}) for ctx in history),
            'has_task_analysis': any('tasks' in ctx.analysis_results.get('analysis', {}) for ctx in history)
        }

    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export session data for analysis or backup"""
        history = self.get_session_history(session_id)
        summary = self.get_session_summary(session_id)
        defaults = self.get_contextual_defaults(session_id)

        return {
            'summary': summary,
            'history': [ctx.to_dict() for ctx in history],
            'contextual_defaults': defaults,
            'user_preferences': self.get_user_preferences(session_id).to_dict() if self.get_user_preferences(
                session_id) else None
        }