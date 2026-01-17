"""In-memory caching for database lookups to improve performance."""

from typing import Callable, Optional

from lexiconweaver.database.models import Project
from lexiconweaver.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseCache:
    """In-memory cache for database queries to reduce lookup overhead.
    
    This cache is particularly useful for:
    - Terms with translations lookups
    - Ignored terms lookups
    - Confirmed terms lookups
    
    The cache is project-scoped and automatically invalidated when terms are modified.
    """

    def __init__(self) -> None:
        """Initialize the cache."""
        # Project-scoped caches
        self._terms_with_translations: dict[int, set[str]] = {}
        self._ignored_terms: dict[int, set[str]] = {}
        self._confirmed_terms: dict[int, set[str]] = {}
        
        self._dirty_flags: dict[int, dict[str, bool]] = {}
        
    def _get_project_id(self, project: Project | None) -> Optional[int]:
        """Get project ID for caching."""
        if project is None:
            return None
        return project.id if hasattr(project, 'id') else None
    
    def _ensure_project_cache(self, project_id: int) -> None:
        """Ensure cache structures exist for a project."""
        if project_id not in self._dirty_flags:
            self._dirty_flags[project_id] = {
                'terms_with_translations': True,
                'ignored_terms': True,
                'confirmed_terms': True,
            }
    
    def get_terms_with_translations(
        self, 
        project: Project | None,
        term_list: list[str],
        fetch_func: Callable[[Project | None, list[str]], set[str]]
    ) -> set[str]:
        """Get terms with translations, using cache if available.
        
        Args:
            project: The project to query
            term_list: List of terms to check
            fetch_func: Function to fetch from database if cache miss
            
        Returns:
            Set of terms that have translations
        """
        project_id = self._get_project_id(project)
        if project_id is None:
            return fetch_func(project, term_list)
        
        self._ensure_project_cache(project_id)
        
        if (self._dirty_flags[project_id]['terms_with_translations'] or 
            project_id not in self._terms_with_translations):
            try:
                from lexiconweaver.database.models import GlossaryTerm
                all_terms = GlossaryTerm.select(GlossaryTerm.source_term).where(
                    GlossaryTerm.project == project
                )
                self._terms_with_translations[project_id] = {
                    term.source_term for term in all_terms
                }
                self._dirty_flags[project_id]['terms_with_translations'] = False
                logger.debug("Cached terms with translations", project_id=project_id, count=len(self._terms_with_translations[project_id]))
            except Exception as e:
                logger.warning("Error caching terms with translations", error=str(e))
                return fetch_func(project, term_list)
        
        cached_terms = self._terms_with_translations[project_id]
        return {term for term in term_list if term in cached_terms}
    
    def get_ignored_terms(
        self,
        project: Project | None,
        fetch_func: Callable[[Project | None], set[str]]
    ) -> set[str]:
        """Get ignored terms, using cache if available.
        
        Args:
            project: The project to query
            fetch_func: Function to fetch from database if cache miss
            
        Returns:
            Set of ignored terms (lowercased)
        """
        project_id = self._get_project_id(project)
        if project_id is None:
            return fetch_func(project)
        
        self._ensure_project_cache(project_id)
        
        if (self._dirty_flags[project_id]['ignored_terms'] or 
            project_id not in self._ignored_terms):
            try:
                from lexiconweaver.database.models import IgnoredTerm
                ignored = IgnoredTerm.select().where(IgnoredTerm.project == project)
                self._ignored_terms[project_id] = {term.term.lower() for term in ignored}
                self._dirty_flags[project_id]['ignored_terms'] = False
                logger.debug("Cached ignored terms", project_id=project_id, count=len(self._ignored_terms[project_id]))
            except Exception as e:
                logger.warning("Error caching ignored terms", error=str(e))
                return fetch_func(project)
        
        return self._ignored_terms[project_id].copy()
    
    def get_confirmed_terms(
        self,
        project: Project | None,
        fetch_func: Callable[[Project | None], set[str]]
    ) -> set[str]:
        """Get confirmed terms, using cache if available.
        
        Args:
            project: The project to query
            fetch_func: Function to fetch from database if cache miss
            
        Returns:
            Set of confirmed source terms
        """
        project_id = self._get_project_id(project)
        if project_id is None:
            return fetch_func(project)
        
        self._ensure_project_cache(project_id)
        
        if (self._dirty_flags[project_id]['confirmed_terms'] or 
            project_id not in self._confirmed_terms):
            try:
                from lexiconweaver.database.models import GlossaryTerm
                terms = GlossaryTerm.select().where(GlossaryTerm.project == project)
                self._confirmed_terms[project_id] = {term.source_term for term in terms}
                self._dirty_flags[project_id]['confirmed_terms'] = False
                logger.debug("Cached confirmed terms", project_id=project_id, count=len(self._confirmed_terms[project_id]))
            except Exception as e:
                logger.warning("Error caching confirmed terms", error=str(e))
                return fetch_func(project)
        
        return self._confirmed_terms[project_id].copy()
    
    def invalidate_glossary_terms(self, project: Project | None) -> None:
        """Invalidate cache when glossary terms are added/updated/deleted."""
        project_id = self._get_project_id(project)
        if project_id is not None and project_id in self._dirty_flags:
            self._dirty_flags[project_id]['terms_with_translations'] = True
            self._dirty_flags[project_id]['confirmed_terms'] = True
            logger.debug("Invalidated glossary terms cache", project_id=project_id)
    
    def invalidate_ignored_terms(self, project: Project | None) -> None:
        """Invalidate cache when ignored terms are added/deleted."""
        project_id = self._get_project_id(project)
        if project_id is not None and project_id in self._dirty_flags:
            self._dirty_flags[project_id]['ignored_terms'] = True
            logger.debug("Invalidated ignored terms cache", project_id=project_id)
    
    def invalidate_all(self, project: Project | None) -> None:
        """Invalidate all caches for a project."""
        project_id = self._get_project_id(project)
        if project_id is not None and project_id in self._dirty_flags:
            for key in self._dirty_flags[project_id]:
                self._dirty_flags[project_id][key] = True
            logger.debug("Invalidated all caches", project_id=project_id)
    
    def clear_project(self, project: Project | None) -> None:
        """Clear all cached data for a project."""
        project_id = self._get_project_id(project)
        if project_id is not None:
            self._terms_with_translations.pop(project_id, None)
            self._ignored_terms.pop(project_id, None)
            self._confirmed_terms.pop(project_id, None)
            self._dirty_flags.pop(project_id, None)
            logger.debug("Cleared all caches for project", project_id=project_id)


# Global cache instance (singleton pattern)
_cache_instance: Optional[DatabaseCache] = None


def get_cache() -> DatabaseCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DatabaseCache()
    return _cache_instance
