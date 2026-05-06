"""
Base classes for task generation and data sources.

This module defines the abstract interfaces that all task types and data sources
must implement. This allows for pluggable, extensible problem generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class BaseDataSource(ABC):
    """
    Abstract base class for time-varying data sources.

    Data sources provide external data that changes over time (smfr, weather, currencies).
    They support fetching, caching, serialization, and updating.
    """

    @abstractmethod
    def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data from the external source.

        Args:
            params: Parameters for the fetch (e.g., ticker, date range)

        Returns:
            Dictionary containing the fetched data
        """
        pass

    @abstractmethod
    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize data for storage and later updates.

        Args:
            data: The data to serialize

        Returns:
            Serializable dictionary with metadata for updates
        """
        pass

    @abstractmethod
    def update(self, serialized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update previously fetched data with latest values.

        Args:
            serialized_data: Previously serialized data with metadata

        Returns:
            Updated data with same structure but new values
        """
        pass

    @abstractmethod
    def get_source_type(self) -> str:
        """
        Return the type identifier for this data source.

        Returns:
            String identifier (e.g., 'smfr', 'weather', 'currency')
        """
        pass


class BaseTask(ABC):
    """
    Abstract base class for problem generation tasks.

    Tasks generate problems that may include haystack data, specific actions/needles,
    questions, and answers. They can use data sources for time-varying information.
    """

    def __init__(self, config: Dict[str, Any], data_sources: Optional[List[BaseDataSource]] = None):
        """
        Initialize the task with configuration and optional data sources.

        Args:
            config: Configuration dictionary for this task
            data_sources: Optional list of data sources this task depends on
        """
        self.config = config
        self.data_sources = data_sources or []

    @abstractmethod
    def generate_haystack(self, seed: int) -> Dict[str, Any]:
        """
        Generate the haystack/distractor data.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing haystack data and metadata
        """
        pass

    @abstractmethod
    def generate_needles(self, haystack: Dict[str, Any], seed: int, count: int = 1) -> List[Dict[str, Any]]:
        """
        Generate the needle/relevant data points.

        Args:
            haystack: The haystack data to embed needles in
            seed: Random seed for reproducibility
            count: Number of needle instances to generate

        Returns:
            List of needle dictionaries with metadata
        """
        pass

    @abstractmethod
    def compute_answer(self, needle: Dict[str, Any]) -> Any:
        """
        Compute the answer for a single needle instance.

        Args:
            needle: A single needle instance

        Returns:
            The answer (type depends on task)
        """
        pass

    @abstractmethod
    def format_problem(self, haystack: Dict[str, Any], needles: List[Dict[str, Any]],
                      question_template: str, extra_vars: Optional[Dict[str, str]] = None) -> str:
        """
        Format the complete problem text.

        Args:
            haystack: The haystack data
            needles: List of needle instances
            question_template: Template for the question
            extra_vars: Optional extra template variables (e.g., comparison_word)

        Returns:
            Formatted problem string
        """
        pass

    @abstractmethod
    def format_cot(self, needles: List[Dict[str, Any]], answers: List[Any]) -> str:
        """
        Format the chain-of-thought reasoning.

        Args:
            needles: List of needle instances
            answers: List of computed answers

        Returns:
            Formatted chain-of-thought string
        """
        pass

    @abstractmethod
    def get_task_type(self) -> str:
        """
        Return the type identifier for this task.

        Returns:
            String identifier (e.g., 'smfr', 'math', 'coding')
        """
        pass

    def get_data_requirements(self) -> List[str]:
        """
        Return the data source types required by this task.

        Returns:
            List of data source type identifiers
        """
        return [ds.get_source_type() for ds in self.data_sources]
