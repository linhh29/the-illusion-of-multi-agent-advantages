"""
Configuration schemas for problem generation.

This module defines configuration dataclasses and schemas for customizing
all aspects of problem generation, from task parameters to composition rules.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal
from enum import Enum


class AggregationOp(Enum):
    """Aggregation operations for combining multiple instances."""
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    AVG = "avg"
    COMPARE_GT = "gt"  # Greater than comparison
    COMPARE_LT = "lt"  # Less than comparison
    COMPARE_EQ = "eq"  # Equals comparison
    FIRST = "first"    # Just return first (no aggregation)
    LAST = "last"      # Just return last (no aggregation)
    EARLIEST_DATE = "earliest_date"  # For reverse questions: who can achieve target first
    LATEST_DATE = "latest_date"      # For reverse questions: who takes longest


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    source_type: str  # 'smfr', 'currency', 'weather'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a single task type."""
    task_type: str  # 'smfr', 'math', 'coding'

    # Random seed for reproducibility
    seed: int = 100

    # Number of distinct entities in haystack (e.g., number of smfr)
    breadth: int = 3

    # Depth of operations per entity (e.g., number of transactions)
    depth: int = 5

    # Data source configurations
    data_sources: List[DataSourceConfig] = field(default_factory=list)

    # Task-specific parameters (extensible)
    task_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionConfig:
    """Configuration for composing multiple task instances."""

    # Number of parallel instances (e.g., 3 investors)
    num_instances: int = 1

    # How to aggregate results from multiple instances
    aggregation_op: AggregationOp = AggregationOp.FIRST

    # Question template with placeholders
    question_template: Optional[str] = None

    # Entity names for instances (e.g., ['Alice', 'Bob', 'Charlie'])
    instance_names: Optional[List[str]] = None


@dataclass
class PipelineConfig:
    """Configuration for a complete problem generation pipeline."""

    # Task configuration
    task: TaskConfig

    # Composition configuration
    composition: CompositionConfig

    # Output format options
    include_cot: bool = True
    include_metadata: bool = True

    # Optional custom templates
    templates: Dict[str, str] = field(default_factory=dict)


# Predefined configurations for common use cases

def smfr_single_investor_config(**kwargs) -> PipelineConfig:
    """Configuration for single investor smfr problem."""
    defaults = {
        'task_type': 'smfr',
        'seed': 100,
        'breadth': 3,
        'depth': 5,
        'task_params': {
            'price_type': 'Open',
            'actions': {
                'buy': ['bought', 'acquired', 'purchased'],
                'sell': ['sold', 'disposed']
            }
        }
    }
    defaults.update(kwargs)

    return PipelineConfig(
        task=TaskConfig(**defaults),
        composition=CompositionConfig(
            num_instances=1,
            aggregation_op=AggregationOp.FIRST
        )
    )


def smfr_comparison_config(num_investors: int = 2, **kwargs) -> PipelineConfig:
    """Configuration for comparing multiple investors."""
    defaults = {
        'task_type': 'smfr',
        'seed': 100,
        'breadth': 3,
        'depth': 5,
        'task_params': {
            'price_type': 'Open',
            'actions': {
                'buy': ['bought', 'acquired', 'purchased'],
                'sell': ['sold', 'disposed']
            }
        }
    }
    defaults.update(kwargs)

    investor_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward',
                     'Fiona', 'George', 'Helen', 'Isaac', 'Julia']

    return PipelineConfig(
        task=TaskConfig(**defaults),
        composition=CompositionConfig(
            num_instances=num_investors,
            aggregation_op=AggregationOp.MAX,
            instance_names=investor_names[:num_investors]
        )
    )


def smfr_comparative_target_config(num_investors: int = 2,
                                    question_type: str = 'reverse_target_sell',
                                    aggregation: str = 'earliest',
                                    **kwargs) -> PipelineConfig:
    """
    Configuration for comparing multiple investors on target achievement.

    Args:
        num_investors: Number of investors to compare
        question_type: 'reverse_target_sell' or 'reverse_target_buy'
        aggregation: 'earliest' (fastest) or 'latest' (slowest)
        **kwargs: Additional parameters including target_percentage or target_amount
    """
    # Extract task_params from kwargs if provided
    task_params = kwargs.pop('task_params', {})

    defaults = {
        'task_type': 'smfr',
        'seed': 100,
        'breadth': 2,
        'depth': 2,  # Must be even for pairing
        'task_params': {
            'price_type': 'Open',
            'question_type': question_type,
            'target_percentage': 5.0  # Default 5% target (reduced to minimize None occurrences)
        }
    }

    # Merge task_params
    defaults['task_params'].update(task_params)

    # Update other top-level parameters
    defaults.update(kwargs)

    investor_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward',
                     'Fiona', 'George', 'Helen', 'Isaac', 'Julia']

    agg_op = AggregationOp.EARLIEST_DATE if aggregation == 'earliest' else AggregationOp.LATEST_DATE

    return PipelineConfig(
        task=TaskConfig(**defaults),
        composition=CompositionConfig(
            num_instances=num_investors,
            aggregation_op=agg_op,
            instance_names=investor_names[:num_investors]
        )
    )


def math_single_config(**kwargs) -> PipelineConfig:
    """Configuration for single math problem."""
    defaults = {
        'task_type': 'math',
        'seed': 100,
        'task_params': {
            'problem_type': 'arithmetic',  # 'arithmetic', 'algebra', 'word_problem'
            'difficulty': 'medium',
            'operations': ['+', '-', '*', '/']
        }
    }
    defaults.update(kwargs)

    return PipelineConfig(
        task=TaskConfig(**defaults),
        composition=CompositionConfig(
            num_instances=1,
            aggregation_op=AggregationOp.FIRST
        )
    )


def mixed_task_config(tasks: List[Dict[str, Any]]) -> List[PipelineConfig]:
    """
    Configuration for mixing different task types.

    Args:
        tasks: List of task specifications, each containing task_type and params

    Returns:
        List of pipeline configs that can be combined
    """
    configs = []
    for task_spec in tasks:
        task_type = task_spec['task_type']
        params = task_spec.get('params', {})

        if task_type == 'smfr':
            config = smfr_single_investor_config(**params)
        elif task_type == 'math':
            config = math_single_config(**params)
        else:
            # Generic config
            config = PipelineConfig(
                task=TaskConfig(task_type=task_type, **params),
                composition=CompositionConfig()
            )

        configs.append(config)

    return configs


# Template library
DEFAULT_TEMPLATES = {
    'smfr_single': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{actions}

How much money has {entity} spent on buying shares if all purchases were made at {price_type} prices?
""",

    'smfr_single_profit_loss': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{actions}

What is {entity}'s total profit or loss from these transactions if all trades were made at {price_type} prices?
""",

    'smfr_single_reverse_sell': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{actions}

{entity} has completed several transactions and holds shares in one remaining smfr that needs to be sold. On which dates could {entity} sell these remaining shares to achieve at least {target_description} overall portfolio profit, if all transactions were made at {price_type} prices?
""",

    'smfr_single_reverse_buy': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{actions}

{entity} has completed several transactions and has already sold shares in one smfr but has not yet bought them. On which dates could {entity} have bought these shares to achieve at least {target_description} overall portfolio profit when they were sold, if all transactions were made at {price_type} prices?
""",

    'smfr_compare_two': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{actions_0}

{actions_1}

Who spent {comparison_word} money buying shares if all purchases were made at {price_type} prices?
""",

    'smfr_compare_two_reverse_sell': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{all_actions}

Each investor has completed several transactions and holds shares in one remaining common smfr. Based on when they could sell these remaining shares to achieve at least {target_description} overall portfolio profit, who has the {timing_word} possible sell date (the {timing_word} date in their list of valid dates) to reach this target, if all transactions were made at {price_type} prices?
""",

    'smfr_compare_two_reverse_buy': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{all_actions}

Each investor has completed several transactions and has already sold shares in one common smfr but has not yet bought them. Based on when they would have had to buy these shares to achieve at least {target_description} overall portfolio profit when they sold, who has the {timing_word} possible buy date (the {timing_word} date in their list of valid dates) to reach this target, if all transactions were made at {price_type} prices?
""",

    'smfr_compare_multi': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{all_actions}

Who spent the {superlative_word} money buying shares if all purchases were made at {price_type} prices?
""",

    'smfr_compare_multi_reverse_sell': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{all_actions}

Each investor has completed several transactions and holds shares in one remaining common smfr. Based on when they could sell these remaining shares to achieve at least {target_description} overall portfolio profit, who has the {timing_word} possible sell date (the {timing_word} date in their list of valid dates) to reach this target, if all transactions were made at {price_type} prices?
""",

    'smfr_compare_multi_reverse_buy': """Here is some data on the smfr prices of a few companies. Use this data to answer the following questions:

{haystack}

{all_actions}

Each investor has completed several transactions and has already sold shares in one common smfr but has not yet bought them. Based on when they would have had to buy these shares to achieve at least {target_description} overall portfolio profit when they sold, who has the {timing_word} possible buy date (the {timing_word} date in their list of valid dates) to reach this target, if all transactions were made at {price_type} prices?
""",

    'math_single': """Solve the following problem:

{problem}

What is the answer?
""",

    'coding_single': """Given the following code:

{code}

{question}
"""
}


def get_template(template_name: str, custom_templates: Optional[Dict[str, str]] = None) -> str:
    """
    Get a template by name, checking custom templates first.

    Args:
        template_name: Name of the template
        custom_templates: Optional dictionary of custom templates

    Returns:
        Template string

    Raises:
        KeyError: If template not found
    """
    if custom_templates and template_name in custom_templates:
        return custom_templates[template_name]

    if template_name in DEFAULT_TEMPLATES:
        return DEFAULT_TEMPLATES[template_name]

    raise KeyError(f"Template '{template_name}' not found")
