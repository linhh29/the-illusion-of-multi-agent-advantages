"""
Problem composition engine.

This module orchestrates the generation of complete problems by coordinating
tasks, applying aggregation operations, and formatting output.
"""

from typing import Dict, Any, List, Optional
from config_schema import PipelineConfig, AggregationOp, get_template, DEFAULT_TEMPLATES
from task_base import BaseTask
from tasks.smfr_task import SMFRTask
from tasks.math_task import MathTask
from tasks.coding_task import CodingTask
from tqdm import tqdm


class ProblemComposer:
    """
    Orchestrates problem generation using tasks and composition rules.

    The composer:
    1. Generates haystack data
    2. Creates multiple needle instances (if requested)
    3. Computes answers for each instance
    4. Applies aggregation operations
    5. Formats the final problem, CoT, and answer
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize composer with pipeline configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.task = self._create_task(config.task)

    def _create_task(self, task_config) -> BaseTask:
        """
        Factory method to create task instances.

        Args:
            task_config: Task configuration

        Returns:
            Task instance

        Raises:
            ValueError: If task type is unknown
        """
        task_type = task_config.task_type
        config_dict = {
            'seed': task_config.seed,
            'breadth': task_config.breadth,
            'depth': task_config.depth,
            'task_params': task_config.task_params
        }

        if task_type == 'smfr':
            return SMFRTask(config_dict)
        elif task_type == 'math':
            return MathTask(config_dict)
        elif task_type == 'coding':
            return CodingTask(config_dict)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def generate_problem(self) -> Dict[str, Any]:
        """
        Generate a complete problem.

        Returns:
            Dictionary containing:
                - problem: Problem text
                - cot: Chain-of-thought reasoning
                - answer: Final answer
                - metadata: Generation metadata
        """
        seed = self.config.task.seed
        num_instances = self.config.composition.num_instances

        # Step 1: Generate haystack
        haystack = self.task.generate_haystack(seed)

        # Step 2: Generate needles (multiple instances if requested)
        needles = self.task.generate_needles(haystack, seed, count=num_instances)

        # Step 3: Compute answers for each instance
        answers = [self.task.compute_answer(needle) for needle in needles]

        # Step 4: Apply aggregation operation
        final_answer = self._apply_aggregation(needles, answers)

        # Step 5: Select appropriate template
        template = self._select_template(num_instances)

        # Step 6: Get template variables based on aggregation
        template_vars = self._get_comparison_words(num_instances)

        # Step 7: Format problem text with comparison words injected
        problem_text = self.task.format_problem(haystack, needles, template, template_vars)

        # Step 8: Format chain-of-thought
        cot_text = self.task.format_cot(needles, answers)

        # Add aggregation step to CoT if needed
        if num_instances > 1 and self.config.composition.aggregation_op != AggregationOp.FIRST:
            cot_text += self._format_aggregation_cot(needles, answers, final_answer)

        # Step 9: Build result
        result = {
            'problem': problem_text,
            'answer': final_answer
        }

        if self.config.include_cot:
            result['cot'] = cot_text

        if self.config.include_metadata:
            result['metadata'] = {
                'task_type': self.task.get_task_type(),
                'seed': seed,
                'num_instances': num_instances,
                'aggregation_op': self.config.composition.aggregation_op.value,
                'haystack_metadata': haystack.get('metadata', {}),
                'data_requirements': self.task.get_data_requirements(),
                'task_params': self.config.task.task_params,  # Store task params for updates
                'breadth': self.config.task.breadth,
                'depth': self.config.task.depth
            }

            # Store data for potential updates
            if self.task.get_data_requirements():
                result['metadata']['updatable_data'] = {
                    'haystack': haystack,
                    'needles': needles
                }

        return result

    def _apply_aggregation(self, needles: List[Dict[str, Any]], answers: List[Any]) -> Any:
        """
        Apply aggregation operation to multiple answers.

        For comparative date questions (EARLIEST_DATE, LATEST_DATE with multiple investors),
        returns structured answer with intermediate calculations to prevent guessing.

        Args:
            needles: List of needle instances
            answers: List of computed answers

        Returns:
            Aggregated answer (may be structured dict for comparative questions)
        """
        op = self.config.composition.aggregation_op

        if op == AggregationOp.FIRST or len(answers) == 1:
            return answers[0]

        if op == AggregationOp.LAST:
            return answers[-1]

        if op == AggregationOp.MAX:
            # Return the entity with max value
            max_idx = answers.index(max(answers))
            # Try to get entity name from needle
            entity_key = self._get_entity_key(needles[0])
            return needles[max_idx].get(entity_key, answers[max_idx])

        if op == AggregationOp.MIN:
            # Return the entity with min value
            min_idx = answers.index(min(answers))
            entity_key = self._get_entity_key(needles[0])
            return needles[min_idx].get(entity_key, answers[min_idx])

        if op == AggregationOp.SUM:
            return sum(answers)

        if op == AggregationOp.AVG:
            return sum(answers) / len(answers)

        if op == AggregationOp.COMPARE_GT:
            # Return entity with greater value
            entity_key = self._get_entity_key(needles[0])
            return needles[0].get(entity_key) if answers[0] > answers[1] else needles[1].get(entity_key)

        if op == AggregationOp.COMPARE_LT:
            # Return entity with lesser value
            entity_key = self._get_entity_key(needles[0])
            return needles[0].get(entity_key) if answers[0] < answers[1] else needles[1].get(entity_key)

        if op == AggregationOp.COMPARE_EQ:
            # Check if all equal
            return all(a == answers[0] for a in answers)

        if op == AggregationOp.EARLIEST_DATE:
            # For reverse questions: find who can achieve target earliest
            # Return structured answer with all intermediate calculations
            from datetime import datetime

            entity_key = self._get_entity_key(needles[0])

            # Build investor_dates mapping
            investor_dates = {}
            comparison = {}

            for needle, date_list in zip(needles, answers):
                investor_name = needle.get(entity_key)
                investor_dates[investor_name] = date_list

                # Get first date for comparison (or None)
                comparison[investor_name] = date_list[0] if date_list else None

            # Find winners (earliest first date)
            # Handle ties: multiple investors with same earliest date
            winners = []
            earliest_date = None

            for investor_name, first_date_str in comparison.items():
                if first_date_str:
                    first_date = datetime.strptime(first_date_str, "%B %d, %Y")

                    if earliest_date is None or first_date < earliest_date:
                        earliest_date = first_date
                        winners = [investor_name]  # New earliest, reset winners
                    elif first_date == earliest_date:
                        winners.append(investor_name)  # Tie, add to winners

            # Sort winners alphabetically for consistency
            answer = sorted(winners) if winners else None

            return {
                "investor_dates": investor_dates,
                "comparison": comparison,
                "answer": answer
            }

        if op == AggregationOp.LATEST_DATE:
            # For reverse questions: find who takes longest to achieve target
            # Return structured answer with all intermediate calculations
            from datetime import datetime

            entity_key = self._get_entity_key(needles[0])

            # Build investor_dates mapping
            investor_dates = {}
            comparison = {}

            for needle, date_list in zip(needles, answers):
                investor_name = needle.get(entity_key)
                investor_dates[investor_name] = date_list

                # Get first date for comparison (or None)
                comparison[investor_name] = date_list[0] if date_list else None

            # Find winners (latest first date)
            # Handle ties: multiple investors with same latest date
            winners = []
            latest_date = None

            for investor_name, first_date_str in comparison.items():
                if first_date_str:
                    first_date = datetime.strptime(first_date_str, "%B %d, %Y")

                    if latest_date is None or first_date > latest_date:
                        latest_date = first_date
                        winners = [investor_name]  # New latest, reset winners
                    elif first_date == latest_date:
                        winners.append(investor_name)  # Tie, add to winners

            # Sort winners alphabetically for consistency
            answer = sorted(winners) if winners else None

            return {
                "investor_dates": investor_dates,
                "comparison": comparison,
                "answer": answer
            }

        return answers[0]

    def _get_entity_key(self, needle: Dict[str, Any]) -> str:
        """
        Determine the key used for entity names in needles.

        Args:
            needle: A needle instance

        Returns:
            Key string (e.g., 'investor', 'person', 'entity')
        """
        # Try common keys
        for key in ['investor', 'person', 'entity', 'name', 'id']:
            if key in needle:
                return key
        return 'entity'

    def _format_aggregation_cot(self, needles: List[Dict[str, Any]], answers: List[Any], final_answer: Any) -> str:
        """
        Format the aggregation step in chain-of-thought.

        Args:
            needles: List of needle instances
            answers: List of individual answers
            final_answer: Aggregated answer

        Returns:
            Formatted aggregation explanation
        """
        op = self.config.composition.aggregation_op

        if op == AggregationOp.MAX:
            return f"\n\nComparing all values, the maximum is {max(answers)}.\nAnswer: {final_answer}"

        if op == AggregationOp.MIN:
            return f"\n\nComparing all values, the minimum is {min(answers)}.\nAnswer: {final_answer}"

        if op == AggregationOp.SUM:
            return f"\n\nSum of all values: {' + '.join(map(str, answers))} = {final_answer}"

        if op == AggregationOp.AVG:
            return f"\n\nAverage of all values: ({' + '.join(map(str, answers))}) / {len(answers)} = {final_answer}"

        if op in [AggregationOp.COMPARE_GT, AggregationOp.COMPARE_LT]:
            return f"\n\nAnswer: {final_answer}"

        return ""

    def _select_template(self, num_instances: int) -> str:
        """
        Select appropriate template based on task type and number of instances.

        Args:
            num_instances: Number of parallel instances

        Returns:
            Template string
        """
        # Check for custom template
        if self.config.composition.question_template:
            return self.config.composition.question_template

        # Check task-specific templates in config
        custom_templates = self.config.templates

        # Select default template based on task type and instance count
        task_type = self.task.get_task_type()

        if task_type == 'smfr':
            # Check question type for smfr tasks
            question_type = self.config.task.task_params.get('question_type', 'spending')

            if num_instances == 1:
                if question_type == 'profit_loss':
                    template_name = 'smfr_single_profit_loss'
                elif question_type == 'reverse_target_sell':
                    template_name = 'smfr_single_reverse_sell'
                elif question_type == 'reverse_target_buy':
                    template_name = 'smfr_single_reverse_buy'
                else:
                    template_name = 'smfr_single'
            elif num_instances == 2:
                if question_type == 'reverse_target_sell':
                    template_name = 'smfr_compare_two_reverse_sell'
                elif question_type == 'reverse_target_buy':
                    template_name = 'smfr_compare_two_reverse_buy'
                else:
                    template_name = 'smfr_compare_two'
            else:
                if question_type == 'reverse_target_sell':
                    template_name = 'smfr_compare_multi_reverse_sell'
                elif question_type == 'reverse_target_buy':
                    template_name = 'smfr_compare_multi_reverse_buy'
                else:
                    template_name = 'smfr_compare_multi'
        elif task_type == 'math':
            template_name = 'math_single'
        elif task_type == 'coding':
            template_name = 'coding_single'
        else:
            # Generic template
            return "{haystack}\n\n{question}"

        return get_template(template_name, custom_templates)

    def _get_comparison_words(self, num_instances: int) -> Dict[str, str]:
        """
        Get appropriate comparison words based on aggregation operation.

        Args:
            num_instances: Number of parallel instances

        Returns:
            Dictionary with comparison_word, superlative_word, target_description, and timing_word
        """
        op = self.config.composition.aggregation_op

        # For two instances (comparison)
        if num_instances == 2:
            if op == AggregationOp.MAX:
                comparison_word = "more"
            elif op == AggregationOp.MIN:
                comparison_word = "less"
            elif op == AggregationOp.COMPARE_GT:
                comparison_word = "more"
            elif op == AggregationOp.COMPARE_LT:
                comparison_word = "less"
            else:
                comparison_word = "more"  # default
        else:
            comparison_word = "more"

        # For multiple instances (superlative)
        if num_instances > 2:
            if op == AggregationOp.MAX:
                superlative_word = "most"
            elif op == AggregationOp.MIN:
                superlative_word = "least"
            else:
                superlative_word = "most"  # default
        else:
            superlative_word = "most"

        # For reverse target questions
        target_percentage = self.config.task.task_params.get('target_percentage')
        target_amount = self.config.task.task_params.get('target_amount')

        if target_percentage is not None:
            target_description = f"{target_percentage}%"
        elif target_amount is not None:
            target_description = f"${target_amount}"
        else:
            target_description = "the target"

        # Timing words for comparative reverse questions
        question_type = self.config.task.task_params.get('question_type', 'spending')

        if question_type == 'reverse_target_sell':
            # For sell questions: use "earliest" and "latest" (future action)
            if op == AggregationOp.EARLIEST_DATE:
                timing_word = "earliest"
            elif op == AggregationOp.LATEST_DATE:
                timing_word = "latest"
            else:
                timing_word = "earliest"
        elif question_type == 'reverse_target_buy':
            # For buy questions: use "first" and "last" (past action)
            if op == AggregationOp.EARLIEST_DATE:
                timing_word = "first"
            elif op == AggregationOp.LATEST_DATE:
                timing_word = "last"
            else:
                timing_word = "first"
        else:
            # Default for other question types
            timing_word = "soonest"

        return {
            'comparison_word': comparison_word,
            'superlative_word': superlative_word,
            'target_description': target_description,
            'timing_word': timing_word
        }


def generate_multiple_problems(configs: List[PipelineConfig], show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Generate multiple problems from a list of configurations.

    Args:
        configs: List of pipeline configurations
        show_progress: Whether to show progress bar (default True)

    Returns:
        List of generated problems
    """
    problems = []
    iterator = tqdm(configs, desc="Generating problems") if show_progress else configs

    for config in iterator:
        composer = ProblemComposer(config)
        problem = composer.generate_problem()
        problems.append(problem)

    return problems


def generate_problem_batch(config: PipelineConfig, count: int, seed_start: int = 100, show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Generate a batch of similar problems with different seeds.

    Args:
        config: Base pipeline configuration
        count: Number of problems to generate
        seed_start: Starting seed value
        show_progress: Whether to show progress bar (default True)

    Returns:
        List of generated problems
    """
    problems = []
    iterator = tqdm(range(count), desc="Generating batch") if show_progress else range(count)

    for i in iterator:
        # Create new config with updated seed
        batch_config = config
        batch_config.task.seed = seed_start + i

        composer = ProblemComposer(batch_config)
        problem = composer.generate_problem()
        problems.append(problem)

    return problems
