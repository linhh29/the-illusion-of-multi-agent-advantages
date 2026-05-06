"""
Generic math problem generation task.

This task generates various types of math problems including arithmetic,
algebra, and word problems. Unlike smfr tasks, these don't require
time-varying data sources.
"""

import random
from typing import Dict, Any, List, Optional
from task_base import BaseTask, BaseDataSource


class MathTask(BaseTask):
    """
    Task for generating generic math problems.

    Supports arithmetic, algebra, and word problems with configurable
    difficulty and operations.
    """

    def __init__(self, config: Dict[str, Any], data_sources: Optional[List[BaseDataSource]] = None):
        """Initialize math task with config."""
        super().__init__(config, data_sources or [])

        self.seed = config.get('seed', 100)
        task_params = config.get('task_params', {})

        self.problem_type = task_params.get('problem_type', 'arithmetic')
        self.difficulty = task_params.get('difficulty', 'medium')
        self.operations = task_params.get('operations', ['+', '-', '*', '/'])
        self.number_range = task_params.get('number_range', (1, 100))

    def generate_haystack(self, seed: int) -> Dict[str, Any]:
        """
        Generate haystack data (for math, this could be a set of equations or facts).

        Args:
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing haystack data
        """
        random.seed(seed)

        # For math problems, haystack could be additional context or related problems
        # For now, keeping it simple
        haystack = {
            'seed': seed,
            'context': '',  # Could add math facts, formulas, etc.
            'metadata': {
                'problem_type': self.problem_type,
                'difficulty': self.difficulty
            }
        }

        return haystack

    def generate_needles(self, haystack: Dict[str, Any], seed: int, count: int = 1) -> List[Dict[str, Any]]:
        """
        Generate math problem needles.

        Args:
            haystack: Haystack data
            seed: Random seed
            count: Number of problem instances

        Returns:
            List of problem dictionaries
        """
        random.seed(seed)
        needles = []

        for i in range(count):
            if self.problem_type == 'arithmetic':
                needle = self._generate_arithmetic_problem(seed + i)
            elif self.problem_type == 'algebra':
                needle = self._generate_algebra_problem(seed + i)
            elif self.problem_type == 'word_problem':
                needle = self._generate_word_problem(seed + i)
            else:
                raise ValueError(f"Unknown problem type: {self.problem_type}")

            needles.append(needle)

        return needles

    def _generate_arithmetic_problem(self, seed: int) -> Dict[str, Any]:
        """Generate an arithmetic problem."""
        random.seed(seed)

        min_val, max_val = self.number_range
        num_operations = {'easy': 2, 'medium': 3, 'hard': 4}.get(self.difficulty, 3)

        # Generate a chain of operations
        numbers = [random.randint(min_val, max_val) for _ in range(num_operations + 1)]
        operations = [random.choice(self.operations) for _ in range(num_operations)]

        # Build expression
        expression = str(numbers[0])
        steps = [f"Start with {numbers[0]}"]

        result = numbers[0]
        for i, op in enumerate(operations):
            num = numbers[i + 1]
            expression += f" {op} {num}"

            if op == '+':
                result += num
                steps.append(f"Add {num}: {result}")
            elif op == '-':
                result -= num
                steps.append(f"Subtract {num}: {result}")
            elif op == '*':
                result *= num
                steps.append(f"Multiply by {num}: {result}")
            elif op == '/':
                # Ensure clean division
                if result % num == 0:
                    result //= num
                    steps.append(f"Divide by {num}: {result}")
                else:
                    result = result / num
                    steps.append(f"Divide by {num}: {result}")

        return {
            'expression': expression,
            'steps': steps,
            'answer': result,
            'problem_text': f"Calculate: {expression}"
        }

    def _generate_algebra_problem(self, seed: int) -> Dict[str, Any]:
        """Generate a simple algebra problem (solve for x)."""
        random.seed(seed)

        min_val, max_val = self.number_range

        # Generate simple linear equation: ax + b = c
        a = random.randint(2, 10)
        b = random.randint(min_val, max_val)
        x = random.randint(1, 20)  # The answer
        c = a * x + b

        problem_text = f"Solve for x: {a}x + {b} = {c}"

        steps = [
            f"Start with: {a}x + {b} = {c}",
            f"Subtract {b} from both sides: {a}x = {c - b}",
            f"Divide both sides by {a}: x = {(c - b) / a}"
        ]

        return {
            'expression': f"{a}x + {b} = {c}",
            'steps': steps,
            'answer': x,
            'problem_text': problem_text
        }

    def _generate_word_problem(self, seed: int) -> Dict[str, Any]:
        """Generate a word problem."""
        random.seed(seed)

        templates = [
            {
                'text': "Sarah has {a} apples. She buys {b} more apples at the store. How many apples does she have now?",
                'operation': '+',
            },
            {
                'text': "John has {a} dollars. He spends {b} dollars on lunch. How much money does he have left?",
                'operation': '-',
            },
            {
                'text': "A box contains {a} items. If there are {b} boxes, how many items are there in total?",
                'operation': '*',
            }
        ]

        template = random.choice(templates)
        min_val, max_val = self.number_range

        a = random.randint(min_val, max_val)
        b = random.randint(1, min(50, max_val))

        problem_text = template['text'].format(a=a, b=b)

        if template['operation'] == '+':
            answer = a + b
            steps = [f"{a} + {b} = {answer}"]
        elif template['operation'] == '-':
            answer = a - b
            steps = [f"{a} - {b} = {answer}"]
        elif template['operation'] == '*':
            answer = a * b
            steps = [f"{a} × {b} = {answer}"]
        else:
            answer = a / b
            steps = [f"{a} ÷ {b} = {answer}"]

        return {
            'expression': None,
            'steps': steps,
            'answer': answer,
            'problem_text': problem_text
        }

    def compute_answer(self, needle: Dict[str, Any]) -> Any:
        """
        Compute answer for a math problem.

        Args:
            needle: Problem data

        Returns:
            The answer
        """
        return needle['answer']

    def format_problem(self, haystack: Dict[str, Any], needles: List[Dict[str, Any]],
                      question_template: str, extra_vars: Optional[Dict[str, str]] = None) -> str:
        """
        Format the complete problem text.

        Args:
            haystack: Haystack data
            needles: List of problem data
            question_template: Template string
            extra_vars: Optional extra template variables

        Returns:
            Formatted problem string
        """
        problem_texts = [needle['problem_text'] for needle in needles]

        if len(needles) == 1:
            return question_template.format(
                problem=problem_texts[0],
                haystack=haystack.get('context', '')
            )
        else:
            # Multiple problems
            all_problems = '\n\n'.join([f"Problem {i+1}: {p}" for i, p in enumerate(problem_texts)])
            return question_template.format(
                problem=all_problems,
                haystack=haystack.get('context', '')
            )

    def format_cot(self, needles: List[Dict[str, Any]], answers: List[Any]) -> str:
        """
        Format chain-of-thought reasoning.

        Args:
            needles: List of problem data
            answers: List of answers

        Returns:
            Formatted chain-of-thought string
        """
        cot_parts = []

        for i, (needle, answer) in enumerate(zip(needles, answers)):
            if len(needles) > 1:
                cot = f"Problem {i+1}:\n"
            else:
                cot = ""

            cot += '\n'.join(needle['steps'])
            cot += f"\nAnswer: {answer}"
            cot_parts.append(cot)

        return '\n\n'.join(cot_parts)

    def get_task_type(self) -> str:
        """Return task type identifier."""
        return "math"
