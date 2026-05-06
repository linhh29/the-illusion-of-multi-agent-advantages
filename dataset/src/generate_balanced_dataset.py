"""Generate balanced dataset with all combinations of parameters."""
import sys
import random
from config_schema import smfr_comparative_target_config
from problem_composer import ProblemComposer
import json
from itertools import product

def generate_balanced_dataset(
    samples_per_combination: int = 10,
    output_file: str = 'balanced_dataset.jsonl',
    seed_start: int = 1000,
    max_retries: int = 10,
    breadth_start: int = 2,
    depth_start: int = 2,
    num_investors_start: int = 2,
    num_distractors_start: int = 2,
    max_none_samples: int = 0

):
    """
    Generate balanced dataset with equal samples of all parameter combinations.

    Args:
        samples_per_combination: Number of samples to generate per unique combination
        output_file: Output JSONL file path
        seed_start: Starting seed value
        max_retries: Maximum number of retries per sample if answer is None
        max_none_samples: Maximum number of None answer samples to retain (default: 15)
    """

    # Define all parameter combinations
    question_types = ['reverse_target_sell', 'reverse_target_buy']
    aggregations = ['earliest', 'latest']  # earliest/first, latest/last based on question type
    price_types = ['Open', 'Close']
    target_percentages = [round(0.1 + i * 0.15, 2) for i in range(13)]  # 0.1 to 2.0 in 0.15 steps

    # Calculate total combinations
    combinations = list(product(
        question_types,
        aggregations,
        price_types,
        target_percentages
    ))

    total_samples = len(combinations) * samples_per_combination

    print("=" * 80)
    print("GENERATING BALANCED DATASET")
    print("=" * 80)
    print(f"Question types: {question_types}")
    print(f"Aggregations: {aggregations}")
    print(f"Price types: {price_types}")
    print(f"Target percentages: {len(target_percentages)} values from {min(target_percentages)}% to {max(target_percentages)}%")
    print(f"Number of investors: Randomly chosen between 5-10 per sample")
    print(f"Number of distractors: Randomly chosen between 5-10 per sample")
    print(f"Breadth: Randomly chosen between 5-10 per sample")
    print(f"Depth: Randomly chosen between 5-10 (even) per sample")
    print(f"\nTotal combinations: {len(combinations)}")
    print(f"Samples per combination: {samples_per_combination}")
    print(f"Total samples to generate: {total_samples}")
    print("=" * 80)

    # Generate samples
    problems = []
    current_seed = seed_start
    none_samples_count = 0  # Track number of None answer samples retained

    for combo_idx, (q_type, agg, price_type, target_pct) in enumerate(combinations, 1):
        print(f"\n[{combo_idx}/{len(combinations)}] Generating {samples_per_combination} samples for:")
        print(f"  - Question: {q_type}")
        print(f"  - Aggregation: {agg}")
        print(f"  - Price: {price_type}")
        print(f"  - Target: {target_pct}%")

        for sample_idx in range(samples_per_combination):
            problem = None
            retry_count = 0

            # Retry until we get a non-None answer or hit max retries
            while retry_count < max_retries:
                # Randomly choose breadth, depth, number of investors, and distractors (5-10)
                breadth = random.randint(breadth_start, breadth_start)
                depth = random.randint(depth_start, depth_start)
                num_investors = random.randint(num_investors_start, num_investors_start)
                num_distractors = random.randint(num_distractors_start, num_distractors_start)

                # Ensure depth is even for proper transaction pairing
                if depth % 2 != 0:
                    depth += 1

                # Create config
                config = smfr_comparative_target_config(
                    num_investors=num_investors,
                    question_type=q_type,
                    aggregation=agg,
                    seed=current_seed,
                    breadth=breadth,
                    depth=depth,
                    task_params={
                        'target_percentage': target_pct,
                        'price_type': price_type,
                        'num_distractors': num_distractors
                    }
                )

                # Generate problem
                try:
                    composer = ProblemComposer(config)
                    problem = composer.generate_problem()

                    # Check if answer is None or contains None values
                    answer = problem.get('answer')
                    has_none = False

                    if answer is None:
                        has_none = True
                    elif isinstance(answer, dict):
                        # For structured answers, check if final answer is None
                        final_answer = answer.get('answer')
                        if final_answer is None:
                            has_none = True
                        # Also check if all investors have no valid dates
                        investor_dates = answer.get('investor_dates', {})
                        if all(not dates for dates in investor_dates.values()):
                            has_none = True

                    if has_none:
                        # None answer - decide whether to keep or retry
                        if none_samples_count < max_none_samples:
                            # Keep this None sample
                            none_samples_count += 1
                            print(f"  ℹ Sample {sample_idx + 1}: Keeping None answer ({none_samples_count}/{max_none_samples})")
                            # Don't break, continue to add metadata below
                        else:
                            # Already have enough None samples, retry with new seed
                            retry_count += 1
                            current_seed += 1
                            if retry_count < max_retries:
                                continue
                            else:
                                print(f"  ⚠ Sample {sample_idx + 1}: Failed to generate non-None answer after {max_retries} retries")
                                break

                    # Success! (Either non-None answer or None answer we're keeping)
                    # Add generation metadata
                    problem['generation_params'] = {
                        'question_type': q_type,
                        'aggregation': agg,
                        'price_type': price_type,
                        'target_percentage': target_pct,
                        'num_investors': num_investors,
                        'num_distractors': num_distractors,
                        'breadth': breadth,
                        'depth': depth,
                        'seed': current_seed,
                        'combination_index': combo_idx,
                        'sample_index': sample_idx + 1,
                        'retries_needed': retry_count
                    }

                    problems.append(problem)
                    current_seed += 1
                    break

                except Exception as e:
                    print(f"  ⚠ Error generating sample {sample_idx + 1}: {e}")
                    current_seed += 1
                    retry_count += 1
                    if retry_count >= max_retries:
                        break
                    continue

        print(f"  ✓ Generated {sample_idx + 1} samples")

    # Shuffle problems before writing
    print(f"\n{'=' * 80}")
    print(f"Shuffling {len(problems)} problems...")
    random.shuffle(problems)
    print("✓ Problems shuffled")

    # Write to file
    print(f"\nWriting {len(problems)} problems to {output_file}...")

    with open(output_file, 'w') as f:
        for problem in problems:
            f.write(json.dumps(problem) + '\n')

    print(f"✓ Dataset written to {output_file}")

    # Print statistics
    print(f"\n{'=' * 80}")
    print("DATASET STATISTICS")
    print("=" * 80)

    # Count by question type
    sell_count = sum(1 for p in problems if p['generation_params']['question_type'] == 'reverse_target_sell')
    buy_count = sum(1 for p in problems if p['generation_params']['question_type'] == 'reverse_target_buy')

    # Count by aggregation
    earliest_count = sum(1 for p in problems if p['generation_params']['aggregation'] == 'earliest')
    latest_count = sum(1 for p in problems if p['generation_params']['aggregation'] == 'latest')

    # Count by price type
    open_count = sum(1 for p in problems if p['generation_params']['price_type'] == 'Open')
    close_count = sum(1 for p in problems if p['generation_params']['price_type'] == 'Close')

    # Count by num_investors
    inv_counts = {}
    for p in problems:
        num_inv = p['generation_params']['num_investors']
        inv_counts[num_inv] = inv_counts.get(num_inv, 0) + 1

    # Count by target percentage
    target_counts = {}
    for p in problems:
        target = p['generation_params']['target_percentage']
        target_counts[target] = target_counts.get(target, 0) + 1

    # Count answers with ties
    tie_count = 0
    for p in problems:
        if isinstance(p['answer'], dict) and isinstance(p['answer'].get('answer'), list):
            tie_count += 1

    # Count None answers
    none_count = 0
    for p in problems:
        answer = p.get('answer')
        if answer is None:
            none_count += 1
        elif isinstance(answer, dict) and answer.get('answer') is None:
            none_count += 1

    # Calculate retry statistics
    retry_counts = [p['generation_params'].get('retries_needed', 0) for p in problems]
    avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0
    max_retries_used = max(retry_counts) if retry_counts else 0
    problems_with_retries = sum(1 for r in retry_counts if r > 0)

    print(f"Total problems: {len(problems)}")
    print(f"\nBy question type:")
    print(f"  - reverse_target_sell: {sell_count}")
    print(f"  - reverse_target_buy: {buy_count}")
    print(f"\nBy aggregation:")
    print(f"  - earliest/first: {earliest_count}")
    print(f"  - latest/last: {latest_count}")
    print(f"\nBy price type:")
    print(f"  - Open: {open_count}")
    print(f"  - Close: {close_count}")
    print(f"\nBy number of investors:")
    for num_inv in sorted(inv_counts.keys()):
        print(f"  - {num_inv} investors: {inv_counts[num_inv]}")
    print(f"\nBy target percentage (first 5):")
    for target in sorted(target_counts.keys())[:5]:
        print(f"  - {target}%: {target_counts[target]}")
    print(f"  ... ({len(target_counts)} unique target percentages)")
    print(f"\nAnswer quality:")
    print(f"  - None answers: {none_count} ({none_count/len(problems)*100:.1f}%) [max allowed: {max_none_samples}]")
    print(f"  - Answers with ties: {tie_count} ({tie_count/len(problems)*100:.1f}%)")
    print(f"\nRetry statistics:")
    print(f"  - Problems requiring retries: {problems_with_retries} ({problems_with_retries/len(problems)*100:.1f}%)")
    print(f"  - Average retries per problem: {avg_retries:.2f}")
    print(f"  - Maximum retries used: {max_retries_used}")
    print("=" * 80)

    return problems


if __name__ == '__main__':
    # Generate 10 samples per combination (default)
    # Total: 2 q_types × 2 aggs × 2 prices × 13 targets × 10 samples = 1,040 problems
    problems = generate_balanced_dataset(
        samples_per_combination=1,
        output_file='balanced_dataset_single_{}.jsonl'.format(sys.argv[1]),
        seed_start=1000,
        breadth_start=int(sys.argv[1]),
        depth_start=int(sys.argv[1]),
        num_investors_start=int(sys.argv[1]),
        num_distractors_start=int(sys.argv[1]),
    )

    print(f"\n✓ Generated {len(problems)} problems")
    print(f"✓ Dataset saved to balanced_dataset.jsonl")
