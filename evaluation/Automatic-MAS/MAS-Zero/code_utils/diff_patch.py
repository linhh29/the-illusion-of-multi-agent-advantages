import re
from typing import List, Tuple, Dict


def _strip_code_fence(patch: str) -> str:
    # Remove ```diff ... ``` fences if present
    m = re.search(r"^```(?:diff)?\s*\n(.*)```$", patch.strip(), flags=re.DOTALL)
    return m.group(1) if m else patch


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _parse_unified_diff(patch: str) -> List[Dict[str, List[str]]]:
    """
    Return a list of hunks. Each hunk dict has:
      - "from_lines": the lines to match in the original (context ' ' + removed '-')
      - "to_lines":   the lines to replace with      (context ' ' + added   '+')
    Headers (diff --git / --- / +++) and line numbers (@@) are ignored.
    """
    lines = _normalize_newlines(_strip_code_fence(patch)).split("\n")
    hunks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            # Collect all hunk body lines until next header/next hunk/end
            i += 1
            body = []
            while i < len(lines):
                l = lines[i]
                if l.startswith("@@") or l.startswith("diff --git ") or l.startswith("--- ") or l.startswith("+++ "):
                    break
                # valid hunk lines start with ' ', '+', '-' (ignore special backslash newline markers)
                if l.startswith((" ", "+", "-")) or (l.startswith("\\") and "No newline" in l):
                    body.append(l)
                i += 1
            # Build from/to sequences
            from_lines = [l[1:] for l in body if l.startswith((" ", "-"))]
            to_lines = [l[1:] for l in body if l.startswith((" ", "+"))]
            hunks.append({"from_lines": from_lines, "to_lines": to_lines})
            continue
        else:
            i += 1
    return hunks


def _build_whitespace_tolerant_regex(lines: List[str]) -> re.Pattern:
    """
    Build a regex that matches the lines in order with flexible whitespace:
      - spaces/tabs within a line -> \s+
      - line boundaries -> \s*\n\s* (allows extra blank/indented lines between)
    """

    # Escape each line, then relax runs of whitespace
    def relax(line: str) -> str:
        # Escape, then replace escaped spaces/tabs/runs with \s+
        esc = re.escape(line)
        esc = re.sub(r"(\\ )+", r"\\s+", esc)  # spaces
        esc = re.sub(r"(\\t)+", r"\\s+", esc)  # tabs
        esc = re.sub(r"(\\s\+)+", r"\\s+", esc)  # collapse repeated
        return esc

    parts = [relax(l) for l in lines]
    # Allow some whitespace around line breaks
    joined = r"\s*\n\s*".join(parts)
    return re.compile(joined, flags=re.MULTILINE)


def apply_unified_diff(
        original_text: str,
        patch_text: str,
        whitespace_fallback: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """
    Apply a unified-diff-like patch to a single text string.

    - Ignores file headers and line numbers.
    - Applies each hunk by matching the concatenated (context + removed) lines
      and replacing with (context + added) lines.
    - Tries exact (strict) match first; if not found and whitespace_fallback=True,
      tries a whitespace-tolerant regex.

    Returns: (new_text, stats) where stats = {"hunks_total": N, "hunks_applied": K, "hunks_failed": N-K}
    """
    text = _normalize_newlines(original_text)
    hunks = _parse_unified_diff(patch_text)

    applied = 0
    for h in hunks:
        from_block = "\n".join(h["from_lines"])
        to_block = "\n".join(h["to_lines"])

        # 1) Strict substring search
        idx = text.find(from_block)
        if idx != -1:
            text = text[:idx] + to_block + text[idx + len(from_block):]
            applied += 1
            continue

        # 2) Whitespace-tolerant fallback
        if whitespace_fallback and h["from_lines"]:
            pat = _build_whitespace_tolerant_regex(h["from_lines"])
            m = pat.search(text)
            if m:
                text = text[:m.start()] + to_block + text[m.end():]
                applied += 1
                continue
        # If neither matched, we leave this hunk unapplied and move on.

    stats = {
        "hunks_total": len(hunks),
        "hunks_applied": applied,
        "hunks_failed": len(hunks) - applied,
    }
    return text, stats


# -------------------------
# Example usage
if __name__ == "__main__":
    snippet = """
async def forward(self, taskInfo, extra_info):
    from collections import namedtuple
    sub_tasks = []
    agents = []
    
    # Sub-task 1: Use Chain-of-Thought to find the smallest prime p
    cot_instruction = "Sub-task 1: Determine the smallest prime number p such that there exists an integer n for which n^4 + 1 is divisible by p^2."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], extra_info, cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, on the purpose of finding smallest prime p, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    
    # Sub-task 2: Use Reflexion to find the smallest integer m based on p
    cot_reflect_instruction = "Sub-task 2: Based on the smallest prime p found in sub-task 1, find the smallest positive integer m such that m^4 + 1 is divisible by p^2."
    cot_agent2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_instruction = "Sub-task 2: Based on the smallest prime p found in sub-task 1, review the current guess for m and provide feedback."
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    
    N_max = self.max_round  # Maximum number of attempts
    cot_inputs = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent2(cot_inputs, extra_info, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2.id}, on the purpose of finding smallest integer m, thinking: {thinking2.content}; answer: {answer2.content}')
    
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking1, answer1, thinking2, answer2], extra_info, critic_instruction, i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, on the purpose of providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent2(cot_inputs, extra_info, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'CoT agent {cot_agent2.id}, on the purpose of refining smallest integer m, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    final_answer = self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer
""".strip("\n")

    patch = """```diff
--- a/main.py
+++ b/main.py
@@ -30,6 +30,9 @@
     # Sub-task 1: Use Chain-of-Thought to find the smallest prime p
     cot_instruction = "Sub-task 1: Determine the smallest prime number p such that there exists an integer n for which n^4 + 1 is divisible by p^2."
     cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
+    # Reuse from previous round
+    thinking1, answer1 = extra_info['round_1'][0]
+    if 'round_2' not in extra_info:
+        extra_info['round_2'] = []
+    extra_info['round_2'].append([thinking1, answer1])
     thinking1, answer1 = await cot_agent([taskInfo], extra_info, cot_instruction, is_sub_task=True)
     agents.append(f'CoT agent {cot_agent.id}, on the purpose of finding smallest prime p, thinking: {thinking1.content}; answer: {answer1.content}')
     sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
@@ -37,7 +40,7 @@
     cot_reflect_instruction = "Sub-task 2: Based on the smallest prime p found in sub-task 1, find the smallest positive integer m such that m^4 + 1 is divisible by p^2."
     cot_agent2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
     critic_instruction = "Sub-task 2: Based on the smallest prime p found in sub-task 1, review the current guess for m and provide feedback. It is known that (8) is not correct."
     critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
+    # Enhance feedback to avoid known incorrect answers
+    critic_instruction = "Sub-task 2: Based on the smallest prime p found in sub-task 1, review the current guess for m and provide feedback. It is known that (8) is not correct. Provide detailed advice for refining guesses."
 
     N_max = self.max_round  # Maximum number of attempts
     cot_inputs = [taskInfo, thinking1, answer1]
```"""

    new_text, stats = apply_unified_diff(snippet, patch)
    print("Applied:", stats)
    print("--- before ---")
    print(snippet)
    print("--- after ----")
    print(new_text)
