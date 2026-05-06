# **Example**: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1],[], []], and a solution might be:
#    moves = [[1 , 0 , 2], [2 , 0 , 1], [1 , 2 , 1], [3 , 0 , 2], [1 , 1 , 0], [2 , 1 , 2], [1 , 0 , 2]]

# This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.
# The code that can solve this puzzle is:
# moves = []
# def hanoi_solution(n, source, target, auxiliary, moves):
#     if n == 1:
#         moves.append([1, source, target])
#         return
#     hanoi_solution(n - 1, source, auxiliary, target, moves)
#     moves.append([n, source, target])
#     hanoi_solution(n - 1, auxiliary, target, source, moves)


# When you perform the recursive algrithom and divide the problem into some sub-problems, please first think about the start configuration and the end configuration of each sub-problems and then output your answer.
# For example:
# With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1], [], []].

# Following the recursive algrithom:
# Sub-problem 1. Move 2 disks from source (Peg 0) to auxiliary (Peg1) peg.
# The start state of this sub-problem is [[3, 2, 1], [], []], the end state of this sub-problem is [[3], [2, 1], []].
# So the moves of this sub-problem is: [[1 , 0 , 2] , [2 , 0 , 1] , [1 , 2 , 1]]

# Sub-problem 2. Move the 3th disk from source (Peg 0) to target (Peg2).
# The start state of this sub-problem is [[3], [2, 1], []] the end state of this sub-problem is [[], [2, 1], [3]].
# So the moves of this sub-problem is: [[3 , 0 , 2]]

# Sub-problem 3. Move 2 disks from auxiliary (Peg1) to target (Peg2).
# The start state of this sub-problem is [[], [2, 1], [3]] the end state of this sub-problem is [[], [], [3, 2, 1]].
# So the moves of this sub-problem is: [[1 , 1 , 0] , [2 , 1 , 2] , [1 , 0 , 2]]

# Finally, combine all the sequence of Sub-problems.
# Answer: [[1 , 0 , 2] , [2 , 0 , 1] , [1 , 2 , 1] , [3 , 0 , 2] , [1 , 1 , 0] , [2 , 1 , 2] , [1 , 0 , 2]]
task_description = '''
Solve this puzzle for me. 
There are three pegs and n disks of different sizes stacked on the first peg. 
The disks are numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
    1. Only one disk can be moved at a time.
    2. Each move consists of taking the upper disk from one stack and placing it on top of another stack.
    3. A larger disk may not be placed on top of a smaller disk.

The goal is to move the entire stack to the third peg.

Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1], [], []], and a solution might be:
moves = [[1 , 0 , 2] , [2 , 0 , 1] , [1 , 2 , 1] , [3 , 0 , 2] , [1 , 1 , 0] , [2 , 1 , 2] , [1 , 0 , 2]]
This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.

**Requirements**:
    • When exploring potential solutions in your thinking process, always include the corresponding complete list of moves.
    • The positions are 0-indexed (the leftmost peg is 0).
    • Ensure your final answer includes the complete list of moves in the format: [[disk id, from peg, to peg], ...].
    • Although the moves sequence will be long for a large number of disks, please be sure to output the full move sequence. Any responses regarding function calls and without the full move sequence will not be accepted.
    • If you think the response will exceed the output size limitations, please output the sequence as long as you can.
'''

# To solve the entire puzzle of moving n disks from peg 0 to peg 2:
# 1. Initialize an empty list moves
# 2. Execute Solve(n, 0, 2, 1, moves)
# 3. The moves list will contain the complete solution
# algorithm = ''
algorithm = '''
Here is a pseudocode of recursive algorithm to solve the puzzle:
ALGORITHM Solve(n, source, target, auxiliary, moves)
    // n = number of disks to move
    // source = starting peg (0, 1, or 2)
    // target = destination peg (0, 1, or 2)
    // auxiliary = the unused peg (0, 1, or 2)
    // moves = list to store the sequence of moves

    IF n equals 1 THEN
    // Get the top disk from source peg
    disk = the top disk on the source peg
    // Add the move to our list: [disk_id, source, target]
    ADD [disk, source, target] to moves
    RETURN
    END IF

    // Move n-1 disks from source to auxiliary peg
    Solve(n-1, source, auxiliary, target, moves)

    // Move the nth disk from source to target
    disk = the top disk on the source peg
    ADD [disk, source, target] to moves

    // Move n-1 disks from auxiliary to target
    Solve(n-1, auxiliary, target, source, moves)
END ALGORITHM
'''

problem_template = '''
I have a puzzle with ${number}$ disks of different sizes with
**Initial configuration**:
    • Peg 0: ${number}$ (bottom), . . . 2, 1 (top)
    • Peg 1: (empty)
    • Peg 2: (empty)

**Goal configuration**:
    • Peg 0: (empty)
    • Peg 1: (empty)
    • Peg 2: ${number}$ (bottom), . . . 2, 1 (top)

**Rules**:
    • Only one disk can be moved at a time.
    • Only the top disk from any stack can be moved.
    • A larger disk may not be placed on top of a smaller disk.

Find the sequence of moves to transform the initial configuration into the goal configuration.
Note that Peg 1 is the auxiliary disk and Peg 2 is the target disk.
Although the moves sequence will be long for a large number of disks, please be sure to output the full move sequence. 
Any responses regarding function calls and without the full move sequence will not be accepted.
If you think the response will exceed the output size limitations, please output the sequence as long as you can.
'''

judge_prompt = '''
Carefully evaluate each candidate answer against these mandatory checks (in order of priority):

1. Basic Rule Validation
    Does each move transfer exactly one disk? (Verify move array length)
    Is there any larger disk placed atop a smaller one? (Simulate peg states after each move)
    Are all moves sourced from non-empty pegs? (Check "from" peg availability)

2. Completeness Verification
    Are all n disks relocated to the target peg (peg 2)?
    Is the final state [[], [], [n, n-1,...,1]]?

3. Formatting Requirements
    Strict adherence to [[disk, from, to],...] format
'''

import re


def find_answer(response_text):
    # match = re.search(r'(\[\s*\[.*\]\s*\])', response_text)
    # if match:
    #     extracted_answer = match.group(1)
    # else:
    if '\\boxed{' in response_text:
        extracted_answer = response_text.split('\\boxed{')[-1].strip()
    elif 'Answer:' in response_text:
        extracted_answer = response_text.split('Answer:')[-1].strip()
    else:
        extracted_answer = response_text

    while len(extracted_answer) > 0 and extracted_answer[0] != '[':
        extracted_answer = extracted_answer[1:]

    while len(extracted_answer) > 0 and extracted_answer[-1] != ']':
        extracted_answer = extracted_answer[:-1]

    if '[TOO_HARD]' in extracted_answer:  # we cannot add [TOO_HARD] in memory
        extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]

    if len(extracted_answer) == 0:
        extracted_answer = '[[1, 0, 3]]'
    return extracted_answer


def load_hanoi(start, end):
    datas = []
    for N in range(start, end + 1):
        example = {'N': N}
        example['problem'] = problem_template.format(number=N) + '\n\n' + algorithm

        moves = []

        def hanoi_solution(n, source, target, auxiliary, moves):
            if n == 1:
                moves.append([1, source, target])
                # print(f"[1, {source}, {target}]")
                return
            hanoi_solution(n - 1, source, auxiliary, target, moves)
            # steps += 1
            moves.append([n, source, target])
            # print(f"[{n}, {source}, {target}]")
            hanoi_solution(n - 1, auxiliary, target, source, moves)

        hanoi_solution(N, 0, 2, 1, moves)
        example['answer'] = moves
        datas.append(example)

    return datas


class HanoiSimulator:
    def __init__(self, num_disks):
        self.num_disks = num_disks
        self.reset()

    def reset(self):
        """reset to initial state"""
        self.pegs = [
            list(range(self.num_disks, 0, -1)),  # Peg 0: [n, n-1,...,1]
            [],  # Peg 1: empty
            []  # Peg 2: empty
        ]
        self.move_history = []

    def validate_move(self, disk, from_peg, to_peg):
        """valid the move"""
        # 1. check source peg number
        if not (0 <= from_peg <= 2) or not (0 <= to_peg <= 2):
            return False, "Invalid peg number (must be 0-2)"

        # 2. check whether there exists disk in the source peg
        if not self.pegs[from_peg]:
            return False, "Source peg is empty"

        # 3. check whether the disk is in the top of source peg
        if self.pegs[from_peg][-1] != disk:
            return False, "Specified disk is not top of source peg"

        # 4. check whether the disk is greater that the top disk of the target peg
        if self.pegs[to_peg] and self.pegs[to_peg][-1] < disk:
            return False, "Cannot place larger disk on smaller disk"

        return True, "Valid move"

    def execute_move(self, disk, from_peg, to_peg):
        """single move"""
        valid, msg = self.validate_move(disk, from_peg, to_peg)
        if not valid:
            return False, msg

        moved_disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(moved_disk)
        self.move_history.append((disk, from_peg, to_peg))
        return True, "Move executed successfully"

    def validate_solution(self, moves):
        """valid total moves"""
        try:
            self.reset()

            for move in moves:
                if len(move) != 3:
                    return False, f"Invalid move format: {move}"

                disk, from_peg, to_peg = move
                success, msg = self.execute_move(disk, from_peg, to_peg)
                if not success:
                    return False, f"Invalid move {move}: {msg}"

            # check final state
            if (not self.pegs[0] and not self.pegs[1] and
                    self.pegs[2] == list(range(self.num_disks, 0, -1))):
                return True, "Solution is valid"
            else:
                return False, "Solution does not reach goal configuration"
        except:
            return False, "Solution does not reach goal configuration"

    def get_current_state(self):
        """get current state"""
        return [peg.copy() for peg in self.pegs]

