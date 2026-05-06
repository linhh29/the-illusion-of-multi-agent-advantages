# -*- coding: utf-8 -*-
"""LLM Neuron for SMFR: uses raw reply as answer (no single-choice parsing)."""
import random
import re
from utils import parse_smfr_raw, generate_answer
from prompt_lib import ROLE_MAP, construct_ranking_message, construct_message, SYSTEM_PROMPT_SMFR


class LLMNeuron:
    def __init__(self, role, mtype="gpt-4o", ans_parser=parse_smfr_raw, qtype="smfr"):
        self.role = role
        self.mtype = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.to_edges = []
        self.from_edges = []
        self.question = None

        if mtype in ["gpt-4o", "gpt-5", "gpt-5-mini"]:
            self.model = mtype
        else:
            raise NotImplementedError(f"Unsupported model: {mtype}")

        def find_array(text):
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                last_match = matches[-1].replace(' ', '')
                try:
                    return list(map(lambda x: int(x) if x.strip().isdigit() else 0, last_match.split(',')))
                except Exception:
                    return []
            return []
        self.weights_parser = find_array

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def activate(self, question):
        self.question = question
        self.active = True
        contexts, formers = self.get_context()
        random.shuffle(formers)
        formers = [mess[0] for mess in formers]
        contexts.append(construct_message(formers, question, self.qtype))
        self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
        self.answer = self.ans_parser(self.reply)
        weights = self.weights_parser(self.reply)
        if len(weights) != len(formers):
            weights = [0 for _ in range(len(formers))]
        for eid, edge in enumerate(self.from_edges):
            if eid < len(weights):
                w = weights[eid]
                edge.weight = w / 5 if 0 < w <= 5 else (1 if w > 5 else 0)
            else:
                edge.weight = 0
        total = sum(e.weight for e in self.from_edges)
        if total > 0:
            for edge in self.from_edges:
                edge.weight /= total
        else:
            for edge in self.from_edges:
                edge.weight = 1.0 / len(self.from_edges)

    def get_context(self):
        if self.qtype == "smfr":
            sys_prompt = ROLE_MAP.get(self.role, ROLE_MAP["Assistant"]) + "\n" + SYSTEM_PROMPT_SMFR
        else:
            raise NotImplementedError("Unsupported qtype")
        contexts = [{"role": "system", "content": sys_prompt}]
        formers = [(edge.a1.reply, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers


class LLMEdge:
    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0


def parse_ranks(completion, max_num=4):
    content = completion
    pattern = r'\[([1234567]),\s*([1234567])\]'
    matches = re.findall(pattern, content)
    try:
        match = matches[-1]
        tops = [int(match[0]) - 1, int(match[1]) - 1]
        tops = [max(0, min(x, max_num - 1)) for x in tops]
    except Exception:
        tops = random.sample(list(range(max_num)), 2)
    return tops


def listwise_ranker_2(responses, question, qtype, model="gpt-4o"):
    if model not in ["gpt-4o", "gpt-5", "gpt-5-mini"]:
        raise NotImplementedError(f"Unsupported model: {model}")
    assert len(responses) > 2
    message = construct_ranking_message(responses, question, qtype)
    completion, prompt_tokens, completion_tokens = generate_answer([message], model)
    return parse_ranks(completion, max_num=len(responses)), prompt_tokens, completion_tokens
