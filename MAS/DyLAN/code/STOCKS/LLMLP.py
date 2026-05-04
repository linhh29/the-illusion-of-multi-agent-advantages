# -*- coding: utf-8 -*-
"""LLMLP for STOCKS: qtype stocks, raw reply as answer (no early consensus)."""
import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_stocks_raw, most_frequent

ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1}


class LLMLP:
    def __init__(self, default_model_name, agents=4, agent_roles=None,
                 rounds=2, activation="listwise", qtype="stocks", mtype="gpt-4o"):
        agent_roles = agent_roles or []
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        # For stocks we never consider two raw replies equal -> no early consensus
        self.cmp_res = lambda x, y: False
        self.ans_parser = parse_stocks_raw
        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]
        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Only listwise activation is implemented")

    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question):
        def get_completions():
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2

        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(0, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        loop_indices = list(range(self.agents, self.agents * 2))
        random.shuffle(loop_indices)
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(1, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents * 2))
        for rid in range(2, self.rounds):
            if self.agents > 3:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[i] for i in indices]
                tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                idx_mask = [idxs[indices[x]] % self.agents for x in tops]
                resp_cnt += self.activation_cost

            loop_indices = list(range(self.agents * rid, self.agents * (rid + 1)))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    print(rid, idx)
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        completions = get_completions()
        # With cmp_res always False, most_frequent returns (first answer, 1) -> one raw reply
        return most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0], resp_cnt, completions, total_prompt_tokens, total_completion_tokens

    def backward(self, result):
        flag_last = False
        for rid in range(self.rounds - 1, -1, -1):
            if not flag_last:
                if len([idx for idx in range(self.agents * rid, self.agents * (rid + 1)) if self.nodes[idx].active]) > 0:
                    flag_last = True
                else:
                    continue
                matching = [idx for idx in range(self.agents * rid, self.agents * (rid + 1)) if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result)]
                ave_w = (1.0 / len(matching)) if matching else 0.0
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    if self.nodes[idx].active and (idx in matching or self.cmp_res(self.nodes[idx].get_answer(), result)):
                        self.nodes[idx].importance = ave_w
                    else:
                        self.nodes[idx].importance = 0
            else:
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance
        return [node.importance for node in self.nodes]
