"""
String embedded in generated Harmony ``forward`` code for sub-agent LLM id.

Replaced at execution time with ``global_node_model`` (see ``execution.execute_code``)
so phase-2 / benchmark can swap models without re-generating code.
"""

MAS_SUB_AGENT_MODEL_PLACEHOLDER = "__MAS_SUB_AGENT_MODEL__"
