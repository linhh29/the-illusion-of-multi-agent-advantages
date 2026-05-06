developer_prompt = """

Channels:
- <thinking>: internal reasoning and planning
- <agent>: definition of agents
- <edge>: definition of edges

Sub-agent LLM (fixed for this run; same as benchmark `--agent-model` / config `agent.model_name`; no selection): [AGENT_MODEL]

MASness Levels:
- medium: one or more agents delegation


Sub-agent Schema (all fields required):
<agent>
    <agent_id>...</agent_id> (a unique id for the agent)
    <agent_name>...</agent_name> (select one of the agents: CoTAgent, SCAgent, DebateAgent, ReflexionAgent)
    <agent_description>...</agent_description>
    <required_arguments> (make sure all required parameters are set. Must follow XML format)
        <...>...</...>
        <...>...</...>
    </required_arguments>
</agent>

Edge Schema (single block; all fields required. Each pair defines a directed link: output of <from> → input of <to>. List ALL links here; use exactly one <edge> block per solution):
<edge>
    <from>...</from> (the source agent_id)
    <to>...</to> (the target agent_id)
</edge>

Available Agents:

CoTAgent: [COT]

SCAgent: [COT_SC]

DebateAgent: [Debate]

ReflexionAgent: [Reflexion]

"""
