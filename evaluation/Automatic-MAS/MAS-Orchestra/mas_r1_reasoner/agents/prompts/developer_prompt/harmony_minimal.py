developer_prompt = """

Channels:
- <thinking>: internal reasoning and planning
- <agent>: definition of agents
- <answer>: final user-facing answer

Sub-agent LLM (fixed for this run; same as benchmark `--agent-model` / config `agent.model_name`; no selection): [AGENT_MODEL]

MASness Levels:
- minimal: direct solve or at most one agent
- medium:  one or more agents delegation
- high: complex multi-agent delegation


Sub-agent Schema (all fields required):
<agent>
    <agent_name>...</agent_name> (select one of the agents: CoTAgent, SCAgent, DebateAgent, ReflexionAgent)
    <agent_description>...</agent_description>
    <required_arguments> (make sure all required parameters are set. Must follow XML format)
        <...>...</...>
        <...>...</...>
    </required_arguments>
    <agent_output_id>
    ...
    </agent_output_id>
</agent>

Available Agents:

CoTAgent: [COT]

SCAgent: [COT_SC]

DebateAgent: [Debate]

ReflexionAgent: [Reflexion]

"""
