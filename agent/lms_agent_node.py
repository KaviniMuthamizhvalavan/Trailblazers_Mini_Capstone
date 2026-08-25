import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from agent.llm import get_llm
from agent.lms_tool import fetch_learner_history

LMS_AGENT_SYSTEM_PROMPT = """You are a student learning path advisor. 
You have access to a tool `fetch_learner_history` to retrieve a learner's completed and currently enrolled courses from the database.

The learner ID is: "{learner_id}"

Your job is to decide whether you need to retrieve this learner's course history from the database to avoid recommending courses they have already taken or are currently enrolled in.
If a learner ID is provided, you MUST call `fetch_learner_history` with that learner ID.
If no learner ID is provided, or if the learner ID is empty or not known, do NOT call the tool; just explain that no history is needed or available.
"""

@traceable(name="lms_agent_node", run_type="chain")
def lms_agent_node(state: dict) -> dict:
    """
    Gives the LLM the option to call fetch_learner_history if it judges
    that it needs the learner's course history.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools([fetch_learner_history])
    
    # Gather state info
    learner_id = state.get("learner_id", "")
    messages = state.get("messages", [])
    
    # Build prompt
    system_message = SystemMessage(content=LMS_AGENT_SYSTEM_PROMPT.format(learner_id=learner_id))
    
    # Invoke the LLM with system message + context
    response = llm_with_tools.invoke([system_message] + messages)
    
    return {"messages": [response]}


# Prebuilt ToolNode wrapping the fetch_learner_history tool
tool_node = ToolNode([fetch_learner_history])


def lms_tool_extractor_node(state: dict) -> dict:
    """
    Runs after lms_tools execution. Inspects the messages list for the latest ToolMessage
    from 'fetch_learner_history', parses its JSON content, and updates the state
    with 'lms_completed' and 'lms_enrolled'.
    """
    messages = state.get("messages", [])
    lms_completed = []
    lms_enrolled = []
    
    # Search backwards for the latest ToolMessage from fetch_learner_history
    for msg in reversed(messages):
        if msg.type == "tool" and msg.name == "fetch_learner_history":
            try:
                # Parse JSON or string representation of dict
                content_str = msg.content.strip()
                # Replace single quotes with double quotes for valid JSON parsing if needed
                if "'" in content_str and '"' not in content_str:
                    content_str = content_str.replace("'", '"')
                data = json.loads(content_str)
                lms_completed = data.get("completed", [])
                lms_enrolled = data.get("enrolled", [])
            except Exception as e:
                print(f"Error parsing ToolMessage: {e}")
            break
            
    return {
        "lms_completed": lms_completed,
        "lms_enrolled": lms_enrolled,
    }
