import os
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools import PolygonLastQuote, PolygonTickerNews, PolygonFinancials, PolygonAggregates
import gradio as gr

import os, openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

os.environ["LANGCHAIN_PROJECT"] = "langgraph-financial-agent"


llm  = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment="gpt-4-32k",
    model="gpt-4-32k",
    streaming=True,
    temperature=0
)

prompt = hub.pull("hwchase17/openai-functions-agent")

polygon = PolygonAPIWrapper()
tools = [
    PolygonLastQuote(api_wrapper=polygon),
    PolygonTickerNews(api_wrapper=polygon),
    PolygonFinancials(api_wrapper=polygon),
    PolygonAggregates(api_wrapper=polygon)
]

from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish

# Define the agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
agent = RunnablePassthrough.assign(
    agent_outcome = agent_runnable
)

# Define the function to execute tools
def execute_tools(data):
    agent_action = data.pop('agent_outcome')
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    observation = tool_to_use.invoke(agent_action.tool_input)
    data['intermediate_steps'].append((agent_action, observation))
    return data

# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    else:
        return "continue"
    
# Define the Graph with nodes and edges
from langgraph.graph import END, Graph

workflow = Graph()
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "exit": END
    }
)
workflow.add_edge('tools', 'agent')
app = workflow.compile()

def financial_agent(input_text):
    result = app.invoke({"input": input_text, "intermediate_steps": []})
    output = result['agent_outcome'].return_values["output"]
    return output

ui = gr.Interface(
    fn=financial_agent,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs=gr.Markdown(),
    title="Financial Research Assistant",
    description="Financial Data Explorer: Leveraging Advanced API Tools for Market Insights"
)

ui.launch()