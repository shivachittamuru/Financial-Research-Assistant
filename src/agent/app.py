
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools import PolygonLastQuote, PolygonTickerNews, PolygonFinancials, PolygonAggregates

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

os.environ["LANGCHAIN_PROJECT"] = "financial-agent"


llm  = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment="gpt-4-32k",
    model="gpt-4-32k",
    streaming=True,
    temperature=0
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        As a Financial Research Assistant, your role is to provide latest financial insights about companies and help individuals make informed investment decisions. 
        
        You have access to 4 tools from Polygon.io API that lets you query the latest financial data from all US stock exchanges. The tools are:
        1. polygon_last_quote: Get the latest quote for ticker
        2. polygon_ticker_news: Get the latest news for ticker
        3. polygon_financials: Get the financials for ticker
        4. polygon_aggregates: Get the aggregates for ticker   

        You may need one or more tools to answer a user query. You can use the tools in any order depending on the user query. 
        
        Conversaion History:
        {chat_history}
        """                 
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "User Input: {input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

polygon = PolygonAPIWrapper()
tools = [
    PolygonLastQuote(api_wrapper=polygon),
    PolygonTickerNews(api_wrapper=polygon),
    PolygonFinancials(api_wrapper=polygon),
    PolygonAggregates(api_wrapper=polygon)
]

import random
from langchain_community.chat_message_histories.cosmos_db import CosmosDBChatMessageHistory
cosmos = CosmosDBChatMessageHistory(
    cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
    cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
    cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
    connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
    session_id="shiva-test" + str(random.randint(1, 10000)),
    user_id="shivac"
    )
cosmos.prepare_cosmos()

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(k=10, return_messages=True,memory_key="chat_history",chat_memory=cosmos)

# Define the agent
from langchain.agents import create_openai_functions_agent, AgentExecutor

agent_runnable = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=False, memory=memory) 

result = agent_executor.invoke({"input": "Give me the top news about SBUX stock?"})
output = result["output"]
print(output)