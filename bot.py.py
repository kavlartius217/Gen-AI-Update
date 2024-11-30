import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from datetime import datetime
import re

# Page config
st.set_page_config(
    page_title="Restaurant Chat Assistant",
    page_icon="🍽️",
    layout="wide"
)

# Same CSS as before...
[Your existing CSS code here]

# Enhanced session state initialization
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to Le Château! How may I assist you today?"}]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'reservation_state' not in st.session_state:
        st.session_state.reservation_state = {
            'status': 'initial',
            'guests': None,
            'time': None,
            'date': None,
            'table': None,
            'offered_tables': [],
            'last_response': None,
            'confirmation_number': None
        }
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []

def validate_time_format(time_str):
    """Validate and standardize time format"""
    try:
        # Accept various time formats and standardize to 24-hour
        time_str = time_str.lower().strip()
        if 'am' in time_str or 'pm' in time_str:
            time_obj = datetime.strptime(time_str, '%I:%M %p')
        else:
            time_obj = datetime.strptime(time_str, '%H:%M')
        return time_obj.strftime('%H:%M')
    except ValueError:
        return None

def parse_reservation_request(text):
    """Enhanced parsing of reservation requests"""
    text = text.lower()
    reservation_info = {}
    
    # Extract guest count
    guests_pattern = r'(?:table|reservation|booking)\s+(?:for\s+)?(\d+)(?:\s+people|\s+persons|\s+guests)?'
    guests_match = re.search(guests_pattern, text)
    if guests_match:
        reservation_info['guests'] = int(guests_match.group(1))
    
    # Extract time
    time_pattern = r'(?:at|for|around)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)'
    time_match = re.search(time_pattern, text)
    if time_match:
        time_str = time_match.group(1)
        validated_time = validate_time_format(time_str)
        if validated_time:
            reservation_info['time'] = validated_time
    
    return reservation_info if reservation_info else None

def extract_table_number(text, offered_tables):
    """Enhanced table number extraction"""
    text = text.lower()
    if not offered_tables:
        return None
    
    # Look for explicit table numbers
    table_pattern = r'table\s*(?:number|#)?\s*(\d+)'
    match = re.search(table_pattern, text)
    if match:
        table_num = int(match.group(1))
        if table_num in offered_tables:
            return table_num
    
    # Look for any mentioned numbers that match offered tables
    numbers = re.findall(r'\d+', text)
    for num in numbers:
        if int(num) in offered_tables:
            return int(num)
    
    return None

def initialize_agent():
    """Initialize the LangChain agent with improved prompt and error handling"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        
        llm = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,  # Reduced temperature for more consistent responses
            max_tokens=200,
            model="gpt-4"  # Using GPT-4 for better comprehension
        )
        
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Load and process table data
        loader = CSVLoader("table_data.csv")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant results
        )
        
        tool = create_retriever_tool(
            retriever,
            "table_information_tool",
            "Search for available tables that match the reservation requirements"
        )

        # Improved system prompt
        system_prompt = """You are a professional restaurant reservation assistant for Le Château. Your primary responsibilities are:

1. Handle reservation requests by checking table availability using the table_information_tool
2. Provide clear, concise responses without repetition
3. Maintain context of the conversation
4. Only suggest tables that match the party size and time requirements
5. Confirm reservations with a unique confirmation number

Guidelines:
- Always verify table availability before making suggestions
- Don't repeat previously provided information
- If a request is unclear, ask for specific missing details
- Once a reservation is confirmed, provide a summary and end the conversation
- Don't ask unnecessary follow-up questions after confirmation

Current restaurant hours:
Mon-Thu: 11:00 AM - 10:00 PM
Fri-Sat: 11:00 AM - 11:00 PM
Sunday: 10:00 AM - 9:00 PM"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_tools_agent(llm, [tool], prompt)
        return AgentExecutor(agent=agent, tools=[tool], verbose=True)
        
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def process_response(response, reservation_state):
    """Process and clean LLM response"""
    # Remove repetitive phrases
    cleaned_response = re.sub(r'(?i)let me check[^.]*\.', '', response)
    cleaned_response = re.sub(r'(?i)is there anything else[^?]*\?', '', cleaned_response)
    
    # Extract offered tables if present
    table_matches = re.findall(r'table (\d+)', cleaned_response.lower())
    if table_matches:
        reservation_state['offered_tables'] = [int(t) for t in table_matches]
    
    # Generate confirmation number if reservation is confirmed
    if 'confirm' in cleaned_response.lower() and reservation_state['table'] and not reservation_state['confirmation_number']:
        reservation_state['confirmation_number'] = f"LC{datetime.now().strftime('%Y%m%d%H%M')}"
        cleaned_response += f"\n\nYour confirmation number is: {reservation_state['confirmation_number']}"
    
    return cleaned_response.strip()

# Main chat interface code...
[Rest of your existing main chat interface code, updated to use the new functions]
