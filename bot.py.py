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

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        margin-bottom: 50px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #F0F2F6;
    }
    .chat-message.assistant {
        background-color: #E8F0FE;
    }
    .chat-message .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        margin-right: 0.8rem;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    div[data-testid="stForm"] {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 600px;
        background-color: white;
        padding: 0.5rem 1rem;
        z-index: 100;
        border-top: 1px solid #ddd;
        display: flex;
        gap: 8px;
    }
    div[data-testid="stForm"] > div {
        display: flex;
        gap: 8px;
        width: 100%;
    }
    div[data-testid="stTextInput"] div {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    button[kind="primaryFormSubmit"] {
        height: 35px !important;
        margin-top: 0 !important;
        padding: 0 20px !important;
    }
    .stTextInput input {
        height: 35px !important;
    }
    section[data-testid="stSidebar"] {
        width: 300px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Welcome to Le Château. How may I assist you today?"}]
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

def validate_time_format(time_str):
    """Validate and standardize time format"""
    try:
        time_str = time_str.lower().strip()
        if 'am' in time_str or 'pm' in time_str:
            time_obj = datetime.strptime(time_str, '%I:%M %p')
        else:
            time_obj = datetime.strptime(time_str, '%H:%M')
        return time_obj.strftime('%H:%M')
    except ValueError:
        return None

def parse_reservation_request(text):
    """Parse reservation details from user input"""
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
    
    # Extract date if present
    date_pattern = r'(?:on|for)\s+([a-zA-Z]+\s+\d{1,2}(?:st|nd|rd|th)?)'
    date_match = re.search(date_pattern, text)
    if date_match:
        reservation_info['date'] = date_match.group(1)
    
    return reservation_info if reservation_info else None

def extract_table_number(text, offered_tables):
    """Extract table number from user input"""
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
    """Initialize the LangChain agent with improved prompt"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        
        llm = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,
            max_tokens=200,
            model="gpt-4"
        )
        
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Load and process table data
        loader = CSVLoader("table_data (1).csv")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        tool = create_retriever_tool(
            retriever,
            "table_information_tool",
            "Search for available tables that match the reservation requirements. Use this to check specific table availability."
        )

        system_prompt = """You are a warm and professional restaurant reservation assistant for Le Château. Follow these guidelines carefully:

1. INITIAL GREETING:
- Respond warmly to general greetings without asking for reservation details
- For greetings like "hi" or "hello", respond with "Hello! Welcome to Le Château. How may I assist you today?"

2. HANDLING RESERVATION REQUESTS:
- When given partial information (like only time or party size), acknowledge what was provided and ask specifically for missing details
- For "2 guests at 6pm", respond: "I'd be happy to help you with a reservation for 2 guests at 6:00 PM. What date would you like to dine with us?"
- Always verify table availability using table_information_tool before suggesting options

3. RESPONSE GUIDELINES:
- Keep responses concise and natural
- Don't repeat previous questions
- Use conversational language
- Only mention missing information once

4. RESERVATION CONFIRMATION:
- Once all details are provided, check availability and suggest specific tables
- After table selection, provide a clear confirmation with all details
- Generate a unique confirmation number only for completed reservations

Restaurant Hours:
Mon-Thu: 11:00 AM - 10:00 PM
Fri-Sat: 11:00 AM - 11:00 PM
Sunday: 10:00 AM - 9:00 PM

Remember: Be warm and professional, avoid repetition, and keep the conversation flowing naturally."""

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

# Initialize session state
init_session_state()

# Initialize agent if not already done
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = initialize_agent()

# Sidebar
with st.sidebar:
    st.header("🏰 Le Château")
    
    with st.expander("📍 Hours & Location"):
        st.markdown("""
        **Hours of Operation:**
        - Mon-Thu: 11:00 AM - 10:00 PM
        - Fri-Sat: 11:00 AM - 11:00 PM
        - Sunday: 10:00 AM - 9:00 PM
        
        **Address:**
        123 Gourmet Street
        Foodie City, FC 12345
        
        **Contact:**
        📞 (555) 123-4567
        """)
    
    with st.expander("ℹ️ Reservation Policy"):
        st.markdown("""
        - Reservations recommended for parties of all sizes
        - Maximum party size: 20 people
        - Special events and private dining available
        - 15-minute grace period for late arrivals
        - Cancellations accepted up to 4 hours before reservation
        """)

# Main chat interface
st.title("🍽️ Restaurant Reservation Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([4, 1])
    with cols[0]:
        user_input = st.text_input("Message", key="user_input", label_visibility="collapsed")
    with cols[1]:
        submit_button = st.form_submit_button("Send", use_container_width=True)

    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        if st.session_state.agent_executor:
            with st.spinner("Thinking..."):
                try:
                    # Parse reservation request
                    reservation_details = parse_reservation_request(user_input)
                    if reservation_details:
                        st.session_state.reservation_state.update(reservation_details)
                        st.session_state.reservation_state['status'] = 'tables_offered'
                    
                    # Format chat history
                    formatted_history = []
                    for msg in st.session_state.chat_history[-4:]:
                        if msg["role"] == "user":
                            formatted_history.append(HumanMessage(content=msg["content"]))
                        else:
                            formatted_history.append(AIMessage(content=msg["content"]))
                    
                    # Get agent response
                    response = st.session_state.agent_executor.invoke({
                        "input": user_input,
                        "chat_history": formatted_history
                    })
                    
                    # Process response
                    cleaned_response = process_response(
                        response['output'], 
                        st.session_state.reservation_state
                    )
                    
                    # Check for table selection
                    if st.session_state.reservation_state['status'] == 'tables_offered':
                        selected_table = extract_table_number(
                            user_input, 
                            st.session_state.reservation_state['offered_tables']
                        )
                        if selected_table:
                            st.session_state.reservation_state['table'] = selected_table
                            st.session_state.reservation_state['status'] = 'confirmed'
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": cleaned_response
                    })
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": cleaned_response
                    })
                    
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
        else:
            st.error("Agent not properly initialized. Please check your configuration.")
        
        st.rerun()

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Welcome to Le Château. How may I assist you today?"}]
    st.session_state.chat_history = []
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
    st.rerun()
