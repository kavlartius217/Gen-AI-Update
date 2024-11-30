import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Page config
st.set_page_config(
    page_title="Restaurant Chat Assistant",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
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

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Le Château!"}]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'reservation_state' not in st.session_state:
    st.session_state.reservation_state = {
        'status': 'initial',  # States: initial, awaiting_details, tables_offered, confirmed
        'guests': None,
        'time': None,
        'table': None,
        'offered_tables': [],
        'last_response': None
    }

def parse_reservation_request(text):
    """Parse guest count and time from reservation request"""
    text = text.lower()
    if 'table for' in text and 'at' in text:
        try:
            parts = text.split('at')
            guests_part = parts[0].split('table for')[1]
            guests = int(''.join(filter(str.isdigit, guests_part)))
            time = parts[1].strip()
            return {'guests': guests, 'time': time}
        except:
            return None
    return None

def extract_table_number(text, offered_tables):
    """Extract table number from selection message"""
    text = text.lower()
    if not offered_tables:
        return None
        
    if 'table' in text and any(str(table) in text for table in offered_tables):
        for table in offered_tables:
            if str(table) in text:
                return table
    return None

def initialize_agent():
    """Initialize the LangChain agent"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        
        llm = ChatOpenAI(
            api_key=api_key,
            temperature=0,
            model="gpt-4-0125-preview",
            max_tokens=150
        )
        
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Load and process table data
        loader = CSVLoader("table_data (1).csv")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        
        tool = create_retriever_tool(
            retriever,
            "table_information_tool",
            "Search for table information in the restaurant"
        )

        # Define the agent prompt
        prompt=ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a reservation chatbot that makes reservations based on the user input. Based on the user input and referencing the tool, suggest the user tables available at the specified time and the location of the respective tables. After the user makes a choice, confirm the reservation and make no further conversation"),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    MessagesPlaceholder(variable_name="chat_history")
])

        # Create and return the agent
        agent = create_openai_tools_agent(llm, [tool], prompt)
        return AgentExecutor(agent=agent, tools=[tool], verbose=True)
        
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

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
                    # Check for reservation request
                    reservation_details = parse_reservation_request(user_input)
                    if reservation_details:
                        st.session_state.reservation_state.update(reservation_details)
                        st.session_state.reservation_state['status'] = 'tables_offered'
                    
                    # Format chat history for context
                    formatted_history = []
                    for msg in st.session_state.chat_history[-4:]:  # Last 4 messages
                        if msg["role"] == "user":
                            formatted_history.append(HumanMessage(content=msg["content"]))
                        else:
                            formatted_history.append(AIMessage(content=msg["content"]))
                    
                    # Get agent response
                    response = st.session_state.agent_executor.invoke({
                        "input": user_input,
                        "chat_history": formatted_history
                    })
                    
                    # Process table selection
                    if st.session_state.reservation_state['status'] == 'tables_offered':
                        selected_table = extract_table_number(user_input, 
                                                           st.session_state.reservation_state['offered_tables'])
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
                        "content": response['output']
                    })
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["output"]
                    })
                    
                    st.session_state.reservation_state['last_response'] = response['output']
                    
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
        else:
            st.error("Agent not properly initialized. Please check your configuration.")
        
        st.rerun()

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Le Château!"}]
    st.session_state.chat_history = []
    st.session_state.reservation_state = {
        'status': 'initial',
        'guests': None,
        'time': None,
        'table': None,
        'offered_tables': [],
        'last_response': None
    }
    st.rerun()
