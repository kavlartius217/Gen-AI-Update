import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.schema import SystemMessage, HumanMessage

# Page config
st.set_page_config(
    page_title="Restaurant Chat Assistant",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS for chat interface
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

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """👋 Welcome to Le Château! I'm your reservation assistant. 
            
How may I help you with your reservation today?"""
        }
    ]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'agent_executor' not in st.session_state:
    @st.cache_resource
    def initialize_agent():
        try:
            # Get API key from Streamlit secrets
            api_key = st.secrets["OPENAI_API_KEY"]
            
            llm = ChatOpenAI(api_key=api_key,temperature=0.3,model="gpt-4-0125-preview")
            embeddings = OpenAIEmbeddings(api_key=api_key)
            
            csv = CSVLoader("table_data (1).csv")
            csv = csv.load()
            
            rcts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = rcts.split_documents(csv)
            
            db = FAISS.from_documents(docs, embeddings)
            retriever = db.as_retriever()
            
            tool = create_retriever_tool(
                retriever,
                "table_information_tool",
                "has information about all the tables in the restaurant"
            )     
           
        prompt = ChatPromptTemplate.from_messages([
          SystemMessage(content="""You are a friendly restaurant host at Le Château. Follow these precise instructions:

1. Initial Greeting:
   - Keep it warm and natural: "Welcome to Le Château! I'd be happy to help you with a table. How many guests will be joining you, and what time would you like to dine?"

2. When Guest Provides Details:
   - Use the tool immediately to check availability
   - Present tables with richer descriptions:
     "For [party size] guests at [time], I can offer you:
     - Table #[number]: [detailed location description] (perfect for intimate dining/group conversation/etc.)
     - Table #[number]: [detailed location description] (mention any special features)
     Which table would you prefer?"

3. When Guest Selects a Table:
   - Acknowledge their choice with enthusiasm
   - Confirm specific details
   - End with: "Perfect! I've reserved Table #[number] for [party size] guests at [time]. Looking forward to welcoming you to Le Château!"

EXAMPLE PERFECT FLOW:
Guest: "hi"
You: "Welcome to Le Château! I'd be happy to help you with a table. How many guests will be joining you, and what time would you like to dine?"

Guest: "2 at 6pm"
You: "For 2 guests at 6 PM, I can offer you:
- Table #4: A lovely window table with garden views (perfect for intimate dining)
- Table #7: A cozy corner booth in our main dining room (offering more privacy)
Which table would you prefer?"

Guest: "table 4 sounds good"
You: "Perfect! I've reserved Table #4 for 2 guests at 6 PM. Looking forward to welcoming you to Le Château!"

IMPORTANT RULES:
- Always provide rich descriptions of table locations and features
- Acknowledge guest choices with enthusiasm
- Include table number in final confirmation
- Keep responses warm and personalized
- Never repeat questions
- Process information immediately"""),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
            agent = create_openai_tools_agent(llm, [tool], prompt)
            return AgentExecutor(agent=agent, llm=llm, tools=[tool], verbose=True)
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            return None
    
    st.session_state.agent_executor = initialize_agent()

# Sidebar with restaurant information
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
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """👋 Welcome to Le Château! I'm your reservation assistant. 
                
How may I help you with your reservation today?"""
            }
        ]
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
st.title("🍽️ Restaurant Reservation Chat")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">👤</div>
            <div class="message">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="avatar">🤖</div>
            <div class="message">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([4, 1])
    with cols[0]:
        user_input = st.text_input("Message", key="user_input", label_visibility="collapsed")
    with cols[1]:
        submit_button = st.form_submit_button("Send", use_container_width=True)

    if submit_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        if st.session_state.agent_executor:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent_executor.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    st.session_state.chat_history.append(user_input)
                    st.session_state.chat_history.append(response['output'])
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["output"]
                    })
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
        else:
            st.error("Agent not properly initialized. Please check your configuration.")
        
        st.rerun()
