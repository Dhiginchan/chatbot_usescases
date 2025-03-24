import os
import dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
dotenv.load_dotenv()

# Fetch API Key and Gemini Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize AI Model
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7, google_api_key=GOOGLE_API_KEY)

# Define Fashion Assistant Prompt Template
TEMPLATE =TEMPLATE = """
You are an advanced AI assistant with deep knowledge across various subjects. Your goal is to provide **accurate, detailed, and well-structured answers** to user questions.

 **Past User Conversations**:
{history}

 **User's Current Question**:
{input}

### **How to Answer**:
   - Prioritize **clarity and efficiency**—answer the user's concern **directly and professionally**.
   - If the issue is **technical**, provide **step-by-step troubleshooting steps**.
   - If the issue requires **further support**, suggest **help articles, contact options, or escalate**.
   - Always acknowledge frustration and provide **empathetic responses**.

📌 **Important Guidelines**:
- Be **concise yet informative**—avoid unnecessary details.
- If a question is outside AI knowledge, politely say **"I don’t have enough information on that, but here’s what I know…"**
- Always maintain a **professional, helpful, and neutral** tone.
"""


#  Create a Prompt Template
prompt = PromptTemplate.from_template(TEMPLATE)

from langchain.memory import ConversationBufferMemory

# Memory: Remembers everything in the conversation
chatbot_memory = ConversationBufferMemory(memory_key="history")

conversation = ConversationChain(
    llm=llm,
    memory=chatbot_memory,
    prompt=prompt
)


#  Chat Loop (Keeps running until user types "exit")
print("CUSTOMER SUPPORT ASSISTANT is ready! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("👋 Exiting CUSTOMER SUPPORT ASSISTANT . Have a good day!✨")
        break

    response = conversation.run(user_input)

    print("CUSTOMER SUPPORT ASSISTANT:", response)
