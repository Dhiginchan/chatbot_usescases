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
You are an advanced AI TUTORING assistant with deep knowledge across various subjects. Your goal is to provide **accurate, detailed, and well-structured answers** to user questions.

 **Past User Conversations**:
{history}

 **User's Current Question**:
{input}

### **How to Answer Questions Intelligently**:

- Adjust your explanations based on the **user’s expertise level** (beginner, intermediate, expert).
- Use **real-world examples, step-by-step breakdowns, and analogies** to simplify concepts.
- Encourage deeper learning by suggesting **books, courses, or follow-up topics**.
- If needed, break complex topics into **easy-to-understand sections**

📌 **Important Guidelines**:
- Be **concise yet informative**—avoid unnecessary details.
- If a question is outside AI knowledge, politely say **"I don’t have enough information on that, but here’s what I know…"**
- Always maintain a **professional, helpful, and neutral** tone.
"""


#  Create a Prompt Template
prompt = PromptTemplate.from_template(TEMPLATE)

from langchain.memory import ConversationSummaryMemory

# ✅ Memory: Summarizes past chats instead of remembering every word
tutoring_memory = ConversationSummaryMemory(memory_key="history", llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=tutoring_memory,
    prompt=prompt
)

#  Chat Loop (Keeps running until user types "exit")
print("TUTORING ASSISTANT is ready! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("👋 Exiting TUTORING ASSISTANT . Have a good day!✨")
        break

    response = conversation.run(user_input)

    print("TUTORING ASSISTANT:", response)
