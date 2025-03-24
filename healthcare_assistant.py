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
You are an advanced AI Health care Assistant with deep knowledge across various subjects. Your goal is to provide **accurate, detailed, and well-structured answers** to user questions.

 **Past User Conversations**:
{history}

 **User's Current Question**:
{input}

### **How to Answer Questions Intelligently**:

- Always provide **general guidance** but **avoid giving direct medical diagnoses**.
- Use **scientific data** and **verified sources** for health-related answers.
- Encourage users to **consult a doctor for serious medical concerns**.
- Offer **self-care tips, symptom explanations, and wellness recommendations**.

📌 **Important Guidelines**:
- Be **concise yet informative**—avoid unnecessary details.
- If a question is outside AI knowledge, politely say **"I don’t have enough information on that, but here’s what I know…"**
- Always maintain a **professional, helpful, and neutral** tone.
"""


#  Create a Prompt Template
prompt = PromptTemplate.from_template(TEMPLATE)

from langchain.memory import ConversationBufferWindowMemory

# ✅ Memory: Remembers only the last 5 messages to keep things focused
healthcare_memory = ConversationBufferWindowMemory(memory_key="history", k=5)

conversation = ConversationChain(
    llm=llm,
    memory=healthcare_memory,
    prompt=prompt
)

print("Health care Assistant is ready! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("👋 Exiting Health care Assistant . Have a good day!✨")
        break

    response = conversation.run(user_input)

    print("THealth care Assistant:", response)
