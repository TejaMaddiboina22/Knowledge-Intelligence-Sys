from langchain.chat_models import ChatOpenAI
from lanhchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config import Config

class LLMService:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=Config.OPENAI_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = ConversationalRetrievalChain.from_llm(llm=self.llm,retriever=vector_store, memory=self.memory)

    def get_response(self, query):
        try:                                        
            response = self.chain.run({"question": query})
            return response['answer']   
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "I encountered an error processing your request."