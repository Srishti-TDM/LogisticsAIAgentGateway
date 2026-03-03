from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.schema import ChatRequest

class LangChainService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # these variables are gonna come from the UI
        self.userrequest = "Hi"
        self.domain = "Bye"
        self.lcrequest = ""

        self.prompt = ChatPromptTemplate.from_template(self.lcrequest)
    

    async def run(self, question: str, request: ChatRequest):

        chain = self.prompt | self.llm
        response = await chain.ainvoke({"question": request})
        lcresponse = response.content
        return lcresponse