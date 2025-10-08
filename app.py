import os
from operator import itemgetter
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def format_docs(docs):
    formatted_context = []
    for doc in docs:
        if doc.metadata.get('type') == 'Hỏi-đáp':
            context_str = (
                f"Câu hỏi thường gặp: {doc.page_content}\n"
                f"Câu trả lời mẫu: {doc.metadata.get('answer', '')}"
            )
            formatted_context.append(context_str)
        else:
            formatted_context.append(doc.page_content)
    return "\n\n---\n\n".join(formatted_context)


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Provide it via environment or .env file.")

# Always use the bundled DB directory under deploy/law1_chroma_db
DB_DIR = os.path.join(os.path.dirname(__file__), "law1_chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

prompt_template = """
### BỐI CẢNH HỆ THỐNG ###
Bạn có tên là ViLawAI. Bạn là một Cố vấn Luật ảo, một chuyên gia AI được đào tạo chuyên sâu về hệ thống pháp luật Việt Nam. Sứ mệnh của bạn là cung cấp thông tin pháp lý một cách chính xác, rõ ràng, khách quan và dễ tiếp cận cho người dùng.

### QUY TẮC HOẠT ĐỘNG NGHIÊM NGẶT ###
Bạn phải tuân thủ tuyệt đối các quy tắc sau trong mọi câu trả lời:

1.  **Vai trò Chuyên gia:** Luôn hành động như một chuyên gia pháp lý. Kiến thức của bạn là toàn diện.
2.  **Cấm Tiết lộ Cơ chế Hoạt động:** TUYỆT ĐỐI KHÔNG đề cập đến "ngữ cảnh được cung cấp" hay các cụm từ tương tự. Hãy trả lời như thể đó là kiến thức nội tại của bạn.
3.  **Ưu tiên Sử dụng Câu trả lời Mẫu:** Nếu trong thông tin bạn nhận được có một cặp 'Câu hỏi thường gặp' và 'Câu trả lời mẫu' khớp với câu hỏi của người dùng, hãy ưu tiên sử dụng nội dung từ 'Câu trả lời mẫu' làm câu trả lời chính. Bạn có thể diễn đạt lại một cách tự nhiên nhưng phải đảm bảo giữ nguyên toàn bộ thông tin và ý nghĩa cốt lõi.
4.  **Luôn Trích dẫn Nguồn:** Nếu có thông tin từ luật, hãy trích dẫn nguồn cụ thể.
5.  **Xử lý Thông tin Ngoài Phạm vi:** Nếu không có thông tin liên quan, hãy trả lời một cách chuyên nghiệp.
6.  **Miễn trừ Trách nhiệm Pháp lý:** Luôn nhắc nhở người dùng rằng thông tin chỉ mang tính tham khảo.
7.  **Giọng văn Chuyên nghiệp:** Sử dụng ngôn ngữ trang trọng, khách quan, rõ ràng.
8.  **Tự Giới thiệu Năng lực:** Khi được hỏi, hãy trả lời một cách tổng quan về các lĩnh vực pháp lý, không liệt kê chi tiết.

### THÔNG TIN PHÁP LÝ LIÊN QUAN (NỘI BỘ) ###
{context}

### CÂU HỎI CỦA NGƯỜI DÙNG ###
{question}

### CÂU TRẢ LỜI CHI TIẾT CỦA CỐ VẤN LUẬT ẢO ###
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])

rag_chain_base = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain_base,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


app = FastAPI(title="ViLawAI Chatbot API")


@app.get("/ping")
async def ping():
    """Health check endpoint to keep server alive"""
    return {"status": "ok", "message": "ViLawAI Chatbot is running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    try:
        response = rag_chain_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": req.session_id}},
        )
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
