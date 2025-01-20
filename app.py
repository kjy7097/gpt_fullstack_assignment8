import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import os

os.environ["OPENAI_API_KEY"] = "dummy_api_key"

answers_prompt = ChatPromptTemplate.from_template(
    """Using ONLY the following context answer the user's question.
If you can't just say you don't know, don't make anything up.

Then, give a score to the answer between 0 and 5.
The score should be high if the answer is related to the user's question, and low otherwise.
If there is no relevant content, the score is 0.
Always provide scores with your answers
Context: {context}

Examples:

Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know
Score: 0

Your turn!
Question: {question}
"""
)

st.set_page_config(page_title="Site GPT", page_icon="🌏")
st.title("Site GPT")

with st.sidebar:
    st.page_link(
        page="https://github.com/kjy7097/gpt_fullstack_assignment8.git",
        label="Click! to go Github Repo.",
    )
    with st.form("api_form"):
        col1, col2 = st.columns(2)
        api_key = col1.text_input(
            "Enter OpenAI API Key....",
        )
        col2.form_submit_button("Apply")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Use ONLY the following pre-existing answers to answer the user's question.
Use the answers that have the highest score (more helpful) and favor the most recent ones.
Cite sources and return the sources of the answers as they are, do not change them.
Answers: {answers}
""",
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", "")


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/vectorize\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.markdown(
    """
    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)


if api_key:
    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")
    query = st.text_input("Ask a question to the website")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.write(result.content.replace("$", "\$"))
