import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import List, Dict, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


llm=ChatGroq(model="qwen-2.5-32b")

class BlogGeneratorState(TypedDict):
    topic: str
    title: str
    content: str
    complete_blog: str





def title_generator_node(state: BlogGeneratorState):
    topic = state["topic"]
    
    # Use the LLM to generate a title
    title_prompt = f"Based on the following '{topic}', create an engaging, SEO-friendly blog post title"
    
    title_message = llm.invoke(title_prompt)
    title = title_message.content
    
    # Return updated state
    return {"title": title}


def content_writer(state: BlogGeneratorState):
    topic = state["topic"]
    title = state["title"]
    
    # Use the LLM to write content for this section
    writer_prompt = f"Write a detailed section for a blog post titled '{title}' with the topic '{topic}'."
    
    content_message = llm.invoke(writer_prompt)
    section_content = content_message.content
    
    # Return the section content
    return {"content": section_content}



def final_blog(state: BlogGeneratorState):
    title = state["title"]
    content = state["content"]

    # Combine all sections into a single blog post
    blog_post = f"# {title}\n\n {content}"
    
    return {"complete_blog": blog_post}
    
    
graph_builder = StateGraph(BlogGeneratorState)

# Add nodes
graph_builder.add_node("title_generator", title_generator_node)
graph_builder.add_node("content_writer", content_writer)
graph_builder.add_node("final_blog", final_blog)

# Add edges to define the flow
graph_builder.add_edge(START, "title_generator")
graph_builder.add_edge("title_generator", "content_writer")
graph_builder.add_edge("content_writer", "final_blog")
graph_builder.add_edge("final_blog", END)

# Compile the graph
graph = graph_builder.compile()


import streamlit as st

st.set_page_config(page_title="Blog Generator", page_icon="📹", layout="wide")

st.markdown('<p class="big-title">Blog Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate engaging blog posts, effortlessly! 🚀</p>', unsafe_allow_html=True)

st.markdown("### 🔗 Enter a Topic:")
user_input = st.text_input("", placeholder="Enter a topic for your blog post")

if user_input:
    initial_state = {"topic": user_input}
    result = graph.invoke(initial_state)
    st.write(result["complete_blog"])







