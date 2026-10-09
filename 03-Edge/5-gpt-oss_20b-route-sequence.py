from typing import TypedDict, Literal, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gpt-oss:20b",
)

class OverAllState(TypedDict):
    topic : str
    type : str
    poem : str
    joke : str
    music : str

def node_poem(state: OverAllState) -> OverAllState:
    poem = model.invoke([f"写一首关于{state['topic']}主题的诗词"]).content
    return {
        "poem": poem
    }

def node_joke(state: OverAllState) -> OverAllState:
    joke = model.invoke([f"写一个关于{state['topic']}主题的笑话"]).content
    return {
        "joke": joke
    }

def node_music(state: OverAllState) -> OverAllState:
    music = model.invoke([f"写一个关于{state['topic']}主题的歌曲"]).content
    return {
        "music": music
    }

# 路由选择函数
def route(state: OverAllState) -> Sequence[Literal["poem","joke", "music"]]:
    if "诗" in state["type"]: return ["poem", "music"]
    else: return ["joke", "music"]

builder = StateGraph(OverAllState)
builder.add_node(node_poem)
builder.add_node(node_joke)
builder.add_node(node_music)
builder.add_conditional_edges(START,route,path_map={
    "poem": "node_poem",
    "joke": "node_joke",
    "music": "node_music",
}) # 因为使用了Sequence，若不加path_map，生成的流程图会出错
builder.add_edge("node_joke",END)
builder.add_edge("node_poem",END)
builder.add_edge("node_music",END)
graph = builder.compile()

print(graph.invoke({"topic": "猫", "type": "诗词"}))
print(graph.get_graph().draw_mermaid())

png_bytes = graph.get_graph().draw_mermaid_png()
with open('5-graph.png', "wb") as f:
    f.write(png_bytes)

