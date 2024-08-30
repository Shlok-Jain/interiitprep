# Agentic RAG
This course was about devloping advanced agent based RAG, which can reason the user query more effectively, and can extract arguments from user prompts to generate more informed responses.

## Making RAG more interactive using arguments.
This notebook explains about how to make python functions runnable through RAG, and infer the arguments passed to the function from the user prompt.
For example:
If we can make a function like:
```python
def add(a:int, b:int):
    return a+b
```
We can convert this python function to llama-index tool.
This tools can be passed to llms while generating responses which require execution of the function.

We can also refer to explicit metdata related to chunked data (like page number etc.) and query them using RAG. 
For example: If you prompt it "Give me overview of page 2", then it can query the database such that it generates response based on page 2.

## Agent reasoning loops
In this notebook, we studied how to make LLM capable of advanced reasoning.
For example: 
If you ask LLM for querying something and summarizing the other thing in the same prompt, then it will divide the tasks in different sub tasks and run them sequentially to generate final output.
It also includes how to debug this kind of systems.

## Agent reasoning over large number of documents
We cannot apply the above method here due to limited context window of llms, So in this method we first run RAG to get top-k tools which will be useful for given tasks, and then we run above method on only these tools to get the final output.