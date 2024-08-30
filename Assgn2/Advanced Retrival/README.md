# Advanced Retrival

## Query Expansion
This method mainly focuses on expanding the query to include more relevant terms. This uses LLM to generate queries that are similar to the original query and covers greater context. This method is useful when the original query is too short or ambiguous.

## Cross encoder re-ranking
This method uses cross encoder transformers to give a value between 0 and 1 for each pair of query and document representing how relevant the document is to the query. Based on this score, the documents are re-ranked and top-k documents are used for RAG.

## Embedding adapters
This method uses human feedback to modify the embeddings of the model to make it more relevant to the domain of knowledge. It first generates embeddings as it normally does for all the queries, then it trains a single layer neural network which takes input the embedding and outputs the changed embedding. This is trained on data collected using user feedback. It does it by taking the cosine similarity of the original embedding and the modified embedding, as well as the user feedback and tries to minimize the difference between the two.