import chromadb
client = chromadb.PersistentClient(path="./storage")
collection = client.get_collection("rag_docs")
print("向量库文档数量：", collection.count())
if collection.count() > 0:
    docs = collection.get()
    print("第一个文档：", docs["documents"][0][:100])