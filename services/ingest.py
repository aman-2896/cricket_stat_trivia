"""
Ingestion pipeline for the "Laws of Cricket" reference document (.docx),
plus the hybrid (vector + keyword) retrieval used to answer rules
questions in services/agent.py and mcp/mcp_server.py.

Pipeline: clean_normalize_law_file_content() extracts and trims the raw
text -> split_chunks_embed_store() chunks + embeds + stores it in Postgres
(pgvector via PGVector) -> extract_laws_images()/get_image_description()/
ingest_image_content() do the same for diagrams embedded in the docx,
using a vision LLM to turn each image into a text description ->
get_hybrid_retrieval_results() combines vector similarity search with
BM25 (via ParadeDB's paradedb.score) for retrieval at query time.
"""

from langchain_community.document_loaders import Docx2txtLoader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from config.config import DB_URL
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
import zipfile
from pathlib import Path
import base64
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from db.db import get_connection
document_path='./documents/rules_docx/laws_of_cricket.docx'

def clean_normalize_law_file_content():
    """
    Extract the Laws of Cricket text from the .docx and trim it down to
    just the actual laws (from "LAW 1 THE PLAYERS" up to the "Preface"
    section that follows them in this document), collapsing extra
    blank lines/whitespace left over from the original formatting.
    """
    loader=Docx2txtLoader(document_path)
    documents=loader.load()
    content=documents[0].page_content
    content=content.replace("\t"," ")
    content=content.strip()
    # Cut everything before the first law (title pages, table of contents).
    law_1=content.find("LAW 1 THE PLAYERS")
    content=content[law_1:]
    # This document's preface section comes after the laws text, not
    # before, so cut everything from "Preface" onward too.
    preface=content.find("Preface")
    content=content[:preface-1]
    # Collapse runs of blank lines / whitespace-only lines down to a
    # single newline so the text splits into clean chunks later.
    content=re.sub(r'\n{2,}','\n',content)
    content = re.sub(r'(?:\r?\n)[ \t]*(?:(?:\r?\n)[ \t]*)+', '\n', content)
    return content



def split_chunks_embed_store():
    """
    Chunk the cleaned Laws of Cricket text, embed each chunk, and store the
    embeddings + text in the pgvector-backed "cricket_laws" collection so
    they're searchable at query time.
    """
    content=clean_normalize_law_file_content()
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        length_function=len
    )
    chunks=splitter.split_text(content)
    store=PGVector(
        embeddings=embeddings,
        connection=DB_URL,
        collection_name="cricket_laws"
    )
    # Skip the first few chunks — leftover front-matter/heading noise that
    # survived the cleanup in clean_normalize_law_file_content().
    texts=chunks[5:]
    metadatas=[{"source":"laws_of_cricket"} for _ in texts]
    store.add_texts(texts=texts,metadatas=metadatas)

def extract_laws_images():
    """
    Pull every embedded diagram out of the .docx (a .docx is just a zip
    archive; images live under word/media/) and save them to disk so
    get_image_description() can describe each one with a vision LLM.
    """
    with zipfile.ZipFile(document_path) as z:
        for name in z.namelist():
            if name.startswith("word/media"):
                file_name=name.split("/")[2]
                # file_name=file_name_temp.split(".")[0]
                # file_extension=file_name_temp.split(".")[1]
                img_bytes=z.read(name)
                with open(f'./documents/rules_docx/images/{file_name}','wb') as img:
                    img.write(img_bytes)


def get_image_description():
    """
    For every extracted diagram image, ask a vision-capable LLM to
    transcribe its labels/measurements into text, so field/equipment
    dimensions shown only as diagrams are still searchable as text.
    """
    document_list=[]
    prompt="""
            This is a diagram from the official Laws of Cricket. Describe in detail what this diagram depicts, and transcribe all measurements, dimensions, labels, and any text shown in the image. Be precise and complete with numbers and units, as this description will be used to answer factual questions about cricket equipment and field dimensions. Do not add/invent
            any unrelated facts or figures. 
        """
    llm=ChatOpenAI(model="gpt-4o-mini")
    path=Path('./documents/rules_docx/images')
    for img in path.iterdir():
        # print(img.name)
        img_bytes=open(img,"rb").read()
        # Vision LLMs here take images as base64 data URLs rather than
        # file paths/uploads, so we encode each image inline.
        b64=base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/{img.suffix};base64,{b64}"
        message=HumanMessage(
            content=[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":data_url}}
            ]
        )
        result=llm.invoke([message])
        output=result.content
        document_list.append(Document(page_content=output,metadata={"source": "laws_of_cricket", "type": "image", "image_file": img.name}))
        # print(f'image name -> {img.name}')
        # print(f'\n{output}')
        # print('-----------------------------------------------------------------\n')
    return document_list

def ingest_image_content(image_documents):
    # Simple/low-complexity: embeds and stores the already-generated image
    # descriptions into the same "cricket_laws" collection as the text
    # chunks, so both are searched together at query time.
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    store=PGVector(
        embeddings=embeddings,
        connection=DB_URL,
        collection_name="cricket_laws"
    )
    store.add_documents(image_documents)


def text_similarity_search(query,k=5):
    # Simple/low-complexity: dense vector (embedding) similarity search
    # over the "cricket_laws" collection, returning (chunk_id, distance)
    # pairs — good at matching meaning/paraphrasing.
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    store=PGVector(
        embeddings=embeddings,
        connection=DB_URL,
        collection_name="cricket_laws"
    )

    search_results = store.similarity_search_with_score(query, k)

    results=[]
    for r in search_results:
        results.append((r[0].id,r[1]))
    return results

def bm_25_search(query,k=5):
    # Keyword/full-text search using ParadeDB's BM25 index (`@@@` operator,
    # paradedb.score) over the same embedding table — good at matching
    # exact terms/numbers that dense vector search can miss (e.g. "22 yards").
    connection=get_connection()
    cursor=connection.cursor()

    sql="""
        SELECT id,paradedb.score(id) as score from langchain_pg_embedding
        where document @@@ %s
        ORDER BY score DESC
        LIMIT %s;
        """
    cursor.execute(sql,(query,k))
    rows=cursor.fetchall()
    cursor.close()
    connection.close()
    return rows

def reciprocal_rank_fusion(result_lists,k=60):
    """
    Merge multiple ranked result lists (here: vector search + BM25) into a
    single ranking using Reciprocal Rank Fusion: each item's score is the
    sum of 1/(k + rank) across every list it appears in, so items ranked
    highly by *either* method rise to the top without needing the two
    methods' raw scores to be on the same scale.
    """
    scores={}
    for result_list in result_lists:
        for rank,id_score_tuple in enumerate(result_list,start=1):
            # print(f"rank {rank} id_score_tuple {id_score_tuple} and {type(id_score_tuple)}")
            scores[id_score_tuple[0]]=scores.get(id_score_tuple[0],0)+1/(k+rank)
    return sorted(scores,key=scores.get,reverse=True)
    # print(scores)
    # return results

def get_search_results(user_query):
    # Simple/low-complexity: runs both retrieval methods and fuses their
    # rankings into one ordered list of chunk IDs.
    text_similarity_results=text_similarity_search(user_query,3)
    bm_25_search_results=bm_25_search(user_query,3)
    results = reciprocal_rank_fusion([text_similarity_results,bm_25_search_results])
    return results

def get_hybrid_retrieval_results(user_query,k=3):
    """
    Public entry point used by the agent/MCP tools: run hybrid (vector +
    BM25) retrieval for a rules question, fetch the matching chunks'
    full text, and format them into a single string (in fused-ranking
    order) ready to hand to an LLM as context.
    """
    retrieved_ids_list = get_search_results(user_query)

    query = """
        SELECT id, document, cmetadata
        FROM langchain_pg_embedding
        WHERE id = ANY(%s) LIMIT %s;
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(query, (retrieved_ids_list,k))
    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    # Create lookup: id -> database row
    by_id = {row[0]: row for row in rows}

    content = ""

    # Follow the RRF ranking (not the DB's arbitrary row order) so the
    # highest-fused chunks appear first in the returned context string.
    for doc_id in retrieved_ids_list:

        if doc_id not in by_id:
            continue

        row = by_id[doc_id]

        source = row[2].get("source", "")
        source_type = row[2].get("type", "")
        doc_text = row[1]

        content += f"Source - {source}"

        if source_type:
            content += f" Type - {source_type}"

        content += f"\n{doc_text}"

        content += "\n---------------------------------------------------------------------------------------\n"

    return content


if __name__=="__main__":
    # text_similarity_results=text_similarity_search("what is the height of the stumps in cricket?",5)
    # bm_25_search_results=bm_25_search("what is the height of the stumps in cricket?",5)

    # results=reciprocal_rank_fusion([text_similarity_results,bm_25_search_results])
    # print(results)
    results=get_hybrid_retrieval_results("what is the height of the stumps in cricket?")
    print(results)
