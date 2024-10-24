import glob
import copy
import os
import pickle
import time
import torch
import torch.nn.functional as F
import numpy as np
np.random.seed(42)

import torch
import json
import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
import src.normalize_text
import argparse

class Retriever:
    def __init__(self, data_path, model_name_or_path, device):
        parser = argparse.ArgumentParser()
        parser.add_argument("--passages", type=str, default=f"{data_path}/psgs_w100.tsv", help="Path to passages (.tsv file)")
        parser.add_argument("--passages_embeddings", type=str, default=f"{data_path}/wikipedia_embeddings/*", help="Glob path to encoded passages")
        parser.add_argument(
            "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
        )
        parser.add_argument(
            "--model_name_or_path", type=str, default=model_name_or_path, help="path to directory containing model weights and config file"
        )
        self.validation_workers = 32 # "Number of parallel processes to validate results"
        self.per_gpu_batch_size = 256

        self.no_fp16 = True # inference in fp32
        self.indexing_batch_size = 1000000 # Batch size of the number of passages indexed

        self.projection_size = 768
        self.n_subquantizers = 0 # "Number of subquantizer used for vector quantization, if 0 flat index is used"
        self.n_bits = 8 # Number of bits per subquantizer
        self.lang = "+"
        self.lowercase = False # "lowercase text before encoding"
        self.normalize_text = False # "lowercase text before encoding"
        self.workers_num = 16
        self.question_maxlength = 512

        args = parser.parse_args()

        print(f"Loading model from: {args.model_name_or_path}")
        model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
        model.eval()
        model = model.cuda()
        if not self.no_fp16:
            model = model.half()
        self.model, self.tokenizer = model, tokenizer
        self.index = src.index.Indexer(self.projection_size, self.n_subquantizers, self.n_bits)

        # index all passages
        input_paths = glob.glob(args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        print(input_paths)
        if os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(input_paths)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            print("Data indexing completed.")


        # load passages
        self.passages = src.data.load_passages(args.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        self.args = args
        self.device = device

        with open(f"{data_path}/entity2id.json") as fin:
            self.entity2id = json.load(fin)

    def add_embeddings(self, embeddings, ids):
        end_idx = min(self.indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        self.index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def index_encoded_data(self, embedding_files):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)
            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > self.indexing_batch_size:
                allembeddings, allids = self.add_embeddings(allembeddings, allids)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(allembeddings, allids)


    def embed_queries(self, queries):
        self.model.eval()
        embeddings, batch_question = [], []
        with torch.no_grad():

            for k, q in enumerate(queries):
                if self.lowercase:
                    q = q.lower()
                if self.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == self.per_gpu_batch_size or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=self.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")
        return embeddings.numpy()

    def add_passages(self, data, top_passages_and_scores, allow_rept=True, n_doc=None):
        assert len(data) == len(top_passages_and_scores)
        for i, d in enumerate(data):
            results_and_scores = top_passages_and_scores[i]
            docs = [self.passage_id_map[doc_id] for doc_id in results_and_scores[0]]
            scores = [str(score) for score in results_and_scores[1]]
            ctxs_num = len(docs)
            exist_doc_title = set()
            d["ctxs"] = []
            for c in range(ctxs_num):
                if n_doc and len(d["ctxs"]) == n_doc:
                    break
                if docs[c]["title"] in exist_doc_title:
                    if not allow_rept:
                        continue
                exist_doc_title.add(docs[c]["title"])
                d["ctxs"].append({
                    "id": results_and_scores[0][c],
                    "title": docs[c]["title"],
                    "text": docs[c]["text"],
                    "score": scores[c],
                })

    def SearchDoc(self, data, n_doc=20):
        data = copy.deepcopy(data)
        questions_embedding = self.embed_queries([d["query"] for d in data])
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, 1000)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        self.add_passages(data, top_ids_and_scores, allow_rept=False, n_doc=n_doc)
        return data

    def SearchPsg(self, data, n_doc=20):
        data = copy.deepcopy(data)
        start_time_retrieval = time.time()
        for d in data:
            questions_embedding = self.embed_queries([d["query"]])
            print("SearchPsg, query:", d["query"])
            print("SearchPsg, title:", d["title"])
            top_ids_and_scores = self.index.search_knn(questions_embedding, n_doc, select_ids=[int(k)-1 for k in self.entity2id[d["title"]]])
            print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
            self.add_passages([d], top_ids_and_scores[:len(self.entity2id[d["title"]])])
        return data

