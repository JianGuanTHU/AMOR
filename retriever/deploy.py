import json
import traceback
from flask import Flask, request
from run_contriever import Retriever
app = Flask(__name__)

appname = "/retriever_wikipedia"
port = 8095
device="cuda:0"

retriever = Retriever(
    data_path = "../data/retriever",   # data path, including 'wikipedia_embeddings', 'psgs_w100_tmp.tsv', 'entity2id.json' 
    model_name_or_path = "./model", # contriever-msmarco model from https://huggingface.co/facebook/contriever-msmarco
    device=device)

def retrieve_helper(data, retriever):
    n_doc = data['n_doc']
    func = data['func']
    data = data['data']
    if func == "SearchDoc":
        data = retriever.SearchDoc(data, n_doc=n_doc)
    elif func == "SearchPsg":
        data = retriever.SearchPsg(data, n_doc=n_doc)
    else:
        raise Exception(f"Function Name {func} Not Found.")
    return data

@app.route(appname, methods=['POST'])
def func():
    try:
        print("----------- in hello func ----------")
        data = json.loads(request.get_data(as_text=True))
        data = retrieve_helper(data, retriever)
        return json.dumps({"response": data}, ensure_ascii=False, indent=4)
    except:
        print(traceback.format_exc().strip())
        return json.dumps({"response": traceback.format_exc().strip()}, ensure_ascii=False,indent=4)

if __name__ == '__main__':
    # test(retriever)
    app.run(host="0.0.0.0", port=port)
