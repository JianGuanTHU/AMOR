import requests

def test_contriever():
    data = {
        "data": [{
            "query": "What was Ed Wood's nationality?",
            "title": 'Aaron',
        }],
        "n_doc": 50,
        # "func": "SearchDoc",
        "func": "SearchPsg",
    }

    resp = requests.post("http://0.0.0.0:8095/retriever_wikipedia", json=data)
    assert resp.status_code == 200, f"request failed: {resp.status_code}, {resp.text}"
    result = resp.json()["response"]
    print(result)
    for k, r in enumerate(result[0]["ctxs"]):
        print(k, r["title"])

test_contriever()