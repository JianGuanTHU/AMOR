import requests
class text_toolkits():
    def test_contriever(self, data, n_doc, func):
        data = {
            "data": data,
            "n_doc": n_doc,
            "func": func,
        }

        resp = requests.post("http://0.0.0.0:8095/retriever_wikipedia", json=data)
        assert resp.status_code == 200, f"request failed: {resp.status_code}, {resp.text}"
        return resp.json()["response"]

    def SearchDoc(self, query):
        data = self.test_contriever(data=[{
                "query": query,
            }], n_doc=10, func="SearchDoc")
        return data[0]["ctxs"]


    def SearchPsg(self, query, title):
        try:
            data = self.test_contriever(data=[{
                    "query": query,
                    "title": title,
                }], n_doc=10, func="SearchPsg")
            return data[0]["ctxs"]
        except:
            return []