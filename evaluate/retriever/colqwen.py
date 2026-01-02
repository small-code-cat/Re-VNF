import requests

class ColQwenRetriever:
    def __init__(self, search_url, top_k):
        self.search_url = search_url
        self.top_k = top_k

    def search(self, query):
        if isinstance(query,str):
            query = [query]
        search_response = requests.get(self.search_url, params={"queries": query})
        search_results = search_response.json()
        image_path_list = [result['image_file'] for result in search_results[0]]
        return image_path_list[:self.top_k]