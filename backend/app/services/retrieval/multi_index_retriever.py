from backend.app.services.retrieval.query_router import QueryRouter


class MultiIndexRetriever:
    def __init__(self, ipc=None, judgment=None, user=None):
        self.ipc = ipc
        self.judgment = judgment
        self.user = user
        self.router = QueryRouter()

    def retrieve(self, query, top_k=5, has_user_docs=False):
        routing = self.router.route(query, has_user_docs)

        results = []

        if routing["ipc"] and self.ipc:
            results.extend(self.ipc.retrieve(query, top_k))

        if routing["judgment"] and self.judgment:
            results.extend(self.judgment.retrieve(query, top_k))

        if routing["user"] and self.user:
            results.extend(self.user.retrieve(query, top_k))

        return results
