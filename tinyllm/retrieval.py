from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class RetrievedPage:
    title: str
    extract: str
    url: str


class WikipediaRetriever:
    endpoint = "https://en.wikipedia.org/w/api.php"

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    def search(self, query: str, limit: int = 3) -> list[RetrievedPage]:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": limit,
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "format": "json",
            "formatversion": 2,
        }
        response = requests.get(
            self.endpoint,
            params=params,
            headers={"User-Agent": "tiny-convo-llm/0.1"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
        return [
            RetrievedPage(page["title"], page.get("extract", "")[:1_500], page["fullurl"])
            for page in pages
            if page.get("extract") and page.get("fullurl")
        ]


def format_retrieval_context(pages: list[RetrievedPage]) -> str:
    if not pages:
        return ""
    entries = [f"[{page.title}]\n{page.extract}\nSource: {page.url}" for page in pages]
    return "Use these live reference extracts when relevant. Cite their source URLs.\n\n" + "\n\n".join(entries)

