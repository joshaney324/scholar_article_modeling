import requests
import json

def get_citations(arxiv_id):
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {"fields": "citationCount,references,citations,embedding"}
    response = requests.get(url, params=params)
    return response.json()["citations"]

class Paper:
    def __init__(self, archive_id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license, abstract):
        self.archive_id = archive_id
        self.submitter = submitter
        self.authors = authors
        self.title = title
        self.comments = comments
        self.journal_ref = journal_ref
        self.doi = doi
        self.report_no = report_no
        self.categories = categories
        self.license = license
        self.abstract = abstract
        self.citations = get_citations(archive_id)


def main():

    papers = []
    with open('../data/raw/arxiv-metadata-oai-snapshot.json', 'r') as f:
        for line in f:

            aPaper = json.loads(line)
            archive_id = aPaper.get('id')
            submitter = aPaper.get('submitter')
            authors = aPaper.get('authors')
            title = aPaper.get('title')
            comments = aPaper.get('comments')
            journal_ref = aPaper.get('journalRef')
            doi = aPaper.get('doi')
            report_no = aPaper.get('reportNo')
            categories = aPaper.get('categories')
            license = aPaper.get('license')
            abstract = aPaper.get('abstract')
            papers.append(Paper(archive_id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license, abstract))

    print(f"Loaded {len(papers)} papers")
    print(papers[0])

if __name__ == "__main__":
    main()