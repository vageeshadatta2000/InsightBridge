import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

def extract_entities(docs):
    entity_summary = defaultdict(list)
    for doc in docs:
        parsed = nlp(doc.page_content)
        for ent in parsed.ents:
            entity_summary[ent.label_].append(ent.text)
    # Deduplicate
    return {label: list(set(entities)) for label, entities in entity_summary.items()}
