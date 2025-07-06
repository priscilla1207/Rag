import spacy

nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def format_entities_as_flashcards(entities):
    flashcards = []
    for entity, label in entities:
        question = f"What is the type of '{entity}'?"
        answer = f"'{entity}' is a {label}."
        flashcards.append((question, answer))
    return flashcards
