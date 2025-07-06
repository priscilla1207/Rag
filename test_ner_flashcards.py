# test_ner_flashcards.py
from ner_utils import extract_named_entities, format_entities_as_flashcards

sample_text = """
Isaac Newton was born in 1643 in Woolsthorpe, England. He studied at the University of Cambridge.
He is known for formulating the laws of motion and universal gravitation.
"""

# Step 1: Extract entities
entities = extract_named_entities(sample_text)

# Step 2: Format flashcards
flashcards = format_entities_as_flashcards(entities)

# Print results
print("Entities Found:")
for e in entities:
    print(e)

print("\nGenerated Flashcards:")
for i, (q, a) in enumerate(flashcards, 1):
    print(f"{i}. Q: {q}\n   A: {a}")
