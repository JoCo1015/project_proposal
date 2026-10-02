import spacy
from spacy.tokens import DocBin
import re

# Sample annotated hardware text training examples
TRAIN_DATA = [
    ("Selling laptop with Intel Core i7-12700K 16GB RAM RTX 3060 512GB SSD", 
     {"entities": [(18, 36, "CPU"), (37, 45, "RAM"), (46, 54, "GPU"), (55, 64, "STORAGE")]}),
    ("Used gaming PC AMD Ryzen 5 5600X 32GB RAM RTX 4070 1TB SSD", 
     {"entities": [(15, 31, "CPU"), (32, 40, "RAM"), (41, 49, "GPU"), (50, 57, "STORAGE")]}),
    ("Budget desktop Intel i3-10100 8GB DDR4 GTX 1650 256GB SSD", 
     {"entities": [(15, 28, "CPU"), (29, 37, "RAM"), (38, 46, "GPU"), (47, 56, "STORAGE")]}),
    ("Apple MacBook Air M2 8GB 512GB SSD", 
     {"entities": [(18, 20, "CPU"), (21, 24, "RAM"), (25, 34, "STORAGE")]}),
    ("Workstation Intel Xeon E5-2680 64GB RAM Quadro P4000 2TB SSD", 
     {"entities": [(12, 29, "CPU"), (30, 38, "RAM"), (39, 51, "GPU"), (52, 59, "STORAGE")]}),
]

def create_spacy_dataset(train_data, output_path="train.spacy"):
    """Convert annotated tuple data into spaCy DocBin binary file format for training."""
    nlp = spacy.blank("en")
    doc_bin = DocBin()

    for text, annot in train_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                print(f"Skipping entity [{text[start:end]}] due to token boundary misalignment.")
            else:
                ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)

    doc_bin.to_disk(output_path)
    print(f"Successfully generated spaCy training dataset: {output_path} ({len(train_data)} samples)")

if __name__ == "__main__":
    create_spacy_dataset(TRAIN_DATA)
