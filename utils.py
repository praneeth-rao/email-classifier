import re
import spacy

nlp = spacy.load("en_core_web_sm")

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone_number": r"(\+91[\-\s]?)?[0]?(91)?[789]\d{9}",
    "aadhar_num": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "credit_debit_no": r"\b(?:\d[ -]*?){13,16}\b",
    "cvv_no": r"\b[0-9]{3}\b",
    "expiry_no": r"\b(0[1-9]|1[0-2])\/?([0-9]{2})\b",
    "dob": r"\b(0?[1-9]|[12][0-9]|3[01])[\/\-\s](0?[1-9]|1[0-2])[\/\-\s](\d{4})\b"
}

def mask_pii(text):
    masked_text = text
    entities = []

    for label, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, masked_text):
            start, end = match.span()
            entity_val = match.group()
            placeholder = f"[{label}]"
            masked_text = masked_text[:start] + placeholder + masked_text[end:]
            entities.append({
                "position": [start, start + len(placeholder)],
                "classification": label,
                "entity": entity_val
            })

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start = masked_text.find(ent.text)
            if start != -1:
                end = start + len(ent.text)
                placeholder = "[full_name]"
                masked_text = masked_text[:start] + placeholder + masked_text[end:]
                entities.append({
                    "position": [start, start + len(placeholder)],
                    "classification": "full_name",
                    "entity": ent.text
                })

    return masked_text, entities