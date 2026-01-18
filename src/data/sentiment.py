import torch
from scipy.special import softmax


def get_finbert_probabilities(sentence, model, tokenizer, device):

    LABELS = model.config.id2label

    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = softmax(logits.numpy().squeeze())

    scores = {LABELS[i]: float(probabilities[i]) for i in range(len(LABELS))}
    
    final_probs = {
        'positive': scores.get('positive', 0.0),
        'negative': scores.get('negative', 0.0),
        'neutral': scores.get('neutral', 0.0)
    }

    return final_probs
