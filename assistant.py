from sklearn.metrics.pairwise import cosine_similarity
from modules.model import get_embedding

def predict_category(embedding_text, category_embeddings, tokenizer, model, device):
    emb = get_embedding(embedding_text, tokenizer, model, device)
    similarities = {
        cat: cosine_similarity([emb], [vec])[0][0]
        for cat, vec in category_embeddings.items()
    }
    return max(similarities, key=similarities.get)