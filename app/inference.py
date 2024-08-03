from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score

def match_speakers(test_embeddings, database_embeddings):
    predictions = []
    for test_file, test_emb in test_embeddings.items():
        similarities = {speaker_id: cosine_similarity(test_emb, db_emb) for speaker_id, db_emb in database_embeddings.items()}
        predicted_speaker = max(similarities, key=similarities.get)
        predictions.append(predicted_speaker)
    return predictions

def evaluate(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1
