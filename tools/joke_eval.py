import numpy as np
import spacy
from strands import tool

# CRITICAL: Use 'md' or 'lg' for actual word vectors. 
# If you are restricted to 'sm' in your environment, similarity will not work.
#try:
nlp = spacy.load("en_core_web_md") 
#except:
#    nlp = spacy.load("en_core_web_sm") # Fallback, but similarity will be poor

@tool
def evaluate_pun(context_words: list[str], m1_label: str, m2_label: str):
    """
    Evaluates if a sentence is a valid joke, standard language, or nonsense
    using Ambiguity (Entropy) and Distinctiveness (KL Divergence).
    """
    # Pre-process labels to avoid repeated nlp() calls
    target_m1 = nlp(m1_label)
    target_m2 = nlp(m2_label)
    
    support_m1 = []
    support_m2 = []
    
    for word in context_words:
        token = nlp(word)
        # Check if the token actually has vector data
        if not token.has_vector or token.is_stop or token.is_punct:
            continue
            
        # Cosine similarity as a proxy for 'Relatedness' R(w, m)
        sim_m1 = token.similarity(target_m1)
        sim_m2 = token.similarity(target_m2)
        
        # We use a small epsilon (1e-9) to avoid log(0) later
        support_m1.append(max(1e-9, sim_m1)) 
        support_m2.append(max(1e-9, sim_m2))

    # --- ERROR HANDLING: Check if we found any context words ---
    if not support_m1:
        return {"error": "No valid context words found for analysis."}

    # 3. CALCULATE AMBIGUITY (Entropy)
    sum_s1, sum_s2 = sum(support_m1), sum(support_m2)
    
    # Use a safe softmax or simple normalization
    # If the sums are very small, np.exp can overflow/underflow
    total_sum = sum_s1 + sum_s2
    probs = [sum_s1 / total_sum, sum_s2 / total_sum]
    
    entropy = -np.sum(probs * np.log2(probs + 1e-9))

    # 4. CALCULATE DISTINCTIVENESS (KL Divergence Proxy)
    f1 = np.array(support_m1) / (sum_s1 + 1e-9)
    f2 = np.array(support_m2) / (sum_s2 + 1e-9)
    
    # Symmetrized KL Divergence
    kl_dist = np.sum(f1 * np.log((f1 + 1e-9) / (f2 + 1e-9))) + \
              np.sum(f2 * np.log((f2 + 1e-9) / (f1 + 1e-9)))

    # 5. VERDICT LOGIC
    is_ambiguous = entropy > 0.7  
    is_distinct = kl_dist > 1.5   
    
    if is_ambiguous and is_distinct:
        verdict = "VALID JOKE (Incongruity Detected)"
    elif is_ambiguous and not is_distinct:
        verdict = "NONSENSE (Confusing/De-punned)"
    else:
        verdict = "STANDARD LANGUAGE (One story dominates)"

    return {
        "ambiguity_entropy": round(float(entropy), 3),
        "distinctiveness_kl": round(float(kl_dist), 3),
        "verdict": verdict,
        "probabilities": {"m1": round(probs[0], 2), "m2": round(probs[1], 2)}
    }