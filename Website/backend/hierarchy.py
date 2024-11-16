import numpy as np

# Function to compute cosine similarity
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    return dot_product / (magnitude1 * magnitude2)

# Step 1: Generate embedding for the description
def get_top_subcategories(description ,model , category_embeddings,subcategory_embeddings, top_n_categories=3, top_n_subcategories=5):
    # Generate embedding for the input description
    description_embedding = model.encode(description)

    # Step 2: Find the top N categories based on cosine similarity
    category_scores = {
        category: cosine_similarity(description_embedding, embedding)
        for category, embedding in category_embeddings.items()
    }
    top_categories = sorted(category_scores, key=category_scores.get, reverse=True)[:top_n_categories]

    # Step 3: Within each of the top categories, find the top subcategories
    top_subcategories = []
    for category in top_categories:
        subcategory_scores = {
            subcategory: cosine_similarity(description_embedding, sub_embedding)
            for subcategory, sub_embedding in subcategory_embeddings[category].items()
        }
        sorted_subcategories = sorted(subcategory_scores, key=subcategory_scores.get, reverse=True)[:top_n_subcategories]
        top_subcategories.extend([(subcategory, subcategory_scores[subcategory]) for subcategory in sorted_subcategories])

    # Sort the collected top subcategories by similarity score across all categories
    top_subcategories = sorted(top_subcategories, key=lambda x: x[1], reverse=True)[:top_n_subcategories]
    
    ans = []
    
    for x , y in top_subcategories:
        ans.append(x)
    return ans

if __name__ == "__main__":
    description = "Loose-fitting, casual dress with long sleeves and a relaxed, oversized silhouette. The dress should have a scoop neckline with raw-edge detailing, a slightly dropped waist, and a rounded hemline that falls above the knee. Made from a soft, lightweight fabric like cotton or jersey for a flowy, comfortable look. Looking specifically for a solid red color, preferably in shades like burgundy, crimson, or wine."
    top_subcategories = get_top_subcategories(description)
    print(top_subcategories)