from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
import os
import torch

# Load DataFrame and other necessary modules
df = pd.read_csv('./dataset/products.csv')

from query import query_call
from hierarchy import get_top_subcategories
from tfidf import retrieve_top_documents_tfidf
from sentencebert import retrieve_top_documents_setencebert

app = Flask(__name__)
CORS(app)

from sentence_transformers import SentenceTransformer

# Load the SentenceBERT model and move it to CUDA:0
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to('cuda:0')

category_descriptions = {
    'books_movies_and_music': "Items related to books, movies, music, and other media.",
    'clothing_shoes_and_accessories': "Men's and women's clothing, shoes, and accessories.",
    'ebay_motors': "Vehicles, vehicle parts, and accessories.",
    'electronics': "Devices and gadgets like cameras, headphones, laptops, and video games.",
    'home_and_garden': "Furniture, decor, and other home and garden items.",
    'jewelry_and_watches': "Jewelry and watches for men and women.",
    'pet_supplies': "Supplies for pets, including dogs, cats, and other animals.",
    'sporting_goods': "Sports equipment and outdoor gear."
}

subcategory_descriptions = {
    'books_movies_and_music': {
        'Books': "Books across various genres and formats.",
        'DVDs': "DVDs of movies, shows, and educational content.",
        'Guitars-Basses': "Guitars, basses, and related musical instruments.",
        'Pianos-Keyboards-Organs': "Pianos, keyboards, and organs for music enthusiasts."
    },
    'clothing_shoes_and_accessories': {
        'Mens-Clothing': "Clothing for men, including shirts, pants, jackets, and more.",
        'Mens-Shoes': "Shoes for men, including casual, formal, and sports shoes.",
        'Travel-Luggage': "Luggage and bags for travel.",
        'Womens-Clothing': "Clothing for women, including dresses, tops, and pants.",
        'Womens-Shoes': "Shoes for women, including heels, flats, and sports shoes."
    },
    'ebay_motors': {
        'ATVs': "All-terrain vehicles for off-road adventures.",
        'Boats': "Boats for leisure, fishing, and water sports.",
        'Cadillac': "Cadillac vehicles and accessories.",
        'Ford': "Ford vehicles and related parts.",
        'Jeep': "Jeep vehicles and accessories.",
        'Mercedes-Benz': "Mercedes-Benz cars and parts.",
        'Scooters-Mopeds': "Scooters and mopeds for urban transportation.",
        'Toyota': "Toyota cars and accessories.",
        'Toyota-Supra-Cars': "Toyota Supra cars and related accessories.",
        'Yamaha': "Yamaha vehicles and musical instruments."
    },
    'electronics': {
        'Digital-Cameras': "Digital cameras for photography.",
        'Headphones': "Headphones for personal audio experiences.",
        'Laptops-Netbooks': "Laptops and netbooks for personal and professional use.",
        'Video-Games': "Video games and gaming consoles."
    },
    'home_and_garden': {
        'Beds-Headboards': "Beds and headboards for comfortable sleeping.",
        'Chairs': "Chairs for seating in various settings.",
        'Chandeliers-Ceiling-Fixtures': "Lighting fixtures including chandeliers.",
        'Tables': "Tables for dining, working, and other uses."
    },
    'jewelry_and_watches': {
        'Engagement-Rings': "Engagement rings in various designs.",
        'Fine-Earrings': "Fine earrings for various occasions.",
        'Watches': "Watches for men and women."
    },
    'pet_supplies': {
        'Dog-Supplies': "Supplies for dogs including food, toys, and accessories.",
        'Fish-Aquariums': "Fish and aquarium supplies."
    },
    'sporting_goods': {
        'Archery-Equipment': "Equipment for archery enthusiasts.",
        'Basketball-Equipment': "Basketball gear and equipment.",
        'Boxing-MMA-Equipment': "Boxing and MMA equipment for training.",
        'Golf-Accessories': "Accessories for golf players."
    }
}

# Generate embeddings for each category and subcategory
category_embeddings = {category: model.encode(description, device='cuda:0') for category, description in category_descriptions.items()}
subcategory_embeddings = {
    category: {subcategory: model.encode(description, device='cuda:0') for subcategory, description in subcategories.items()}
    for category, subcategories in subcategory_descriptions.items()
}
def filter_l2_category(df, l2_categories):
    return [str(doc) + '.txt' for doc in df[df['L2 Category'].isin(l2_categories)]['Product ID'].values.tolist()]

def meta_data_images(df, final_results):
    final_results = [result[:-4] for result in final_results]
    print(final_results)
    images = []
    meta_data = []

    # Set 'Product ID' as the index for faster lookups
    df['Product ID'] = df['Product ID'].astype(str).str.strip()  # Ensure Product ID is string and stripped
    df = df.set_index('Product ID')

    for product_id in final_results:
        product_id = str(product_id).strip()  # Ensure product_id is string and stripped
        # Check if the product ID exists in the DataFrame
        if product_id in df.index:
            # Extract required fields
            l1_category = df.at[product_id, 'L1 Category']
            l2_category = df.at[product_id, 'L2 Category']
            l3_category = df.at[product_id, 'L3 Category']
            image_path_rel = df.at[product_id, 'Image Path']

            # Construct paths using os.path.join for better OS compatibility
            meta_data_path = os.path.join(
                './dataset', l1_category, l2_category, l3_category, product_id, f"{product_id}.json"
            )
            image_path = os.path.join(
                './dataset', l1_category, l2_category, l3_category, product_id, image_path_rel
            )

            # Read and append specific metadata parts with handling for missing cases
            with open(meta_data_path, 'r', encoding='utf-8') as file:
                meta_data_info = json.load(file)
                title = meta_data_info.get('title', 'Unknown Title')
                item_specifics = meta_data_info.get('item_specifics', 'Unknown Item Specifics')
                meta_data.append({'title': title, 'item_specifics': item_specifics})

            # Append image path
            images.append(image_path)
        else:
            print(f"Product ID '{product_id}' not found in DataFrame.")

    return images, meta_data
@app.route('/query', methods=['POST'])
def query():
    query_given = request.form.get('query_given', '').strip()
    image_file = request.files.get('image_file' ,"None")
    query_type = request.form.get('query_type', 'text')
    retrieval_model = request.form.get('model', "sentencebert")

    # Process the query and get description
    query, description = query_call(query_given, image_file, query_type)

    # Get top subcategories based on the description
    top_subcategories = get_top_subcategories(description, model, category_embeddings, subcategory_embeddings)

    # Filter txt files based on the top subcategories
    txt_files = filter_l2_category(df, top_subcategories)

    # Retrieve top documents using the selected retrieval model
    if retrieval_model == "bm25":
        pass
    elif retrieval_model == "tfidf":
        final_results = retrieve_top_documents_tfidf(txt_files, description)
    elif retrieval_model == "sentencebert":
        final_results = retrieve_top_documents_setencebert(txt_files, description,model)
    else:
        return jsonify({'error': 'Invalid retrieval model selected.'}), 400

    # Get images and metadata for the top 3 results
    images, meta_data = meta_data_images(df, final_results[:3])  # Get top 3 results

    # Prepare the response data
    results = []
    for img_path, meta in zip(images, meta_data):
        # Get the relative path of the image with respect to the 'dataset' directory
        img_relative_path = os.path.relpath(img_path, './dataset')

        # Create the URL for the image
        img_url = request.host_url + 'image/' + img_relative_path.replace('\\', '/')

        results.append({
            'image_url': img_url,
            'title': meta.get('title', 'No Title'),
            'item_specifics': meta.get('item_specifics', {})
        })

    # Return the results as JSON
    return jsonify({'results': results})

# Route to serve images
@app.route('/image/<path:filename>')
def serve_image(filename):
    # Send file from the 'dataset' directory
    return send_from_directory('dataset', filename)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)