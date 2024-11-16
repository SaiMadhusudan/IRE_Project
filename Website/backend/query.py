import anthropic
import base64


anthropic_client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)
def query_call(query_given , image_file , query_type):

    if query_type == "text":
        final_query = f"Query: {query_given}. Please provide: 1. Detailed Product Description: - Generate a comprehensive description of the product(s) mentioned in the query - Include key attributes like color, style, material, purpose, and other relevant characteristics - Focus on features that would help categorize the product accurately - Length: 100-150 words 2. Enhanced Search Query: - Reformulate the original query to be more precise and detailed - Include important product specifications and attributes - Make it suitable for semantic search - Format it as a clear, specific search statement. Ensure the description and query capture all relevant details while remaining focused on the core product attributes that would help with categorization. Also your output should be in the following format: Enhanced query: <enhanced query should be here> Product Description: <detailed product desciption should be here>"
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": final_query
                        }
                    ]
                }
            ]
        )
        content = message.content[0].text
        query, description = content.split("\n\n")
        return query.split(":")[1] , description.split(":")[1]
    else:              
        if image_file:
            image_file_path = f"/tmp/{image_file.filename}"
            image_file.save(image_file_path)
            with open(image_file_path, "rb") as image:
                image_read = image.read()
                # Encode the image in base64 and decode it to a string
                image_64_encode = base64.b64encode(image_read).decode('utf-8')

            final_query = f"Query: {query_given}. Please provide: 1. Detailed Product Description: - Generate a comprehensive description of the product(s) mentioned in the query - Include key attributes like color, style, material, purpose, and other relevant characteristics - Focus on features that would help categorize the product accurately - Length: 100-150 words 2. Enhanced Search Query: - Reformulate the original query to be more precise and detailed - Include important product specifications and attributes - Make it suitable for semantic search - Format it as a clear, specific search statement. Ensure the description and query capture all relevant details while remaining focused on the core product attributes that would help with categorization. Also your output should be in the following format: Enhanced query: <enhanced query should be here> Product Description: <detailed product desciption should be here>"
            message = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": final_query
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_64_encode
                                }
                            }
                        ]
                    }
                ]
            )
            content = message.content[0].text
            query, description = content.split("\n\n")
            return query.split(":")[1] , description.split(":")[1]
        
if __name__ == "__main__":
    query, description = query_call("give me a sweater", "nothing", "text_only")
    print(query)
    print("*" * 25)
    print(description)