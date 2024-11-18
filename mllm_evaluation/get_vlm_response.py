import anthropic
import base64

from prompt import final_prompt

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)

def claude_response(image_path, prompt):
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode(image_data).decode("utf-8")

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    )

    return message.content

if __name__ == "__main__":
    image_path = "/Users/venkatakesavvenna/Sem7/IRE/Project/MY_COMMITS/project_part_2/mllm_evaluation/images.jpeg"
    prompt = final_prompt("Please provide a detailed description of the image.", "The image shows a man standing confidently with his hands on his hips. He has curly black hair and is smiling warmly. He is wearing a crisp white shirt with a pocket on the left side, and the background features a simple beige and brown wall with a striped design running horizontally. The setting appears casual and well-lit, giving off a cheerful and approachable vibe.")
    response = claude_response(image_path, prompt)
    print(response)