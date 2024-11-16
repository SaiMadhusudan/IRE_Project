import anthropic
import base64

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)


if __name__ == '__main__':

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
                        "text": "What kind of Book is this?"
                    }
                ]
            }
        ]
    )   
    print(message.content)
