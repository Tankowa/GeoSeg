from openai import OpenAI

client = OpenAI(
    api_key="sk-5b29732002aa63ae367d47cca8ff5a3137606e7accd8f1e1062dc9c2a63fb359",
    base_url="https://14o.kangaroom.top/v1"
)

response = client.responses.create(
    model="gpt-5.4",
    reasoning={"effort": "low"},
    input="Hello"
)

print(response.output_text)