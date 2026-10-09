import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

class ChatService:

    def __init__(self):
        api_key = os.environ["OPENROUTER_API_KEY"]

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    async def ask(self, question: str) -> str:
        response = await self.client.chat.completions.create(
            model="deepseek/deepseek-v4.1-flash",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        if not response.choices:
            raise ValueError(f"Empty choices in response. Full response: {response}")

        return response.choices[0].message.content


async def main():
    chat_service = ChatService()

    query = input("Query:")
    answer = await chat_service.ask(query)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())