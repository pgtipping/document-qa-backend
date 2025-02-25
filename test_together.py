from app.services.llm import LLMService
import asyncio

async def test_together():
    service = LLMService()
    result = await service.test_provider('together')
    print(f'Together test result: {result}')

if __name__ == '__main__':
    asyncio.run(test_together()) 