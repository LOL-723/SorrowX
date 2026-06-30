from fastapi import FastAPI
import asyncio
import time
import sys

"""
app=FastAPI(root_path="/api/v1")

data=[
    {1:"ok"},
    {1:"not ok"},
]
@app.get("/readtest")
async def ReadAllTest():
    return {"test":data}
@app.get("/gettest/{status}")
async def GetOneTest(status:str): # type: ignore
    for test in data:
        if test.get(1)==status:
            return {"data":test}
    raise HTTPException(status_code=404)
"""
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

timeout = float(os.getenv("LLM_TIMEOUT", "60"))
model = os.getenv("LLM_MODEL")
if not model:
    raise RuntimeError("LLM_MODEL is not set in .env")

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    timeout=timeout,
)

async def getresponse(delay:int,data:str):
    await asyncio.sleep(delay)
    response =await client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":data}],
    )
    print(response.choices[0].message.content)
    print("//////////////////")

async def main():
    print(f"first time:{time.strftime('%X')}")
    await asyncio.gather(getresponse(1,"回复你好世界"),getresponse(2,"reply hello world"))
    print(f"end time:{time.strftime('%X')}")

asyncio.run(main())
