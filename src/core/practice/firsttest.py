import aiohttp
import asyncio
from typing import Any
import time


"""
async def say_after(delay,data):
    await asyncio.sleep(delay)
    print(data)
    return f"{data}+{delay}"
async def main():
    print(f"start time:{time.strftime('%X')}")
    ret=await asyncio.gather(say_after(1,"heelo"),say_after(2,"woorld"))
    print(ret)
    print(f"end time:{time.strftime('%X')}")

asyncio.run(main())
"""
async def get_json(
    session:aiohttp.ClientSession,
    url:str,
    params:dict[str,str]|None=None,
    headers:dict[str,str]|None=None ,
)->dict[str,Any]:
    try:
        async with session.get(url,params=params,headers=headers)as resp:
            if resp.status<200 or resp.status>=300 :
                error_text=await resp.text() 
                print(error_text)
                return None
            data=await resp.text()
            return data
    except aiohttp.ClientError:
        print("client error")
        return None
    except asyncio.TimeoutError:
        print("timeout error")
        return None

async def main():
    timeout=aiohttp.ClientTimeout(total=5)
    headers={
        "User-Agent": "my-aiohttp-client/0.1",
        "X-Request-From": "python-learning",
    }
    async with aiohttp.ClientSession(timeout=timeout) as session:
        get_result=await get_json(
            session=session,
            url="https://baidu.com/",
            params={
                "a":"b",
                "c":"d",
            },
            headers=headers,
        )
    print("get result:\n")
    print(get_result)
        
asyncio.run(main())

"""
asyncio.run(main())
    ↓
创建事件循环，开始执行 main 协程
    ↓
创建 timeout 配置
    ↓
创建 headers 字典
    ↓
async with aiohttp.ClientSession(timeout=timeout) as session
    ↓
创建 aiohttp 客户端会话，内部管理连接资源
    ↓
await get_text(...)
    ↓
进入 get_text 协程
    ↓
session.get(url, params=params, headers=headers)
    ↓
aiohttp 发送 GET 请求
    ↓
params 被拼接到 URL 查询参数里
    ↓
headers 被放入 HTTP 请求头里
    ↓
如果 URL 错误、连接失败、DNS 失败等，进入 ClientError
    ↓
如果请求超过 5 秒，进入 TimeoutError
    ↓
如果服务器返回了响应，拿到 resp
    ↓
检查 resp.status
    ↓
如果不是 2xx,读取响应体并返回 None
    ↓
如果是 2xx,读取响应体文本
    ↓
return data
    ↓
data 赋值给 get_result
    ↓
关闭响应对象
    ↓
关闭 ClientSession
    ↓
打印 get_result
asyncio.run(main()) 启动事件循环执行 main();main() 中创建超时
配置、请求头和 ClientSession;然后 await get_text(...) 发起异步
GET 请求。params 会被拼接进 URL,headers 会被放进请求头。如果
连接层失败或超时，会进入对应异常处理；如果服务器返回响应，则通过
resp.status 判断 HTTP 状态码，失败时读取错误响应体并返回
None,成功时读取响应体文本并返回给 get_result,最后打印结果。
"""