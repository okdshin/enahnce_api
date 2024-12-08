import argparse
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import json
from typing import Dict, Any, List, AsyncGenerator


parser = argparse.ArgumentParser()
parser.add_argument("--target-api-endpoint-url", type=str)
args = parser.parse_args()

app = FastAPI()

TARGET_URL = args.target_api_endpoint_url
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
TIMEOUT = 600


async def execute_tool_selection(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    headers: Dict[str, str],
) -> List[str]:
    """ツール選択のfunction callingを実行"""
    tool_selector = {
        "type": "function",
        "function": {
            "name": "select_tools",
            "description": "Select appropriate tools based on the user's request",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of the tools to use",
                    }
                },
                "required": ["selected_tools"],
            },
        },
    }

    selector_request = {
        "messages": messages,
        "tools": [tool_selector],
        "tool_choice": "select_tools",
        "stream": False,  # tool選択は常に非ストリーミング
    }

    response = await client.post(
        url=f"{TARGET_URL}{CHAT_COMPLETIONS_PATH}",
        json=selector_request,
        headers=headers,
    )

    response_data = response.json()
    tool_call = (
        response_data.get("choices", [{}])[0]
        .get("message", {})
        .get("tool_calls", [{}])[0]
    )

    if tool_call and tool_call.get("function", {}).get("name") == "select_tools":
        try:
            args = json.loads(tool_call["function"]["arguments"])
            return args.get("selected_tools", [])
        except Exception:
            return []

    return []


async def stream_sequential_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    selected_tools: List[str],
    client: httpx.AsyncClient,
    headers: Dict[str, str],
) -> AsyncGenerator[bytes, None]:
    """複数ツールを順番にストリーミング実行"""
    for i, tool_name in enumerate(selected_tools):
        tool_request = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_name,
            "stream": True,
        }

        async with client.stream(
            "POST",
            f"{TARGET_URL}{CHAT_COMPLETIONS_PATH}",
            json=tool_request,
            headers=headers,
            timeout=TIMEOUT,
        ) as response:
            is_last_tool = i == len(selected_tools) - 1

            async for chunk in response.aiter_bytes():
                if not chunk.strip():
                    continue

                try:
                    chunk_str = chunk.decode("utf-8")
                    if chunk_str.startswith("data: "):
                        chunk_str = chunk_str[6:]  # Remove 'data: ' prefix

                    # [DONE]の場合、最後のツール以外はスキップ
                    if chunk_str.strip() == "[DONE]":
                        if is_last_tool:
                            yield b"data: [DONE]\n\n"
                        continue

                    chunk_data = json.loads(chunk_str)

                    # 最初以外のツールの場合、IDを変更
                    if i > 0:
                        if "id" in chunk_data:
                            chunk_data["id"] = f"{chunk_data['id']}-{i}"

                    # チャンクを適切な形式で出力
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode("utf-8")

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue


async def forward_chat_request(request: Request, client: httpx.AsyncClient) -> Response:
    body = await request.body()
    body_json = json.loads(body)

    headers = {
        "Content-Type": "application/json",
        "Authorization": request.headers.get("Authorization", ""),
    }

    is_streaming = body_json.get("stream", False)
    is_auto_tool = body_json.get("tool_choice") == "auto"

    # tool_choice=auto以外は直接転送
    if not is_auto_tool:
        response = await client.post(
            url=f"{TARGET_URL}{CHAT_COMPLETIONS_PATH}",
            content=body,
            headers=headers,
            timeout=TIMEOUT,
        )

        if is_streaming:
            return StreamingResponse(
                response.aiter_bytes(), media_type="text/event-stream"
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    # ツール選択の実行（非ストリーミング）
    selected_tools = await execute_tool_selection(
        body_json.get("messages", []), body_json.get("tools", []), client, headers
    )

    if not selected_tools:
        # ツールが選択されなかった場合は通常のレスポンス
        body_json.pop("tool_choice", None)
        response = await client.post(
            url=f"{TARGET_URL}{CHAT_COMPLETIONS_PATH}",
            json=body_json,
            headers=headers,
            timeout=TIMEOUT,
        )
        if is_streaming:
            return StreamingResponse(
                response.aiter_bytes(), media_type="text/event-stream"
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    # 選択されたツールを実行
    if is_streaming:
        return StreamingResponse(
            stream_sequential_tools(
                body_json.get("messages", []),
                body_json.get("tools", []),
                selected_tools,
                client,
                headers
            ),
            media_type="text/event-stream"
        )

    # 非ストリーミングの場合は全ツールを実行し結果をマージ
    results = []
    for tool_name in selected_tools:
        tool_request = {
            "messages": body_json.get("messages", []),
            "tools": body_json.get("tools", []),
            "tool_choice": tool_name,
            "stream": False,
        }
        response = await client.post(
            url=f"{TARGET_URL}{CHAT_COMPLETIONS_PATH}",
            json=tool_request,
            headers=headers,
        )
        results.append(response.json())

    # 結果のマージ
    if not results:
        return Response(content=json.dumps({}), media_type="application/json")

    merged = results[0].copy()
    all_tool_calls = []
    for result in results:
        tool_calls = (
            result.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        )
        all_tool_calls.extend(tool_calls)

    if all_tool_calls:
        merged["choices"][0]["message"]["tool_calls"] = all_tool_calls

    return Response(content=json.dumps(merged), media_type="application/json")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    async with httpx.AsyncClient() as client:
        return await forward_chat_request(request, client)


def serve():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
