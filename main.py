import json
import logging
import asyncio
from flask import Flask, request, Response, stream_with_context
import httpx
import time
from dotenv import load_dotenv
import os
import ast
import uuid

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置 httpx 的日志级别
logging.getLogger("httpx").setLevel(logging.DEBUG)

# 加载环境变量
load_dotenv()

# 从环境变量获取API基础URL
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "http://2ya41yc84533.vicp.fun:35850/v1")

# 增加超时时间
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

class DifyModelManager:
    def __init__(self):
        self.api_keys = []
        self.name_to_api_key = {}  # 应用名称到API Key的映射
        self.api_key_to_name = {}  # API Key到应用名称的映射
        self.app_types = {}  # 新增：存储应用类型
        self.load_api_keys()

    def load_api_keys(self):
        """从环境变量加载API Keys和类型"""
        api_keys_str = os.getenv('DIFY_API_KEYS', '')
        if not api_keys_str:
            logger.warning("No API keys found in environment variables")
            return
        
        logger.info("Loading API keys from environment...")
        
        # 清空现有数据
        self.api_keys = []
        self.name_to_api_key = {}
        self.api_key_to_name = {}
        self.app_types = {}
        
        # 创建一个 httpx 客户端
        with httpx.Client(
            transport=httpx.HTTPTransport(
                proxy="http://127.0.0.1:10809"
            )
        ) as client:
            # 解析配置
            for item in api_keys_str.split(','):
                try:
                    api_key, app_type = item.strip().split(':')
                    logger.debug(f"Processing API key: {api_key[:8]}... (type: {app_type})")
                    
                    # 获取应用名称
                    try:
                        response = client.get(
                            f"{DIFY_API_BASE}/info",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json"
                            },
                            params={"user": "default_user"},
                            timeout=10.0
                        )
                        
                        if response.status_code == 200:
                            app_info = response.json()
                            name = app_info.get("name", f"App-{api_key[:8]}")
                            
                            self.api_keys.append(api_key)
                            self.name_to_api_key[name] = api_key
                            self.api_key_to_name[api_key] = name
                            self.app_types[name] = app_type
                            
                            logger.info(f"Loaded {app_type} app '{name}' with API key: {api_key[:8]}...")
                        else:
                            logger.error(f"Failed to get app info for API key {api_key[:8]}: {response.status_code}")
                            
                    except Exception as e:
                        logger.error(f"Error fetching app info for API key {api_key[:8]}: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"Error processing API key entry: {item}: {str(e)}")
                
        model_count = len(self.name_to_api_key)
        if model_count > 0:
            logger.info(f"Successfully loaded {model_count} models: {', '.join(self.name_to_api_key.keys())}")
        else:
            logger.warning("No valid models were loaded")

    def fetch_app_name(self, api_key):
        """只获取应用名称"""
        try:
            response = httpx.get(
                f"{DIFY_API_BASE}/info",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                params={"user": "default_user"}
            )
            
            if response.status_code == 200:
                app_info = response.json()
                return {
                    "name": app_info.get("name", f"App-{api_key[:8]}")
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching app name: {str(e)}")
            return None

    def get_api_key(self, model_name):
        """根据模型名称获取API Key"""
        return self.name_to_api_key.get(model_name)

    def get_available_models(self):
        """获取可用模型列表"""
        return [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dify"
            }
            for name in self.name_to_api_key.keys()
        ]

# 创建模型管理器实例
model_manager = DifyModelManager()

app = Flask(__name__)

def get_api_key(model_name):
    """根据模型名称获取对应的API密钥"""
    api_key = model_manager.get_api_key(model_name)
    if not api_key:
        logger.warning(f"No API key found for model: {model_name}")
    return api_key

def transform_openai_to_dify(openai_request, endpoint):
    """将OpenAI格式的请求转换为Dify格式"""
    
    if endpoint == "/chat/completions":
        messages = openai_request.get("messages", [])
        stream = openai_request.get("stream", False)
        
        dify_request = {
            "inputs": {},
            "query": messages[-1]["content"] if messages else "",
            "response_mode": "streaming" if stream else "blocking",
            "conversation_id": openai_request.get("conversation_id", None),
            "user": openai_request.get("user", "default_user")
        }

        # 添加历史消息
        if len(messages) > 1:
            history = []
            for msg in messages[:-1]:  # 除了最后一条消息
                history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            dify_request["conversation_history"] = history

        return dify_request
    
    return None

def transform_dify_to_openai(dify_response, model="claude-3-5-sonnet-v2", stream=False):
    """将Dify格式的响应转换为OpenAI格式"""
    
    if not stream:
        return {
            "id": dify_response.get("message_id", ""),
            "object": "chat.completion",
            "created": dify_response.get("created", int(time.time())),
            "model": model,  # 使用实际使用的模型
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": dify_response.get("answer", "")
                },
                "finish_reason": "stop"
            }]
        }
    else:
        # 流式响应的转换在stream_response函数中处理
        return dify_response

def create_openai_stream_response(content, message_id, model="claude-3-5-sonnet-v2"):
    """创建OpenAI格式的流式响应"""
    return {
        "id": message_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": content
            },
            "finish_reason": None
        }]
    }

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        openai_request = request.get_json()
        logger.info(f"Received request: {json.dumps(openai_request, ensure_ascii=False)}")
        
        model = openai_request.get("model")
        logger.info(f"Using model: {model}")
        
        # 检查模型类型
        model_type = model_manager.app_types.get(model)
        if not model_type:
            error_msg = f"Model {model} not found. Available models: {', '.join(model_manager.name_to_api_key.keys())}"
            logger.error(error_msg)
            return {
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error"
                }
            }, 404
            
        logger.info(f"Model type: {model_type}")
        
        # 根据模型类型选择处理方式
        if model_type == "workflow":
            logger.info("Handling as workflow request")
            return handle_workflow_request(openai_request)
        else:
            logger.info("Handling as chat request")
            return handle_chat_request(openai_request)
            
    except Exception as e:
        logger.exception("Request error occurred")
        return {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }, 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    logger.info("Listing available models")
    try:
        models = []
        for name, api_key in model_manager.name_to_api_key.items():
            model_type = model_manager.app_types.get(name, "unknown")
            models.append({
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dify",
                "type": model_type  # 添加类型信息
            })
            logger.info(f"Found model: {name} (type: {model_type})")
            
        return {
            "object": "list",
            "data": models
        }
    except Exception as e:
        logger.exception("Error listing models")
        return {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }, 500

# 新增 Workflow 相关路由
@app.route('/v1/workflow/completions', methods=['POST'])
def workflow_completions():
    """处理 workflow 完成请求"""
    try:
        openai_request = request.get_json()
        logger.info(f"Received workflow request: {json.dumps(openai_request, ensure_ascii=False)}")
        
        model = openai_request.get("model", "default_workflow")
        api_key = get_api_key(model)
        
        if not api_key:
            error_msg = f"Workflow {model} not found. Available workflows: {', '.join(model_manager.name_to_api_key.keys())}"
            logger.error(error_msg)
            return {
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                    "code": "workflow_not_found"
                }
            }, 404

        workflow_request = transform_openai_to_workflow(openai_request)
        logger.info(f"Transformed workflow request: {json.dumps(workflow_request, ensure_ascii=False)}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        stream = openai_request.get("stream", False)
        workflow_endpoint = f"{DIFY_API_BASE}/workflows/run"

        if stream:
            return stream_workflow_response(workflow_endpoint, workflow_request, headers, model)
        else:
            return sync_workflow_response(workflow_endpoint, workflow_request, headers, model)
            
    except Exception as e:
        logger.exception("Workflow error occurred")
        return {
            "error": {
                "message": str(e),
                "type": "workflow_error",
            }
        }, 500

def transform_openai_to_workflow(openai_request):
    """将 OpenAI 格式转换为 Workflow 格式"""
    messages = openai_request.get("messages", [])
    stream = openai_request.get("stream", False)
    
    # 从消息中提取输入
    last_message = messages[-1]["content"] if messages else ""
    
    workflow_request = {
        "inputs": {
            "query": last_message  # 使用 query 而不是 text
        },
        "response_mode": "streaming" if stream else "blocking",
        "user": openai_request.get("user", "default_user"),
        "conversation_id": openai_request.get("conversation_id")
    }
    
    return workflow_request

def sync_workflow_response(endpoint, request_data, headers, model):
    """处理同步 workflow 响应"""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(
                endpoint,
                json=request_data,
                headers=headers
            )
            
            if response.status_code == 400:
                error_msg = response.json().get("message", "Bad Request")
                logger.error(f"Bad request: {error_msg}")
                return {
                    "error": {
                        "message": error_msg,
                        "type": "invalid_request_error",
                    }
                }, 400
                
            if response.status_code != 200:
                error_msg = f"Workflow API error: {response.text}"
                logger.error(f"Request failed: {error_msg}")
                return {
                    "error": {
                        "message": error_msg,
                        "type": "workflow_error",
                        "code": response.status_code
                    }
                }, response.status_code

            workflow_response = response.json()
            logger.info(f"Received workflow response: {json.dumps(workflow_response, ensure_ascii=False)}")
            return transform_workflow_to_openai(workflow_response, model)
        
    except httpx.RequestError as e:
        error_msg = f"Failed to connect to workflow API: {str(e)}"
        logger.error(error_msg)
        return {
            "error": {
                "message": error_msg,
                "type": "workflow_error",
                "code": "connection_error"
            }
        }, 503

def transform_workflow_to_openai(workflow_response, model="default_workflow"):
    """将 Workflow 响应转换为 OpenAI 格式"""
    try:
        if isinstance(workflow_response, str):
            # 如果是字符串，尝试解析JSON
            workflow_response = json.loads(workflow_response)
            
        # 工作流响应可能有不同的格式
        content = None
        if "answer" in workflow_response:
            content = workflow_response["answer"]
        elif "data" in workflow_response and "outputs" in workflow_response["data"]:
            content = workflow_response["data"]["outputs"]
        elif "outputs" in workflow_response:
            content = workflow_response["outputs"]
        else:
            content = str(workflow_response)  # 如果找不到预期的字段，转换整个响应
            
        return {
            "id": workflow_response.get("workflow_run_id", str(time.time())),
            "object": "workflow.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                },
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        logger.error(f"Error transforming workflow response: {str(e)}")
        return {
            "error": {
                "message": "Failed to transform workflow response",
                "type": "workflow_error"
            }
        }

def stream_workflow_response(endpoint, request_data, headers, model):
    """处理 Workflow 的流式响应"""
    def generate():
        client = httpx.Client(
            timeout=None,
            transport=httpx.HTTPTransport(
                proxy="http://127.0.0.1:10809"
            )
        )
        
        try:
            with client.stream('POST', endpoint, json=request_data, headers=headers) as response:
                if response.status_code != 200:
                    error_msg = response.json().get("message", "Unknown error")
                    logger.error(f"Workflow API error: {error_msg}")
                    error_chunk = {
                        "error": {
                            "message": error_msg,
                            "type": "workflow_error",
                            "code": response.status_code
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    return

                # 读取响应内容
                for chunk in response.iter_text():
                    if not chunk:
                        continue
                        
                    # 处理每一行
                    for line in chunk.splitlines():
                        if not line or not line.startswith('data: '):
                            continue
                            
                        try:
                            event_data = json.loads(line[6:])  # 去掉 "data: " 前缀
                            logger.debug(f"Received event: {event_data}")
                            
                            # 只处理 text_chunk 事件
                            if event_data.get('event') == 'text_chunk':
                                text = event_data.get('data', {}).get('text', '')
                                if text:
                                    # 构造 OpenAI 格式的响应
                                    chunk_data = {
                                        "id": "chatcmpl-" + str(uuid.uuid4()),
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [
                                            {
                                                "delta": {"content": text},
                                                "index": 0,
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                            
                            # 处理结束事件
                            elif event_data.get('event') == 'workflow_finished':
                                # 发送结束标记
                                end_chunk = {
                                    "id": "chatcmpl-" + str(uuid.uuid4()),
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "delta": {},
                                            "index": 0,
                                            "finish_reason": "stop"
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(end_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse event data: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error in workflow response streaming: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            
        finally:
            client.close()

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        }
    )

def create_workflow_chunk(event_data, model):
    """创建工作流响应块"""
    return {
        "id": event_data.get("workflow_run_id", ""),
        "object": "workflow.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": json.dumps(event_data.get("outputs", {}), ensure_ascii=False)
            }
        }]
    }

def handle_workflow_request(openai_request):
    """处理工作流请求"""
    try:
        model = openai_request.get("model")
        api_key = get_api_key(model)
        
        if not api_key:
            error_msg = f"Workflow {model} not found. Available workflows: {', '.join(model_manager.name_to_api_key.keys())}"
            logger.error(error_msg)
            return {
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                }
            }, 404

        workflow_request = transform_openai_to_workflow(openai_request)
        logger.info(f"Transformed workflow request: {json.dumps(workflow_request, ensure_ascii=False)}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        stream = openai_request.get("stream", False)
        workflow_endpoint = f"{DIFY_API_BASE}/workflows/run"
        logger.info(f"Sending request to workflow endpoint: {workflow_endpoint}, stream={stream}")

        if stream:
            return stream_workflow_response(workflow_endpoint, workflow_request, headers, model)
        else:
            return sync_workflow_response(workflow_endpoint, workflow_request, headers, model)
            
    except Exception as e:
        logger.exception("Workflow error occurred")
        return {
            "error": {
                "message": str(e),
                "type": "workflow_error",
            }
        }, 500

def handle_chat_request(openai_request):
    """处理聊天请求"""
    try:
        model = openai_request.get("model")
        api_key = get_api_key(model)
        
        if not api_key:
            error_msg = f"Model {model} not found"
            logger.error(error_msg)
            return {
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                }
            }, 404

        dify_request = transform_openai_to_dify(openai_request, "/chat/completions")
        logger.info(f"Transformed request: {json.dumps(dify_request, ensure_ascii=False)}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        stream = openai_request.get("stream", False)
        chat_endpoint = f"{DIFY_API_BASE}/chat-messages"
        logger.info(f"Sending request to Dify endpoint: {chat_endpoint}, stream={stream}")

        if stream:
            return stream_chat_response(chat_endpoint, dify_request, headers, model)
        else:
            return sync_chat_response(chat_endpoint, dify_request, headers, model)
            
    except Exception as e:
        logger.exception("Chat error occurred")
        return {
            "error": {
                "message": str(e),
                "type": "chat_error",
            }
        }, 500

def stream_chat_response(endpoint, request_data, headers, model):
    """处理 Chat 的流式响应"""
    def generate():
        client = httpx.Client(
            timeout=None,
            transport=httpx.HTTPTransport(
                proxy="http://127.0.0.1:10809"
            )
        )
        
        try:
            with client.stream('POST', endpoint, json=request_data, headers=headers) as response:
                if response.status_code != 200:
                    error_msg = response.json().get("message", "Unknown error")
                    logger.error(f"Chat API error: {error_msg}")
                    error_chunk = {
                        "error": {
                            "message": error_msg,
                            "type": "chat_error",
                            "code": response.status_code
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    return

                logger.info(f"Chat response status: {response.status_code}")
                
                # 读取响应内容
                for chunk in response.iter_text():
                    if not chunk:
                        continue
                        
                    # 处理每一行
                    for line in chunk.splitlines():
                        if not line or not line.startswith('data: '):
                            continue
                            
                        try:
                            event_data = json.loads(line[6:])  # 去掉 "data: " 前缀
                            logger.debug(f"Received chat event: {event_data}")
                            
                            # 处理消息事件
                            if event_data.get('event') in ['message', 'message_end']:
                                text = event_data.get('answer', '')
                                if text:
                                    # 构造 OpenAI 格式的响应
                                    chat_chunk = {
                                        "id": "chatcmpl-" + str(uuid.uuid4()),
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [
                                            {
                                                "delta": {"content": text},
                                                "index": 0,
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(chat_chunk)}\n\n"
                                    
                                # 如果是消息结束事件，发送结束标记
                                if event_data.get('event') == 'message_end':
                                    end_chunk = {
                                        "id": "chatcmpl-" + str(uuid.uuid4()),
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [
                                            {
                                                "delta": {},
                                                "index": 0,
                                                "finish_reason": "stop"
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(end_chunk)}\n\n"
                                    yield "data: [DONE]\n\n"
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse chat event data: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error in chat response streaming: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "chat_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            
        finally:
            client.close()

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        }
    )

def sync_chat_response(endpoint, request_data, headers, model):
    """处理同步聊天响应"""
    try:
        with httpx.Client(timeout=TIMEOUT, proxies={
            "http://": "http://127.0.0.1:10809",
            "https://": "http://127.0.0.1:10809"
        }) as client:
            response = client.post(
                endpoint,
                json=request_data,
                headers=headers
            )
            
            if response.status_code == 400:
                error_msg = response.json().get("message", "Bad Request")
                logger.error(f"Bad request: {error_msg}")
                return {
                    "error": {
                        "message": error_msg,
                        "type": "invalid_request_error",
                    }
                }, 400
                
            if response.status_code != 200:
                error_msg = f"Chat API error: {response.text}"
                logger.error(f"Request failed: {error_msg}")
                return {
                    "error": {
                        "message": error_msg,
                        "type": "chat_error",
                        "code": response.status_code
                    }
                }, response.status_code

            chat_response = response.json()
            logger.info(f"Received chat response: {json.dumps(chat_response, ensure_ascii=False)}")
            return transform_dify_to_openai(chat_response, model)
        
    except httpx.RequestError as e:
        error_msg = f"Failed to connect to chat API: {str(e)}"
        logger.error(error_msg)
        return {
            "error": {
                "message": error_msg,
                "type": "chat_error",
                "code": "connection_error"
            }
        }, 503

if __name__ == '__main__':
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", 5000))
    logger.info(f"Starting server on http://{host}:{port}")
    app.run(debug=True, host=host, port=port)
