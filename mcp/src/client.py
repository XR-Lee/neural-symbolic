import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env
XAI_API_KEY = os.getenv("XAI_API_KEY")

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )
    # methods will go here
        
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        # Convert tools to dicts with required fields
        for i, tool in enumerate(tools):
            tools[i] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
        print("\nConnected to server with tools:", [tool["function"]["name"] for tool in tools])
        self.tools = tools

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        available_tools = self.tools
        
        response = self.client.chat.completions.create(
            model="grok-3-beta",
            # max_tokens=1000,
            messages=messages,
            tools=available_tools
            )
        # Initial Claude API call
        final_text = []

        
        message = response.choices[0].message
        print(f"\nresponse: {message}")
        if message.tool_calls:
            print(f"\n[Tool call] {message.tool_calls}")
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                print(f"\n[Tool call] {tool_name}")
                tool_args = json.loads(tool_call.function.arguments)

                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content
                })

                # Get next response from Claude
                response = self.client.chat.completions.create(
                    model="grok-3-beta",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto"
                )

                message = response.choices[0].message
                final_text.append(message.content or "")
        else:
            final_text.append(message.content or "")

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        # while True:
        try:
            query = input("\nQuery: ").strip()

            if query.lower() == 'quit':
                # break
                return

            response = await self.process_query(query)
            print("\n" + response)

        except Exception as e:
            print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
     
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())