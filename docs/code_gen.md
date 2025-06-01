## Before You Start
This code generation and exec process is based on mcp protocal, if you are not familair with it check the link : https://modelcontextprotocol.io/introduction

## prepare the .env

create the .env file like below:

```
/neural-symbolic/mcp/.env
```
paste the API key like this:
```
XAI_API_KEY = 'xai-xxxxxxxxxxxxxM1g5hmPuvx6xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

## prepare the result json

Use the generated json from [here](./VQA.md)


```
python ./mcp/src/client.py ./mcp/src/server.py
```

paste your prompt like the following:

```
call generate_and_execute_code  provide json_path='somewhere_to_your_json/neural-symbolic/dataset/output_json/10120/info/315967196449927219-ls.json' when neccesary
```

you will see the log in ./execution_logs folder