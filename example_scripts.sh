# example of scripts 

# Ministral small on evaluated on the spain region with questions and options in spanish
uv run eval_mcq --model ministral-small --provider mistral --region es-es --lang o

# Ministral small evaluated on the spain region with questions and options in english
uv run eval_mcq --model ministral-small --provider mistral --region es-es --lang en

# Llama3-7B evaluated on 100 questions about brazilian culture with questions and options in portuguese 
uv run eval_mcq --model llama3-7b --provider openai --region pt-br --lang o --limit 100