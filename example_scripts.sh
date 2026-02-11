# example of scripts 

# Ministral small on evaluated on the spain region with questions and options in spanish
python eval_mcq.py --model ministral-small --provider mistral --region es-es --lang o

# Ministral small evaluated on the spain region with questions and options in english
python eval_mcq.py --model ministral-small --provider mistral --region es-es --lang en

# Llama3-7B evaluated on 100 questions about brazilian culture with questions and options in portuguese 
python eval_mcq.py --model llama3-7b --provider openai --region pt-br --lang o --limit 100