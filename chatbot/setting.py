import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
#model_name = 'Mizuiro-sakura/deberta-v2-base-japanese-finetuned-QA'

tokenizer = AutoTokenizer.from_pretrained('ku-nlp/deberta-v2-base-japanese')
model = AutoModelForQuestionAnswering.from_pretrained('Mizuiro-sakura/deberta-v2-base-japanese-finetuned-QAe')

# ファイルを読み込む関数を定義
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# テキストファイルのパスを指定
file_path = 'C:\\Users\\User\\.spyder-py3\\chatbot/book.txt'

# テキストファイルを読み込んで変数textに格納
text = read_file(file_path)

def get_answer(context, question):
    input_ids = tokenizer.encode(question, context)
    output = model(torch.tensor([input_ids]))
    start_index = torch.argmax(output.start_logits)
    end_index = torch.argmax(output.end_logits) + 1
    answer = tokenizer.decode(input_ids[start_index:end_index])
    return answer

def chat():
    print("チャットを開始します。質問を入力してください。終了するには 'exit' と入力してください。")
    while True:
        user_input = input("質問: ")
        if user_input.lower() == 'exit':
            print("チャットを終了します。")
            break
        else:
            answer = get_answer(text, user_input)
            print("回答:", answer)

# チャットを開始
chat()