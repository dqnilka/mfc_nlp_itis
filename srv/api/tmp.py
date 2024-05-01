import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "IlyaGusev/saiga2_7b_lora"
BASE_MODEL_PATH = "TheBloke/Llama-2-7B-fp16"
# BASE_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "ТТы — Мария, русскоязычный автоматический ассистент Многофункционального центра предоставления государственных и муниципальных услуг. Ты разговариваешь с людьми и помогаешь им."

class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        start_token_id=1,
        bot_token_id=9225
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    # load_in_8bit=True,
    torch_dtype=torch.float32,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float32
)
model.eval()

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)


inp = "Почему небо голубое? Ответь коротко в одно предложение !"
conversation = Conversation()
conversation.add_user_message(inp)
prompt = conversation.get_prompt(tokenizer)

output = generate(model, tokenizer, prompt, generation_config)
print(inp)
print(output)
print()
print("==============================")
print()