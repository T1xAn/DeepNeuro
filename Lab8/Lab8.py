import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def setup_model():
    
    print("Загрузка локальной модели...")

    local_model_path = "modelSmall"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True        )
        
        if torch.cuda.is_available():
            print("GPU доступен, используем ускорение")
            torch_dtype = torch.float16
            device_map = "auto"
        else:
            print("GPU не доступен, используем CPU")
            torch_dtype = torch.float32
            device_map = None
        
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path, torch_dtype=torch_dtype,
            device_map=device_map, trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            model = model.to("cpu")
        
        print(f"Модель успешно загружена из: {local_model_path}")
        print(f"Размер модели: {model.num_parameters():,} параметров")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None, None

def analyze_with_model(article_text, model, tokenizer):
    
    system_prompt = """Инструкции:
1. Отвечай ТОЛЬКО на основе информации из предоставленного текста
2. Если информация не найдена, так и напиши
3. Будь точным в датах и именах
4. Используй формат JSON для ответа"""

    full_prompt = f"""{system_prompt}

ТЕКСТ ДЛЯ АНАЛИЗА:
{article_text}

ВОПРОСЫ:
1. В каком году была обозначена проблема взрывающихся градиентов (exploding gradients)?
2. Кто в 1891 году разработал метод уничтожающей производной (destroying derivative)?
3. Кто предложил цепное правило дифференцирования (chain rule of differentiation) и в каком году?

Ответь в формате JSON:
{{
    "gradient_explosion_year": "год или 'не найдено'",
    "destroying_derivative_author": "имя или 'не найдено'",
    "chain_rule_author": "имя или 'не найдено'",
    "chain_rule_year": "год или 'не найдено'"
}}"""

    print("Отправка запроса к модели")
    
    try:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        if not torch.cuda.is_available():
            generation_config["max_new_tokens"] = 256  # Меньше для скорости на CPU
        
        generated_ids = model.generate(**model_inputs, **generation_config)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response, full_prompt
        
    except Exception as e:
        print(f"Ошибка при генерации ответа: {e}")
        return None, full_prompt

def save_prompts_and_responses(prompt, response, filename_prefix="LLM"):
    
    prompt_filename = f"{filename_prefix}_prompt.txt"
    with open(prompt_filename, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    response_filename = f"{filename_prefix}_response.txt"
    with open(response_filename, 'w', encoding='utf-8') as f:
        f.write(response)
    
    print(f"Промпт сохранен в: {prompt_filename}")
    print(f"Ответ сохранен в: {response_filename}")
    
    return prompt_filename, response_filename

def parse_response(response):
    
    try:
        
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        json_str = response[start_idx:end_idx]
        data = json.loads(json_str)
        return data

    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON: {e}")
        print(f"Полный ответ: {response}")
        return None

def main():

    print("\n1.Чтение файла")
    article_text =  open("ENG_article.txt", 'r', encoding='utf-8').read()
    
    print(f"Прочитано символов: {len(article_text)}")

    print("\n2. Настройка локальной модели...")
    model, tokenizer = setup_model()
    
    # Анализируем статью
    print("\n3. Работа с моделью")
    llm_response, full_prompt = analyze_with_model(article_text, model, tokenizer)
    
    if llm_response:
        print("Полученый ответ:")
        print(llm_response)
        
        save_prompts_and_responses(full_prompt, llm_response)

        print("Результаты:")
        
        results = parse_response(llm_response)
        
        if results:
            print(f"Год проблемы взрывающихся градиентов: {results.get('gradient_explosion_year', 'не найдено')}")
            print(f"Автор метода уничтожающей производной (1891): {results.get('destroying_derivative_author', 'не найдено')}")
            print(f"Автор цепного правила: {results.get('chain_rule_author', 'не найдено')}")
            print(f"Год цепного правила: {results.get('chain_rule_year', 'не найдено')}")
        else:
            print("Не удалось извлечь структурированные данные из ответа")
            
    else:
        print("Не удалось получить ответ от модели")

if __name__ == "__main__":
    main()