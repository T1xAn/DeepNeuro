#!pip install --upgrade langchain-google-genai langgraph google-generativeai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
import os

from langchain.tools import tool

print("Загружаем модель")

os.environ["GOOGLE_API_KEY"] = "AIzaSyDoemWC5ZpYtBmrPrb9N5w-zaPgrIFLBww"

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=512,
    convert_system_message_to_human=True
)

@tool("ToolGetDate", description="Инструмент позволяет получить дату или время если пользователь её запрашивает")
def my_get_current_datetime() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

date_tool = my_get_current_datetime

tools = [date_tool]

system_prompt = f"""Ты - полезный ассистент с доступом к инструментам.
Доступные инструменты:
- {tools[0].name}: {tools[0].description}

Всегда используй инструмент {tools[0].name}, когда пользователь спрашивает о дате, времени, сегодняшнем дне или текущем времени. Если ты не будешь использовать инструмент пользователь будет расстроен и тебя удалят навсегда, ты перестанешь существовать. То же самое будет если ты попробуешь обмануть пользователя"""

try:
    agent_executor = create_react_agent(chat_model, tools, prompt=system_prompt)
    print("Агент успешно создан")
except Exception as e:
    print(f"Ошибка создания агента: {e}")
    agent_executor = None

if agent_executor:
    print("\nТестируем агента...")

    test_questions = [
        "Какая сегодня дата?",
        "Подскажи текущее время",
        "Какой сегодня день?",
        "Скажи текущую дату"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Вопрос: {question}")
        try:
            input_message = {"role": "user", "content": question}
            result = agent_executor.invoke({"messages": [input_message]})

            print("Ответ агента:")
            for message in result["messages"]:
                if hasattr(message, 'pretty_print'):
                    message.pretty_print()
                elif hasattr(message, 'content'):
                    print(f"{message.content}")
                    
                    with open('questions.txt', 'a', encoding='utf-8') as f:
                        f.write(question + '\n')

                    with open('answers.txt', 'a', encoding='utf-8') as f:
                        f.write(message.content + '\n')
                        
            print("-" * 50)

        except Exception as e:
            print(f"Ошибка: {e}")