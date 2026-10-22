import os
import sys
from dotenv import load_dotenv


from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


os.environ['LANGCHAIN_PROJECT']="llm structured DEMO"

load_dotenv()



# 1. Define the Pydantic schema using pydantic
class evalschema(BaseModel):
    feedback: str = Field(description="Detailed feedback on the essay")
    score: float = Field(description="Score out of 10 of the essay", ge=0, le=10)


modellm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=2048,
)

# 2. Assign the structured output runnable
evaluator = modellm.with_structured_output(evalschema)

essaytopic = "write an essay on computer"
essay = """A computer is one of the most important inventions of modern science and technology. It is an electronic device that accepts data, processes it, and produces meaningful information. Computers are used in almost every field of life, including education, healthcare, business, banking, and entertainment.

Computers make our work faster, easier, and more accurate. In schools and colleges, students use computers to attend online classes, prepare assignments, and gain knowledge. In offices, computers help employees manage data, prepare reports, and communicate with others. Doctors use computers to maintain patient records and assist in medical examinations.

Computers also play an important role in communication. Through the internet, people can connect with others across the world, share information, and access various online services. In addition, computers are essential for scientific research, space exploration, and the development of artificial intelligence.

However, computers also have some disadvantages. Excessive use can affect our health, reduce physical activity, and lead to addiction. Cybercrime, data theft, and online fraud are other major concerns. Therefore, computers should be used responsibly and safely.

In conclusion, computers have become an essential part of our daily lives. They have transformed the way we learn, work, and communicate. When used wisely, computers can help us improve our knowledge, productivity, and quality of life."""

# 3. Invoke the structured evaluator
response = evaluator.invoke(f"Evaluate the essay based on the topic '{essaytopic}':\n\n{essay}")
print(f"Score: {response.score}")
print(f"Feedback:\n{response.feedback}")


