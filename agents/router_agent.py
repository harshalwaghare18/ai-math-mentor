from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json

class RouterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    
    def route(self, parsed_problem: dict):
        """Determine which solver approach to use"""
        prompt = PromptTemplate(
            input_variables=["problem"],
            template="""Given this math problem:
{problem}

Determine:
1. What type of solver strategy is needed?
2. Should we use RAG retrieval for similar problems?
3. Do we need computational tools (calculator, solver)?
4. Confidence level (0-1) in approach

Return JSON:
{{
    "strategy": "algebraic_manipulation|calculus_based|probability|linear_system",
    "use_rag": true/false,
    "computational_tools": ["calculator", "solver"],
    "confidence": 0.9
}}"""
        )
        
        chain = prompt | self.llm
        response = chain.invoke({"problem": str(parsed_problem)})
        
        return json.loads(response.content)
