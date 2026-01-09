from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json

class ExplainerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5)  # Allow clarity
    
    def explain(self, problem: str, solution: dict):
        """Create student-friendly explanation"""
        prompt = PromptTemplate(
            input_variables=["problem", "solution"],
            template="""Create a clear, student-friendly explanation.

Problem: {problem}

Solution: {solution}

Write:
1. Why this approach works
2. Key concepts involved
3. Common mistakes to avoid
4. Real-world context (if applicable)
5. Practice problems to try

Use simple language. Explain every step."""
        )
        
        chain = prompt | self.llm
        response = chain.invoke({
            "problem": problem,
            "solution": json.dumps(solution, indent=2)
        })
        
        return response.content
