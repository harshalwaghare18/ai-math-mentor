from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json

class VerifierAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    
    def verify(self, problem: str, solution: str):
        """Check if solution is correct and complete"""
        prompt = PromptTemplate(
            input_variables=["problem", "solution"],
            template="""Verify this solution:

Problem: {problem}

Solution: {solution}

Check:
1. Mathematical correctness
2. All steps are justified
3. Units/domain are valid
4. Edge cases considered
5. Answer is complete and clear

Return JSON:
{{
    "is_correct": true/false,
    "confidence": 0.95,
    "issues": ["issue 1", ...],
    "suggestions": ["suggestion 1", ...],
    "needs_human_review": false
}}"""
        )
        
        chain = prompt | self.llm
        response = chain.invoke({
            "problem": problem,
            "solution": solution
        })
        
        return json.loads(response.content)
