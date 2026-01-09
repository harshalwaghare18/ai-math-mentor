import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

class ParsedProblem(BaseModel):
    problem_text: str
    topic: str  # algebra, probability, calculus, linear_algebra
    variables: list[str]
    constraints: list[str]
    additional_context: str
    needs_clarification: bool
    clarification_questions: list[str]

class ParserAgent:
    def __init__(self, model: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["raw_input"],
            template="""You are a precise math problem parser.
            
Raw input (from OCR/ASR/typing):
{raw_input}

Task:
1. Clean up the text
2. Identify the math topic
3. Extract variables and constraints
4. Identify if there's ambiguity

Return ONLY valid JSON (no markdown):
{{
    "problem_text": "cleaned problem statement",
    "topic": "algebra|probability|calculus|linear_algebra",
    "variables": ["x", "y"],
    "constraints": ["x > 0", "y ≥ 0"],
    "additional_context": "any assumptions or context",
    "needs_clarification": false,
    "clarification_questions": []
}}"""
        )
    
    def parse(self, raw_input: str) -> ParsedProblem:
        """Parse raw input into structured problem"""
        chain = self.prompt | self.llm
        response = chain.invoke({"raw_input": raw_input})
        
        try:
            parsed = json.loads(response.content)
            return ParsedProblem(**parsed)
        except json.JSONDecodeError:
            # Fallback: extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return ParsedProblem(**parsed)
            else:
                raise ValueError(f"Failed to parse: {response.content}")
