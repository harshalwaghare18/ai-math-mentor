from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rag.knowledge_base import MathKnowledgeBase
import json

class SolverAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.kb = MathKnowledgeBase()
        self.kb.load_index()
    
    def solve(self, parsed_problem: dict, route_info: dict):
        """Solve the problem using RAG context + reasoning"""
        
        # Retrieve relevant docs
        retrieved_docs = self.kb.retrieve(
            parsed_problem.get("problem_text", ""),
            k=3
        )
        
        context = "\n---\n".join([doc[0].page_content for doc in retrieved_docs])
        
        prompt = PromptTemplate(
            input_variables=["problem", "context"],
            template="""You are an expert math tutor. Solve this problem step-by-step.

Problem:
{problem}

Relevant knowledge:
{context}

Provide:
1. Solution approach
2. Step-by-step calculation
3. Final answer with proper formatting
4. Confidence in solution (0-1)

Return JSON:
{{
    "approach": "explanation of approach",
    "steps": [
        {{"step": 1, "description": "...", "calculation": "..."}},
        ...
    ],
    "final_answer": "the answer",
    "confidence": 0.95,
    "sources": ["formula reference", ...]
}}"""
        )
        
        chain = prompt | self.llm
        response = chain.invoke({
            "problem": str(parsed_problem),
            "context": context
        })
        
        solution = json.loads(response.content)
        solution["retrieved_sources"] = retrieved_docs
        
        return solution
