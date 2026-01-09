import json
from datetime import datetime
from pathlib import Path
from typing import Optional

class MemoryManager:
    def __init__(self, memory_dir: str = "memory/stored"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def save_attempt(self, 
                     problem: str,
                     solution: str,
                     feedback: str = "pending",
                     verified: bool = False):
        """Save a solution attempt for future reference"""
        
        memory_record = {
            "timestamp": datetime.now().isoformat(),
            "problem": problem,
            "solution": solution,
            "feedback": feedback,
            "verified": verified,
            "useful": feedback == "correct"
        }
        
        # Create filename based on problem hash
        problem_hash = hash(problem) % 10000
        filepath = self.memory_dir / f"attempt_{problem_hash}_{datetime.now().timestamp()}.json"
        
        with open(filepath, 'w') as f:
            json.dump(memory_record, f, indent=2)
    
    def find_similar_problems(self, current_problem: str, threshold: float = 0.7) -> list:
        """Find similar previously solved problems"""
        from difflib import SequenceMatcher
        
        similar = []
        for file in self.memory_dir.glob("*.json"):
            with open(file, 'r') as f:
                record = json.load(f)
            
            similarity = SequenceMatcher(None, current_problem, record["problem"]).ratio()
            if similarity >= threshold and record["verified"]:
                similar.append({
                    "problem": record["problem"],
                    "solution": record["solution"],
                    "similarity": similarity
                })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)
