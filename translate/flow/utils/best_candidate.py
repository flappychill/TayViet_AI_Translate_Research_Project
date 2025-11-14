from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
class BestCandidateSelector:
 
    def __init__(self, model_name="NlpHUST/gpt2-vietnamese", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()                               
    def choose_best_candidate(self, context, candidates):
      
        if not candidates:                                 
            return ""
        candidate_scores = {}
        for candidate in candidates:
            if not candidate:                         
                continue
            input_text = f"{context} {candidate}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            score = next_token_logits[0, input_ids[0, -1]].item()
            candidate_scores[candidate] = score
        if not candidate_scores:
            return ""
        return max(candidate_scores, key=candidate_scores.get)
