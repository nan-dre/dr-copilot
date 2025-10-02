import dspy
import json
import random
from typing import Dict, List, Any, Tuple, Optional
from dspy.teleprompt import BootstrapFewShot
from optimizers.base import BaseOptimizer

random.seed(0)

class BootstrapFewshotOptimizer(BaseOptimizer):
    def __init__(self, metric_fn, k: int = 4, field=None, field_type=None):
        self.k = k
        self.metric_fn = metric_fn
        self.teleprompter = BootstrapFewShot(metric=metric_fn)

    def optimize(
        self, module, train_set, val_set
    ) -> Tuple[dspy.Module, List, float, float]:
        print(
            f"Starting optimization with {len(train_set)} training examples and {len(val_set)} validation examples..."
        )

        compiled_module = self.teleprompter.compile(
            student=module,
            trainset=train_set,
        )

        if self.metric_fn and val_set:
            print("\nEvaluating base model vs optimized model on validation set...")
            
            base_score, results, all_scores = dspy.Evaluate(
                devset=val_set,
                metric=self.metric_fn,
                display_progress=True,
                return_outputs=True, 
                return_all_scores=True
            )(module)
            results = list(map(lambda result: (result[0].toDict(), result[1].toDict(), result[2]), results))
            
            optimized_score, optimized_results, optimized_all_scores = dspy.Evaluate(
                devset=val_set,
                metric=self.metric_fn,
                display_progress=True,
                return_outputs=True,
                return_all_scores=True
            )(compiled_module)
            optimized_results = list(map(lambda result: (result[0].toDict(), result[1].toDict(), result[2]), optimized_results))
            
            print(f"\nBase model score: {base_score:.4f}")
            print(f"Optimized model score: {optimized_score:.4f}")
            print(f"Improvement: {optimized_score - base_score:.4f})")
        
        return compiled_module, base_score, results, all_scores, optimized_score, optimized_results, optimized_all_scores