import dspy
import json
import random
from typing import Dict, List, Any, Tuple, Optional
from dspy.teleprompt import LabeledFewShot
from optimizers.base import BaseOptimizer

random.seed(0)

class FewShotOptimizer(BaseOptimizer):
    def __init__(self, k: int = 4, field=None, field_type=None, metric_fn=None):
        self.k = k
        self.metric_fn = metric_fn
        self.teleprompter = LabeledFewShot(k=k)

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

    def save_demos(self, demos: List, path: str) -> None:
        """
        Save the selected demos to a file.

        Args:
            demos: List of demos to save
            path: Path to save the demos
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(demos, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(demos)} demo examples to {path}")


class SelectiveFewShotOptimizer(BaseOptimizer):
    def __init__(self, k: int = 4, field=None, field_type=None, metric_fn=None):
        self.k = k
        self.metric_fn = metric_fn
        self.field = field
        self.field_type = field_type
        self.teleprompter = LabeledFewShot(k=k)

    def optimize(
        self, module, train_set, val_set
    ) -> Tuple[dspy.Module, List, float, float]:
        print(
            f"Starting optimization with {len(train_set)} training examples and {len(val_set)} validation examples..."
        )

        # Select relevant examples from the trainset
        relevant_examples = []
        random.shuffle(train_set)

        # Skip selection if field is not specified
        if not self.field or not self.field_type:
            relevant_examples = train_set[:self.k]
        elif self.field_type == "bool":
            true_examples = []
            false_examples = []
            for sample in train_set:
                field_value = getattr(sample, self.field, None)
                if field_value is True:
                    true_examples.append(sample)
                else:
                    false_examples.append(sample)
            
            relevant_examples = (
                true_examples[: self.k // 2] + false_examples[: self.k // 2]
            )
        elif self.field_type == "int":
            examples_by_value = {}
            for sample in train_set:
                field_value = getattr(sample, self.field, None)
                if field_value is not None:
                    if field_value not in examples_by_value:
                        examples_by_value[field_value] = []
                    examples_by_value[field_value].append(sample)
            
            unique_values = list(examples_by_value.keys())
            num_values = len(unique_values)
            
            if num_values == 0:
                relevant_examples = train_set[:self.k]
            else:
                examples_per_value = max(1, self.k // num_values)
                relevant_examples = []
                
                for value in sorted(unique_values):
                    value_examples = examples_by_value[value][:examples_per_value]
                    relevant_examples.extend(value_examples)
        else:
            # Default case: just take first k examples
            relevant_examples = train_set[:self.k]

        compiled_module = self.teleprompter.compile(
            student=module, trainset=relevant_examples, sample=False
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
