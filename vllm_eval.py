from vllm import LLM, SamplingParams
import datasets
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer
from evaluate.evaluate_utils.grader import grade_answer
import argparse
import json
import os

def extract_solution(solution_str):
    if solution_str is None:
        return '[invalid answer]'
    boxed_str = last_boxed_only_string(solution_str)
    if boxed_str is None:
        return '[invalid answer]'
    return remove_boxed(boxed_str)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model on MATH dataset')
    parser.add_argument('--model', type=str, default="model/RecQwen-0.5B-8844-RL-merged",
                      help='Path to the model')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature')
    parser.add_argument('--n', type=int, default=4,
                      help='Number of samples to generate per question')
    parser.add_argument('--max_tokens', type=int, default=1024,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--output_dir', type=str, default="results",
                      help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory using model name
    model_name = os.path.basename(args.model)
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    llm = LLM(model=args.model, trust_remote_code=True, task="generate")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_samples = dataset["test"]

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n=args.n
    )

    # Prepare all prompts in batch
    prompts = []
    for example in test_samples:
        problem = example["problem"]
        chat = [
            {"role": "user", "content": problem + r" Let's think step by step and output the final answer within \boxed{}."}
        ]
        messages = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(messages)

    # Generate answers in batch
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # Process and print results
    total_correct = 0
    total_answers = 0
    questions_with_correct = 0
    
    # Store detailed results
    detailed_results = []

    print("\n=== Results ===")
    for idx, (output, example) in enumerate(zip(outputs, test_samples)):
        correct_answer = extract_solution(example["solution"])
        question_correct = False
        
        # Debug: Check number of generated answers
        if len(output.outputs) != args.n:
            print(f"Warning: Question {idx} generated {len(output.outputs)} answers instead of {args.n}")
        
        question_results = {
            "question_idx": idx,
            "problem": example["problem"],
            "correct_answer": correct_answer,
            "generated_answers": [],
            "is_correct": False
        }
        
        if idx < 2:
            print(f"\nQuestion {idx + 1}:")
            print(f"User: {example['problem']}")
        
        for generated_output in output.outputs:
            generated_answer = extract_solution(generated_output.text)
            
            is_correct = grade_answer(generated_answer, correct_answer)
            if is_correct:
                question_correct = True
                total_correct += 1
            total_answers += 1
            
            answer_result = {
                "answer": generated_answer,
                "is_correct": is_correct
            }
            question_results["generated_answers"].append(answer_result)
            
            if idx < 2:
                print(f"Assistant: {generated_answer}")
                print(f"Status: {'✓' if is_correct else '✗'}")
        
        question_results["is_correct"] = question_correct
        detailed_results.append(question_results)
        
        if question_correct:
            questions_with_correct += 1
            
        if idx < 2:
            print(f"Correct Answer: {correct_answer}")
            print("-" * 80)

    # Calculate metrics
    coverage = (questions_with_correct/len(test_samples))*100
    average_at_1 = (total_correct/total_answers)*100

    # Print final statistics
    print("\n=== Final Statistics ===")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Samples per question: {args.n}")
    print(f"Total Questions: {len(test_samples)}")
    print(f"Total Answers: {total_answers}")
    print(f"Correct Answers: {total_correct}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Average@1: {average_at_1:.2f}%")

    # Save results
    results = {
        "config": {
            "model": args.model,
            "temperature": args.temperature,
            "samples_per_question": args.n,
            "max_tokens": args.max_tokens
        },
        "metrics": {
            "total_questions": len(test_samples),
            "total_answers": total_answers,
            "correct_answers": total_correct,
            "coverage": coverage,
            "average_at_1": average_at_1
        },
        "detailed_results": detailed_results
    }

    # Save to JSON file with temperature in filename
    output_file = os.path.join(output_dir, f"temp{args.temperature}_n{args.n}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()