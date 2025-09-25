"""
Enhanced GAIA runner with detailed execution tracing.

This script runs GAIA benchmarks while capturing detailed execution traces
for performance analysis and optimization.
"""

import argparse
import json
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from dotenv import load_dotenv

from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.task import Task
from examples.gaia.prompt import system_prompt
from examples.gaia.utils import (
    add_file_path,
    load_dataset_meta,
    question_scorer,
    report_results,
)
# Import tracer_integration from the same directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tracer_integration import GaiaTracingAgent, TracingTaskRunner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_enhanced_logging(args):
    """Setup enhanced logging for tracing runs."""
    log_dir = Path(os.getenv("AWORLD_WORKSPACE", "~")) / "tracing_logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gaia_tracing_{timestamp}_{args.start}_{args.end}.log"
    
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Enhanced logging enabled: {log_file}")
    return log_file

def main():
    parser = argparse.ArgumentParser(description="GAIA benchmark with execution tracing")
    parser.add_argument("--start", type=int, default=0, help="Start index of the dataset")
    parser.add_argument("--end", type=int, default=5, help="End index of the dataset")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--q", type=str, help="Specific question ID")
    parser.add_argument("--skip", action="store_true", help="Skip processed questions")
    parser.add_argument("--trace-output", type=str, help="Output directory for traces")
    parser.add_argument("--high-performance", action="store_true", help="Use high-performance configuration")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Setup enhanced logging
    log_file = setup_enhanced_logging(args)
    
    # Setup output directory for traces
    if args.trace_output:
        trace_dir = Path(args.trace_output)
    else:
        trace_dir = Path(os.getenv("AWORLD_WORKSPACE", "~")) / "gaia_traces"
    
    trace_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting GAIA benchmark with tracing")
    logger.info(f"Range: {args.start}-{args.end}, Split: {args.split}")
    logger.info(f"Trace output: {trace_dir}")
    logger.info(f"High performance mode: {args.high_performance}")
    
    # Load dataset
    gaia_dataset_path = os.getenv("GAIA_DATASET_PATH", "dataset/GAIA/2023")
    full_dataset = load_dataset_meta(gaia_dataset_path, split=args.split)
    logger.info(f"Loaded {len(full_dataset)} questions from dataset")
    
    # Load MCP configuration
    try:
        with open(Path(__file__).parent / "mcp.json", mode="r", encoding="utf-8") as f:
            mcp_config: dict[dict[str, Any]] = json.loads(f.read())
            available_servers: list[str] = list(mcp_config.get("mcpServers", {}).keys())
            logger.info(f"MCP Available Servers: {available_servers}")
    except json.JSONDecodeError as e:
        logger.error(f"Error loading mcp.json: {e}")
        mcp_config = {}
        available_servers = []
    
    # Configure agent for optimal performance
    agent_config = AgentConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o" if not args.high_performance else "gpt-4o"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", 0.0))
    )
    
    if args.high_performance:
        logger.info("Configuring for high performance...")
        # Add any high-performance specific configurations here
        # e.g., different model, optimized prompts, etc.
    
    # Create tracing agent
    tracing_agent = GaiaTracingAgent(
        conf=agent_config,
        name="gaia_tracing_agent",
        system_prompt=system_prompt,
        mcp_config=mcp_config,
        mcp_servers=available_servers,
    )
    
    # Create tracing task runner
    task_runner = TracingTaskRunner(tracing_agent)
    
    # Load existing results
    results_file = Path(os.getenv("AWORLD_WORKSPACE", "~")) / "results.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results: List[Dict[str, Any]] = json.load(f)
    else:
        results: List[Dict[str, Any]] = []
    
    # Process dataset slice
    if args.q:
        dataset_slice = [item for item in full_dataset if item["task_id"] == args.q]
    else:
        dataset_slice = full_dataset[args.start:args.end]
    
    logger.info(f"Processing {len(dataset_slice)} questions")
    
    successful_traces = []
    failed_traces = []
    
    try:
        for i, dataset_item in enumerate(dataset_slice):
            task_id = dataset_item["task_id"]
            logger.info(f"Processing {i+1}/{len(dataset_slice)}: {task_id}")
            logger.info(f"Question: {dataset_item['Question'][:100]}...")
            logger.info(f"Level: {dataset_item['Level']}")
            
            # Skip if already processed and correct
            if args.skip:
                existing = next((r for r in results if r["task_id"] == task_id), None)
                if existing and existing.get("is_correct", False):
                    logger.info(f"Skipping already correct answer")
                    continue
            
            try:
                # Prepare question with file paths
                question = add_file_path(dataset_item, file_path=gaia_dataset_path, split=args.split)["Question"]
                
                # Create task
                task = Task(input=question, agent=tracing_agent, conf=TaskConfig())
                
                # Run with tracing
                task_trace = task_runner.run_task_with_tracing(task)
                
                # Extract answer
                task_result = task_trace["result"]
                if task_result:
                    match = re.search(r"<answer>(.*?)</answer>", task_result)
                    if match:
                        answer = match.group(1)
                    else:
                        answer = task_result
                        logger.warning(f"No <answer> tag found, using raw response")
                else:
                    answer = "No response"
                    logger.error(f"No result from task execution")
                
                # Score the answer
                correct_answer = dataset_item["Final answer"]
                is_correct = question_scorer(answer, correct_answer)
                
                logger.info(f"Agent answer: {answer}")
                logger.info(f"Correct answer: {correct_answer}")
                logger.info(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
                
                # Create result record
                result_record = {
                    "task_id": task_id,
                    "level": dataset_item["Level"],
                    "question": question,
                    "answer": correct_answer,
                    "response": answer,
                    "is_correct": is_correct,
                    "trace_metadata": {
                        "agent_traces": len(task_trace["agent_traces"]),
                        "has_llm_calls": any(
                            trace.get("trace_data", {}).get("llm_calls", [])
                            for trace in task_trace["agent_traces"]
                        )
                    }
                }
                
                # Update results
                existing_index = next(
                    (idx for idx, result in enumerate(results) if result["task_id"] == task_id),
                    None,
                )
                
                if existing_index is not None:
                    results[existing_index] = result_record
                else:
                    results.append(result_record)
                
                # Store trace data
                if is_correct:
                    successful_traces.append(task_trace)
                    logger.info(f"Stored successful trace")
                else:
                    failed_traces.append(task_trace)
                    logger.info(f"Stored failed trace")
                
            except Exception as e:
                logger.error(f"Error processing {task_id}: {e}")
                logger.error(traceback.format_exc())
                
                # Still try to capture the failed trace
                try:
                    task_trace = {
                        "task_id": task_id,
                        "task_input": dataset_item["Question"],
                        "success": False,
                        "error": str(e),
                        "agent_traces": tracing_agent.execution_traces.copy(),
                        "timestamp": datetime.now().isoformat()
                    }
                    failed_traces.append(task_trace)
                except:
                    pass
                
                continue
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Save results
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        # Save traces
        if successful_traces:
            success_trace_file = trace_dir / f"successful_traces_{timestamp}_{args.start}_{args.end}.json"
            with open(success_trace_file, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": {
                        "total_successful": len(successful_traces),
                        "timestamp": timestamp,
                        "range": f"{args.start}-{args.end}",
                        "split": args.split
                    },
                    "traces": successful_traces
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved {len(successful_traces)} successful traces to {success_trace_file}")
        
        if failed_traces:
            failed_trace_file = trace_dir / f"failed_traces_{timestamp}_{args.start}_{args.end}.json"
            with open(failed_trace_file, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": {
                        "total_failed": len(failed_traces),
                        "timestamp": timestamp,
                        "range": f"{args.start}-{args.end}",
                        "split": args.split
                    },
                    "traces": failed_traces
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved {len(failed_traces)} failed traces to {failed_trace_file}")
        
        # Save all task traces from runner
        all_traces_file = trace_dir / f"all_task_traces_{timestamp}_{args.start}_{args.end}.json"
        task_runner.save_all_traces(str(all_traces_file))
        
        # Report final results
        report_results(results)
        
        logger.info(f"Tracing run completed!")
        logger.info(f"Successful traces: {len(successful_traces)}")
        logger.info(f"Failed traces: {len(failed_traces)}")
        logger.info(f"Traces saved to: {trace_dir}")
        logger.info(f"Log file: {log_file}")

if __name__ == "__main__":
    main()
