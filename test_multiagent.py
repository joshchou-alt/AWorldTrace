#!/usr/bin/env python3
"""
Multi-Agent AWorld Test Script

This script demonstrates a simple multi-agent workflow with two agents:
- A researcher agent that gathers information
- A summarizer agent that creates concise summaries
"""

import os
from aworld.agents.llm_agent import Agent
from aworld.runner import Runners
from aworld.core.agent.swarm import Swarm

def main():
    print("🚀 Testing AWorld Multi-Agent System...")
    
    # Check if required environment variables are set
    required_vars = ['LLM_MODEL_NAME', 'LLM_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == '{YOUR_CONFIG}':
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing or unconfigured environment variables: {missing_vars}")
        print("Please edit your .env file and set the required values.")
        return False
    
    print(f"✅ Using model: {os.getenv('LLM_MODEL_NAME')}")
    
    try:
        # Create two agents
        researcher = Agent(
            name="Research Agent",
            system_prompt="You are a research specialist. Provide detailed, informative responses about topics."
        )
        
        summarizer = Agent(
            name="Summary Agent",
            system_prompt="You are a summarization expert. Take information and create clear, concise summaries."
        )
        
        # Create a multi-agent swarm (workflow: researcher -> summarizer)
        swarm = Swarm(topology=[(researcher, summarizer)])
        
        print("\n🤖 Running multi-agent workflow...")
        print("📋 Task: Research and summarize information about AI agents")
        
        # Run the multi-agent workflow
        result = Runners.sync_run(
            input="What are AI agents and what are their main capabilities? Please research this topic and then provide a clear summary.",
            swarm=swarm,
        )
        
        print(f"\n📖 Final Result: {result.answer}")
        print("\n✅ Multi-Agent System is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error running multi-agent system: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is correct")
        print("2. Verify your model name is valid")
        print("3. Ensure you have internet connection")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
