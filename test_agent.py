#!/usr/bin/env python3
"""
Simple AWorld Agent Test Script

This script creates a basic agent and tests if everything is working.
Make sure you have configured your .env file with your API credentials.
"""

import os
from aworld.agents.llm_agent import Agent
from aworld.runner import Runners

def main():
    print("🚀 Testing AWorld Agent...")
    
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
    print(f"✅ API key configured: {'*' * 10}{os.getenv('LLM_API_KEY')[-4:]}")
    
    try:
        # Create a simple agent
        agent = Agent(
            name="Test Agent",
            system_prompt="You are a helpful assistant. Keep responses brief and friendly."
        )
        
        print("\n🤖 Running agent test...")
        
        # Test the agent with a simple question
        result = Runners.sync_run(
            input="Say hello and tell me you're working correctly in exactly one sentence.",
            agent=agent,
        )
        
        print(f"\n🎉 Agent Response: {result.answer}")
        print("\n✅ AWorld is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error running agent: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is correct")
        print("2. Verify your model name is valid")
        print("3. Ensure you have internet connection")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
