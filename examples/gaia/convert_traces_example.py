#!/usr/bin/env python3
"""
Example script showing how to convert GAIA traces to executable notebooks.

This script demonstrates how to use the TraceToNotebookConverter to transform
execution traces into step-by-step reproducible Jupyter notebooks.
"""

import os
import sys
from pathlib import Path
from trace_to_notebook import TraceToNotebookConverter

def main():
    """Convert GAIA traces to notebooks - example usage."""
    
    # Setup paths
    trace_dir = Path("~/gaia_traces").expanduser()
    output_dir = Path("./generated_notebooks")
    
    print("🚀 GAIA Trace to Notebook Converter - Example Usage")
    print("=" * 60)
    
    # Check if trace directory exists
    if not trace_dir.exists():
        print(f"❌ Trace directory not found: {trace_dir}")
        print("Please update the trace_dir path to point to your trace files.")
        return 1
    
    # Find trace files
    trace_files = list(trace_dir.glob("*.json"))
    if not trace_files:
        print(f"❌ No trace files found in: {trace_dir}")
        return 1
    
    print(f"📁 Found {len(trace_files)} trace files:")
    for trace_file in trace_files:
        print(f"  - {trace_file.name}")
    
    # Create converter
    converter = TraceToNotebookConverter(str(output_dir))
    
    # Convert each trace file
    total_notebooks = 0
    successful_conversions = 0
    
    for trace_file in trace_files:
        print(f"\n🔄 Converting: {trace_file.name}")
        
        try:
            notebooks = converter.convert_trace_file(str(trace_file))
            total_notebooks += len(notebooks)
            successful_conversions += 1
            
            print(f"  ✅ Generated {len(notebooks)} notebooks")
            for notebook in notebooks[:3]:  # Show first 3
                print(f"    📓 {Path(notebook).name}")
            if len(notebooks) > 3:
                print(f"    ... and {len(notebooks) - 3} more")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n🎉 Conversion Complete!")
    print(f"📊 Statistics:")
    print(f"  - Files processed: {successful_conversions}/{len(trace_files)}")
    print(f"  - Notebooks generated: {total_notebooks}")
    print(f"  - Output directory: {output_dir}")
    
    if total_notebooks > 0:
        print(f"\n📝 Next steps:")
        print(f"  1. Install Jupyter: pip install jupyter")
        print(f"  2. Start Jupyter: jupyter notebook {output_dir}")
        print(f"  3. Open and run the generated notebooks")
        print(f"\n💡 Each notebook contains:")
        print(f"  - Task setup and metadata")
        print(f"  - Step-by-step agent actions")
        print(f"  - Executable mock tool calls")
        print(f"  - Results analysis")
    
    return 0

if __name__ == "__main__":
    exit(main())
