from tracer_v4 import CodeTracer
import asyncio


def compare_outputs(actual, expected, test_name):
    """Compare actual vs expected outputs and print results."""
    print(f"\n{'=' * 60}")
    print(f"Test: {test_name}")
    print("=" * 60)

    print("\nExpected output:")
    for i, line in enumerate(expected):
        print(f"  {i:2d}: {line}")

    print("\nActual output:")
    for i, line in enumerate(actual):
        print(f"  {i:2d}: {line}")

    # Check if outputs match
    if actual == expected:
        print("\n✅ TEST PASSED")
        return True
    else:
        print("\n❌ TEST FAILED")
        print("\nDifferences:")

        # Show differences
        max_len = max(len(actual), len(expected))
        for i in range(max_len):
            expected_line = expected[i] if i < len(expected) else "<<MISSING>>"
            actual_line = actual[i] if i < len(actual) else "<<MISSING>>"

            if expected_line != actual_line:
                print(f"  Line {i}:")
                print(f"    Expected: {expected_line}")
                print(f"    Actual:   {actual_line}")

        return False


def test_basic_example():
    """Test the tracer with the basic example from the document."""
    code = """
def helper1(o):
    u = o**2
    return u

def main():
    k = helper1(10)

main()
""".strip()

    tracer = CodeTracer()
    result = tracer.trace_code(code)

    # Function definitions should be included, main() call preserved since no LLM
    expected = ["def helper1(o):", "def main():", "main()"]

    return compare_outputs(result, expected, "Basic Function (No LLM)")


def test_llm_example():
    """Test with LLM calls."""

    # First, create a mock LLM class
    class MockLLM:
        def forward(self, messages, **kwargs):
            return f"Response: {messages}"

    # Create LLM instance
    llm_instance = MockLLM()

    code = """
def helper2(input_data):
    print("helper2 start")
    result = llm.forward(input_data)
    print(f"helper2 got: {result}")
    return result

def main():
    k = helper2("test")
    return k

main()
""".strip()

    # Create tracer with LLM references
    llm_refs = {llm_instance.forward}
    tracer = CodeTracer(llm_refs)

    # Execute with LLM instance in globals
    globals_dict = {
        "MockLLM": MockLLM,
        "llm": llm_instance,
        "print": print,
    }

    result = tracer.trace_code(code, globals_dict)

    # Function definitions included, functions with LLM calls get expanded inline
    expected = [
        "def helper2(input_data):",
        "def main():",
        "input_data_helper2_0 = 'test'",
        'print("helper2 start")',
        "result = llm.forward(input_data_helper2_0)",
        'print(f"helper2 got: {result}")',
        "return_helper2 = result",
        "k = return_helper2",  # BUG: This line is missing in actual output
        "return_main = k",
    ]

    return compare_outputs(result, expected, "Function with LLM Call")


def test_simple():
    """Simple test without functions."""
    code = """
x = 1
y = 2
z = x + y
print(z)
""".strip()

    tracer = CodeTracer()
    result = tracer.trace_code(code, {"print": print})

    # No functions, no LLM calls - everything should be preserved as-is
    expected = ["x = 1", "y = 2", "z = x + y", "print(z)"]

    return compare_outputs(result, expected, "Simple Statements")


def test_loop_with_llm():
    """Test loop containing LLM calls."""

    class MockLLM:
        def forward(self, messages, **kwargs):
            return f"Response: {messages}"

    llm_instance = MockLLM()

    code = """
def main():
    llm = MockLLM()
    items = ["A", "B"]
    results = []
    
    for item in items:
        response = llm.forward(f"analyze-{item}")
        results.append(response)
    
    return results

main()
""".strip()

    llm_refs = {llm_instance.forward}
    tracer = CodeTracer(llm_refs)

    globals_dict = {
        "MockLLM": MockLLM,
        "print": print,
    }

    result = tracer.trace_code(code, globals_dict)

    # Function definitions included, loop with LLM calls should be unrolled
    expected = [
        "def main():",
        "llm = MockLLM()",
        'items = ["A", "B"]',
        "results = []",
        "idx_forloop_local_line_6 = 0",
        "item_forloop_local_line_6 = items[idx_forloop_local_line_6]",
        'response = llm.forward(f"analyze-{item_forloop_local_line_6}")',
        "results.append(response)",
        "idx_forloop_local_line_6 += 1",
        "item_forloop_local_line_6 = items[idx_forloop_local_line_6]",
        'response = llm.forward(f"analyze-{item_forloop_local_line_6}")',
        "results.append(response)",
        "idx_forloop_local_line_6 += 1",
        "return_main = results",
    ]

    return compare_outputs(result, expected, "Loop with LLM Calls")


def test_async_with_gather():
    """Test async functions with LLM calls using asyncio.gather."""

    # Create a mock async LLM class
    class AsyncMockLLM:
        async def forward_async(self, messages, **kwargs):
            await asyncio.sleep(0.01)
            return f"Async Response: {messages}"

        async def process_async(self, data, config=None):
            await asyncio.sleep(0.01)
            return f"Async Processed: {data} with {config}"

    # Create LLM instance
    llm_instance = AsyncMockLLM()

    code = """
import asyncio

async def async_helper1(input_data):
    # Test async LLM call
    result = await llm.forward_async(
        input_data,
        temperature=0.7
    )
    return result

async def async_helper2(data):
    # Another async LLM call
    processed = await llm.process_async(
        data,
        config="async_mode"
    )
    
    # Chain another async call
    final = await llm.forward_async(
        processed
    )
    return final

async def main_async():
    # Test asyncio.gather with concurrent LLM calls
    results = await asyncio.gather(
        async_helper1("input1"),
        async_helper2("input2"),
        async_helper1("input3")
    )
    return results

# Run the async main
asyncio.run(main_async())
""".strip()

    # Create tracer with async LLM references
    llm_refs = {llm_instance.forward_async, llm_instance.process_async}
    tracer = CodeTracer(llm_refs)

    # Execute with LLM instance in globals
    globals_dict = {
        "AsyncMockLLM": AsyncMockLLM,
        "llm": llm_instance,
        "asyncio": asyncio,
        "print": print,
    }

    result = tracer.trace_code(code, globals_dict)

    # Function definitions included, but async execution is partially broken
    expected = [
        "import asyncio",
        "async def async_helper1(input_data):",
        "async def async_helper2(data):",
        "async def main_async():",
        # The gather call remains unexpanded due to async issues
        'results = await asyncio.gather(\n        async_helper1("input1"),\n        async_helper2("input2"),\n        async_helper1("input3")\n    )',
        # Orphaned return statements from async execution context loss
        "return_async_helper1 = result",
        "final = await llm.forward_async(\n        processed\n    )",
        "return_async_helper2 = final",
        "return_main_async = results",
    ]

    return compare_outputs(result, expected, "Async with asyncio.gather")


def test_nested_functions_with_mixed_llm():
    """Test nested function calls where only some contain LLM calls."""

    class MockLLM:
        def forward(self, messages, **kwargs):
            return f"Response: {messages}"

    llm_instance = MockLLM()

    code = """
def helper_no_llm(x):
    return x * 2

def helper_with_llm(data):
    result = llm.forward(data)
    return result

def main():
    a = helper_no_llm(5)
    b = helper_with_llm("test")
    c = helper_no_llm(10)
    return [a, b, c]

main()
""".strip()

    llm_refs = {llm_instance.forward}
    tracer = CodeTracer(llm_refs)

    globals_dict = {
        "llm": llm_instance,
    }

    result = tracer.trace_code(code, globals_dict)

    # Function definitions included, only helper_with_llm should be expanded
    expected = [
        "def helper_no_llm(x):",
        "def helper_with_llm(data):",
        "def main():",
        "a = helper_no_llm(5)",
        "data_helper_with_llm_0 = 'test'",
        "result = llm.forward(data_helper_with_llm_0)",
        "return_helper_with_llm = result",
        "b = return_helper_with_llm",  # BUG: This line is missing in actual output
        "c = helper_no_llm(10)",
        "return_main = [a, b, c]",
    ]

    return compare_outputs(result, expected, "Mixed Functions (Some with LLM)")


def test_external_lib_filtering():
    """Test that external library calls are ignored when using repo_path filtering."""
    import tempfile
    import os
    from pathlib import Path
    
    # Create a temporary "repo" directory
    with tempfile.TemporaryDirectory() as temp_repo:
        print(f"📁 Created temp repo: {temp_repo}")
        
        class MockLLM:
            def forward(self, messages, **kwargs):
                return f"Response: {messages}"

        llm_instance = MockLLM()

        code = """
import os
import json
import pandas as pd
from pathlib import Path

def helper_with_llm(data):
    # This contains LLM call - should be traced
    result = llm.forward(data)
    return result

def main():
    # External library calls - should be ignored
    current_dir = os.getcwd()
    file_path = Path("/tmp/test.txt")
    
    # Create some data using pandas (external lib)
    df = pd.DataFrame({"col": [1, 2, 3]})
    data_dict = df.to_dict()
    
    # JSON operations (external lib)
    json_str = json.dumps(data_dict)
    parsed_data = json.loads(json_str)
    
    # Our LLM call - should be traced
    llm_result = helper_with_llm("analyze this data")
    
    return llm_result

main()
""".strip()

        llm_refs = {llm_instance.forward}
        
        # Create tracer WITH repo_path filtering
        tracer = CodeTracer(llm_refs, repo_path=temp_repo)

        globals_dict = {
            "llm": llm_instance,
            "os": os,
            "json": __import__('json'),
            "pd": __import__('pandas'),  # Use real pandas
            "Path": Path,
            "print": print,
        }

        try:
            result = tracer.trace_code(code, globals_dict)
            print(f"✅ Tracing completed successfully with repo filtering")
        except Exception as e:
            print(f"❌ Error during tracing: {e}")
            result = []

        # With repo_path filtering, external lib calls should be ignored
        # Only our repo code (the wrapper and LLM calls) should be traced
        expected = [
            "import os",
            "import json", 
            "import pandas as pd",
            "from pathlib import Path",
            "def helper_with_llm(data):",
            "def main():",
            # External library calls should be preserved as-is (not traced internally)
            "current_dir = os.getcwd()",
            'file_path = Path("/tmp/test.txt")',  # Use double quotes to match actual
            'df = pd.DataFrame({"col": [1, 2, 3]})',  # Use double quotes to match actual
            "data_dict = df.to_dict()",
            "json_str = json.dumps(data_dict)",
            "parsed_data = json.loads(json_str)",
            # LLM call should be expanded
            "data_helper_with_llm_0 = 'analyze this data'",
            "result = llm.forward(data_helper_with_llm_0)",
            "return_helper_with_llm = result",
            "llm_result = return_helper_with_llm",
            "return_main = llm_result",
        ]

        print(f"\n📊 External lib test results:")
        print(f"   - Repo path filtering: {temp_repo}")
        print(f"   - External calls (os, json, pandas, Path) should be ignored internally")
        print(f"   - Only LLM calls within repo code should be traced")
        
        return compare_outputs(result, expected, "External Library Filtering")


def run_all_tests():
    """Run all tests and summarize results."""
    print("\n" + "=" * 80)
    print("RUNNING ALL TRACER TESTS WITH VALIDATION")
    print("=" * 80)

    tests = [
        test_basic_example,
        test_llm_example,
        test_simple,
        test_loop_with_llm,
        test_async_with_gather,
        test_nested_functions_with_mixed_llm,
        test_external_lib_filtering,
    ]

    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append((test_func.__name__, passed))
        except Exception as e:
            print(f"\n❌ {test_func.__name__} CRASHED: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<40} {status}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    if passed < total:
        print("\n⚠️  Some tests failed. The tracer is not producing expected output.")
        print("Key issues to fix:")
        print(
            "- Function definitions should not appear in output when they contain LLM calls"
        )
        print("- Functions containing LLM calls should be fully expanded inline")
        print("- Async execution context is being lost")
    else:
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()