from types import SimpleNamespace

from agent.demo import print_banner, print_tool_trace


def test_print_banner(capsys):
    print_banner()

    output = capsys.readouterr().out

    assert "SVHN ML System Analyst Agent" in output
    assert "Type 'quit' or 'exit' to stop." in output


def test_print_tool_trace_with_no_results(capsys):
    print_tool_trace([])

    output = capsys.readouterr().out

    assert "[trace] No MCP tools executed." in output


def test_print_tool_trace_displays_tool_result(capsys):
    tool_result = SimpleNamespace(
        tool_name="get_digit_metrics",
        structured_content={
            "class_label": 3,
            "precision": 0.9615,
        },
        is_error=False,
        error_message=None,
    )

    print_tool_trace([tool_result])

    output = capsys.readouterr().out

    assert "[trace] MCP tool execution:" in output
    assert "1. get_digit_metrics" in output
    assert "'class_label': 3" in output
    assert "'precision': 0.9615" in output

def test_print_tool_trace_displays_tool_error(capsys):
    tool_result = SimpleNamespace(
        tool_name="get_evaluation_summary",
        structured_content=None,
        is_error=True,
        error_message="Evaluation data unavailable.",
    )

    print_tool_trace([tool_result])

    output = capsys.readouterr().out

    assert "[trace] MCP tool execution:" in output
    assert "1. get_evaluation_summary" in output
    assert "error: Evaluation data unavailable." in output