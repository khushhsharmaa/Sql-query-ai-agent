from backend.agent import validate_sql_node

def test_select_is_allowed():
    result = validate_sql_node({"sql": "SELECT name FROM employees"})
    assert result["validation_error"] == ""

def test_delete_is_rejected():
    result = validate_sql_node({"sql": "DELETE FROM employees"})
    assert result["validation_error"]

def test_multiple_statements_are_rejected():
    result = validate_sql_node({"sql": "SELECT * FROM employees; DELETE FROM employees"})
    assert result["validation_error"]
