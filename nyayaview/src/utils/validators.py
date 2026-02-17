def is_valid_query(query):
    if not query or not isinstance(query, str):
        return False, "Query must be a non-empty string."
    if len(query) < 5:
        return False, "Query must be at least 5 characters long."
    return True, ""

def is_valid_threshold(threshold):
    if not isinstance(threshold, (int, float)):
        return False, "Threshold must be a number."
    if threshold < 0 or threshold > 1:
        return False, "Threshold must be between 0 and 1."
    return True, ""