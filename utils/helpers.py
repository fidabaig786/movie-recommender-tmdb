"""Helper utilities for JSON parsing and string processing."""

import json
import ast


def parse_json_column(data):
    """
    Parse JSON-like string column.
    
    Args:
        data: String representation of JSON list of dicts
        
    Returns:
        List of extracted names, or empty list if parsing fails
    """
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return [item.get('name', '') for item in parsed]
        except (json.JSONDecodeError, ValueError, TypeError):
            try:
                parsed = ast.literal_eval(data)
                if isinstance(parsed, list):
                    return [item.get('name', '') for item in parsed]
            except (ValueError, SyntaxError, TypeError):
                return []
    return []


def extract_cast(cast_data, top_n=3):
    """
    Extract top N cast members.
    
    Args:
        cast_data: String representation of cast JSON
        top_n: Number of top cast members to extract
        
    Returns:
        List of top N cast member names
    """
    cast_list = parse_json_column(cast_data)
    return cast_list[:top_n]


def extract_director(crew_data):
    """
    Extract director from crew JSON.
    
    Args:
        crew_data: String representation of crew JSON
        
    Returns:
        Director name, or empty string if not found
    """
    if isinstance(crew_data, str):
        try:
            parsed = json.loads(crew_data)
            if isinstance(parsed, list):
                for member in parsed:
                    if member.get('job') == 'Director':
                        return member.get('name', '')
        except (json.JSONDecodeError, ValueError, TypeError):
            try:
                parsed = ast.literal_eval(crew_data)
                if isinstance(parsed, list):
                    for member in parsed:
                        if member.get('job') == 'Director':
                            return member.get('name', '')
            except (ValueError, SyntaxError, TypeError):
                pass
    return ''


def clean_string(text):
    """
    Clean string by converting to lowercase and removing spaces.
    
    Args:
        text: String to clean
        
    Returns:
        Cleaned string
    """
    if not isinstance(text, str):
        return ''
    return text.lower().replace(' ', '')
