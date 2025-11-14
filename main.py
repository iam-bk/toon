from toon_format import encode, decode
import tiktoken
import os
import json


def encode_json_to_toon(json_data):
    """Encodes JSON data to Toon format."""
    return encode(json_data)


def decode_toon_to_json(toon_data):
    """Decodes Toon format data back to JSON."""
    return decode(toon_data)


def calculate_tokens(data: str) -> int:
    """Calculates the number of tokens in the given data using tiktoken."""

    encoder = tiktoken.encoding_for_model("gpt-5")
    tokens = encoder.encode(data)
    return len(tokens)


def compare_toon_and_json(json_data):
    """Compares the token count between Toon format and JSON format."""
    toon_data = encode_json_to_toon(json_data)

    json_str = str(json_data)

    toon_token_count = calculate_tokens(toon_data)
    json_token_count = calculate_tokens(json_str)

    return {
        "toon_token_count": toon_token_count,
        "json_token_count": json_token_count,
        "difference": json_token_count - toon_token_count,
    }


def print_results(results):
    print(
        f"{'Index':<5} {'File Name':<30} {'JSON Tokens':<15} {'Toon Tokens':<15} {'Difference':<12} {'Efficiency (%)':<15}"
    )
    print("-" * 100)
    for i, result in enumerate(results):
        print(
            f"{i:<5} {result['file_name']:<30} {result['json_token_count']:<15} {result['toon_token_count']:<15} {result['difference']:<12} {result['efficiency']:.2f}"
        )


def get_sample_files():
    return [file for file in os.listdir("./data") if file.endswith(".json")]


def read_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def compare_file_tokens(file_name, json_data):
    comparison = compare_toon_and_json(json_data)
    comparison["file_name"] = file_name
    comparison["efficiency"] = (
        (comparison["difference"] / comparison["json_token_count"]) * 100
        if comparison["json_token_count"] > 0
        else 0
    )
    return comparison


def main():
    results = []
    files = get_sample_files()
    for file in files:
        json_data = read_file(os.path.join("./data", file))
        result = compare_file_tokens(file, json_data)
        results.append(result)
    print_results(results)


if __name__ == "__main__":
    main()
