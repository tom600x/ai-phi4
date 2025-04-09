import json
import sys

try:
    # Read the raw dataset from a JSON file
    with open("plsqlpairs.json", "r") as file:
        content = file.read()
        # Try to fix JSON format by converting JSONL (JSON Lines) to a proper JSON array
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        raw_data = []
        for line in lines:
            try:
                # Parse each line as a separate JSON object
                obj = json.loads(line)
                raw_data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:50]}...")
                print(f"Error: {e}")
        
        if not raw_data:
            print("No valid JSON objects found in the file.")
            sys.exit(1)
        else:
            print(f"Successfully parsed {len(raw_data)} JSON objects.")

    # Convert to the fine-tuning format
    fine_tuning_data = []
    for item in raw_data:
        # Make sure we have all the needed fields
        if all(key in item for key in ['plsql', 'linq']):
            description = item.get('description', '')
            input_text = f"Convert the following PL/SQL query to LINQ: {item['plsql']} Description: {description}."
            output_text = item['linq']
            fine_tuning_data.append({"input": input_text, "output": output_text})

    # Save the preprocessed dataset
    with open("fine_tuning_dataset.json", "w") as f:
        json.dump(fine_tuning_data, f, indent=2)

    print(f"Dataset preprocessed and saved to 'fine_tuning_dataset.json' with {len(fine_tuning_data)} items.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)