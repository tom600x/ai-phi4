import json
import sys

try:
    # Read the raw dataset from a JSON file
    with open("plsqlpairs.json", "r") as file:
        content = file.read()
        # Process JSONL (JSON Lines) format
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

    # Convert to the Phi-4 fine-tuning format
    # For Phi-4, we use a specific format with "messages" containing "role" and "content"
    fine_tuning_data = []
    for item in raw_data:
        # Make sure we have all the needed fields
        if all(key in item for key in ['plsql', 'linq']):
            description = item.get('description', '')
            
            # Create conversation format for Phi-4
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Convert the following PL/SQL code to C# LINQ: \n\n```sql\n{item['plsql']}\n```\n\n{description if description else ''}"
                    },
                    {
                        "role": "assistant",
                        "content": f"Here's the equivalent C# LINQ code:\n\n```csharp\n{item['linq']}\n```"
                    }
                ]
            }
            
            fine_tuning_data.append(conversation)

    # Save the preprocessed dataset as a single JSON array (not JSONL)
    with open("phi4_fine_tuning_dataset.json", "w") as f:
        # Write the entire dataset as a single JSON array
        json.dump(fine_tuning_data, f, indent=2)

    print(f"Dataset preprocessed and saved to 'phi4_fine_tuning_dataset.json' with {len(fine_tuning_data)} items.")
    print("Format is compatible with Phi-4 fine-tuning requirements.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)