import json
import csv

THRESHOLD = 0.00001

def is_correct(exec_func, testcases):

    for row in testcases:
        # Split inputs and expected output
        inputs = row[:-1]  # All columns except the last are inputs
        expected_output = row[-1]  # The last column is the expected output
        
        # Run the function with the inputs
        actual_output = exec_func(*inputs)

        #print(f"Inputs: {inputs}, Expected: {expected_output}, Actual: {actual_output}")
        
        # Check if the output matches the expected output
        if abs(actual_output - expected_output) > THRESHOLD:
            return False

    return True

def individual_to_function(individual, func_name="generated_function"):
   
    def parse_node(node):
        """Recursively parse the individual node into Python code."""
        if isinstance(node, list):
            # Handle single-element lists as constants
            if len(node) == 1 and not isinstance(node[0], list):
                return str(node[0])  # Treat as a constant
            
            operator = node[0]
            if operator == "+":
                return f"({parse_node(node[1])} + {parse_node(node[2])})"
            elif operator == "-":
                return f"({parse_node(node[1])} - {parse_node(node[2])})"
            elif operator == "*":
                return f"({parse_node(node[1])} * {parse_node(node[2])})"
            elif operator == "/":
                return f"({parse_node(node[1])} / ({parse_node(node[2])} if {parse_node(node[2])} != 0 else 1))"
            elif operator == "max":
                return f"max({parse_node(node[1])}, {parse_node(node[2])})"
            elif operator == "min":
                return f"min({parse_node(node[1])}, {parse_node(node[2])})"
            elif operator == "abs":
                return f"abs({parse_node(node[1])})"
            elif operator == "pow":
                return f"pow({parse_node(node[1])}, {parse_node(node[2])})"
            elif operator.startswith("x"):
                return f"x{operator[1:]}"  # Variable reference
            else:
                raise ValueError(f"Unknown operator: {operator}")
        else:
            return str(node)  # Treat constants as strings

    # Parse the individual into a Python expression
    expression = parse_node(individual)

    # Generate the full function string
    function_str = f"def {func_name}(*args):\n"
    function_str += f"    x0, x1 = args  # Adjust based on the number of variables\n"
    function_str += f"    return {expression}\n"

    #print(function_str)

    # Execute the function string to define the function
    local_namespace = {}
    exec(function_str, {}, local_namespace)

    return local_namespace[func_name]



def analyze_json(generations_file_path, csv_file_path):
  
    with open(generations_file_path, 'r') as file:
        data = json.load(file)
    
        # Count the number of keys in the dictionary
        generations = data["solution_values"]
        
        #take last generation
        last_generation = generations[-1]

        #get testcases
        testcases = []
        with open(csv_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                inputs = list(map(float, row[:-1]))  # Convert inputs to float
                expected_output = float(row[-1])
                inputs.append(expected_output)
                testcases.append(inputs)

        print(f"Number of test cases: {len(testcases)}")
        testcases=testcases[:1000]

        #check how many solutions are correct
        n_correct = 0
        for individual in last_generation:
            exec_func = individual_to_function(individual)
            # Validate the function with the test cases
            if is_correct(exec_func, testcases):
                n_correct+=1
        return n_correct


# Example usage
print(analyze_json("number_io_results_gp_2/alfa_ec_llm_solution_values.json", "tests/data/number_io_cases.csv"))