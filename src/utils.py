import os
import datetime
import json

def remove_empty_lines(raw_response: str, re_join:bool = True, max_line_length: int = 1024) -> str:
    """
    Remove empty lines from the string.
    """
    # Split the string into lines and filter out empty lines
    if isinstance(raw_response, str):
        raw_response = raw_response.strip()
        lines = raw_response.splitlines()
    elif isinstance(raw_response, list):
        lines = raw_response
    else:
        raise ValueError("raw_response must be a string or a list of strings")
    non_empty_lines = [line for line in lines if line.strip()]
    # split the lines that is too long
    non_empty_lines_ = []
    for line in non_empty_lines:
        if len(line) > max_line_length:
            # split the line into chunks of max_line_length
            chunks = [line[i:i + max_line_length] for i in range(0, len(line), max_line_length)]
            non_empty_lines_.extend(chunks)
        else:
            non_empty_lines_.append(line)
    non_empty_lines = non_empty_lines_
    # Join the non-empty lines back into a single string
    if re_join:
        raw_response = "\n".join(non_empty_lines)
        raw_response = raw_response.strip()
    else:
        raw_response = non_empty_lines
    return raw_response
  


def remove_think(raw_response: str) -> str:
    """
    Remove the 'thinking' part from the string.
    """
    # Remove "<think>" and "</think>" tags and their content
    initial_len = len(raw_response)
    i= 0
    while "<think>" in raw_response and "</think>" in raw_response:
        start = raw_response.find("<think>")
        end = raw_response.find("</think>") + len("</think>")
        raw_response = raw_response[:start] + raw_response[end:]
        i += 1
        if i > 100:
            break
    final_len = len(raw_response)
    return raw_response.strip()

def strip_any_unnecessary_chars_for_json(raw_response: str) -> str:
    # remove any unnecessary characters for json. remove from start to the first { or [
    # and from the last } or ] to the end
    # this is a bit of a hack, but it works for now
    first_brace = raw_response.find("{")
    first_bracket = raw_response.find("[")
    start = min(first_brace if first_brace != -1 else float('inf'),
                first_bracket if first_bracket != -1 else float('inf'))

    last_brace = raw_response.rfind("}")
    last_bracket = raw_response.rfind("]")
    end = max(last_brace if last_brace != -1 else -float('inf'),
              last_bracket if last_bracket != -1 else -float('inf'))

    if start == float('inf') or end == -float('inf'):
        raise ValueError("No valid JSON structure found in the response")

    return raw_response[start:end + 1]


def split_by_sharp(raw_response: str, split_by: str = "#") -> list[str]:
    """
    Split the string by the specified character (default is '#') and return a list of non-empty parts.
    """
    if not raw_response:
        return []
    # Split the string by the specified character
    parts = raw_response.split(split_by)
    # Remove leading and trailing whitespace from each part and filter out empty parts
    non_empty_parts = [part.strip() for part in parts if part.strip()]
    # remove any parts that is lesser than 3 lines
    non_empty_parts = [part for part in non_empty_parts if len(part.splitlines()) >= 3]
    return non_empty_parts


def log_response(user_message, assistant_response, log_file):
    """
    Log the user message and assistant response to a file.
    """
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# User message:\n")
        f.write(user_message + "\n\n")
        f.write("# Assistant response:\n")
        f.write(assistant_response + "\n\n")

def get_boolean_result_anyway(response: str, wanted_key: str) -> bool:
    """
    Get a boolean result from the raw response, regardless of its format.
    This function attempts to parse the response as JSON, and if that fails,
    it checks for the presence of the wanted key in the response.
    """
    try:
        # Try to parse the response as JSON
        json_response = json.loads(response)
        return json_response[wanted_key]
    except (json.JSONDecodeError, KeyError):
        response_lines = response.split("\n")
        response_line = [line for line in response_lines if line.find(wanted_key) != -1][0].lower()
        if "true" in response_line:
            return True
        elif "false" in response_line:
            return False
        else:
            # raise ValueError("No valid JSON structure found in the response: " + response)
            print(f"Warning: No valid JSON structure found in the response for key '{wanted_key}'. Returning False by default.")
            return False
        
def get_number_result_anyway(response: str, wanted_key: str) -> int:
    """
    Get a number result from the raw response, regardless of its format.
    This function attempts to parse the response as JSON, and if that fails,
    it checks for the presence of the wanted key in the response.
    """
    try:
        # Try to parse the response as JSON
        json_response = json.loads(response)
        return json_response[wanted_key]
    except (json.JSONDecodeError, KeyError):
        response_lines = response.split("\n")
        response_line = [line for line in response_lines if line.find(wanted_key) != -1][0].lower()
        try:
            return int(response_line)
        except ValueError:
            print(f"Warning: No valid JSON structure found in the response for key '{wanted_key}'. Returning 0 by default.")
            return 0

class NovelWritingLogger(object):
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """
        This is a singleton class. It ensures that only one instance of Logger exists.
        """
        if not cls._instance:
            cls._instance = super(NovelWritingLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_dir: str = "", name: str = "", debug: bool = True):

        self.start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name:
            name = f"Generation_{self.start_time}"
        self.name = name
        self.base_dir = base_dir if base_dir else os.path.join(os.getcwd(), self.name)

        self.log_file = os.path.join(self.base_dir, "logfile.txt")
        self.debug_dir = os.path.join(self.base_dir, "debug")
        self.step_counter = 0
        self.debug = debug

        # Create necessary directories
        os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize the log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Logger initialized at {self.start_time}\n")

    def log(self, module_name: str, message: str, user_message: str = None, assistant_response: str = None):
        """
        Log a message to the logfile with the module name and step count.
        """
        self.step_counter += 1
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[Step {self.step_counter:>4d}] {message}\n")
        print(f"[Step {self.step_counter:>4d}] {message}")
        if not self.debug:
            return
        if not user_message:
            user_message = "No user message provided."
        if not assistant_response:
            assistant_response = "No assistant response provided."
        debug_file_name = f"{self.step_counter:03d}_{module_name}.md"
        debug_file_path = os.path.join(self.debug_dir, debug_file_name)
        if os.path.exists(debug_file_path):
            os.rename(debug_file_path, debug_file_path.replace(".md", "_old.md", 1))
        with open(debug_file_path, "w", encoding="utf-8") as f:
            f.write(f"# User message:\n")
            f.write(user_message + "\n\n")
            f.write(f"# Assistant response:\n")
            f.write(assistant_response + "\n\n")